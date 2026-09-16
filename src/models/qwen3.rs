#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};
use image::{imageops::FilterType, DynamicImage};
use serde::Deserialize;
use std::io::Cursor;
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;

use crate::models::qwen3_vl::{Qwen3VLVisionModel, VisionConfig};

fn default_true() -> bool {
    true
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub head_dim: Option<usize>,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<serde_json::Value>,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub use_cache: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_window_layers: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(Deserialize)]
struct Qwen3VLConfig {
    text_config: Config,
}

fn parse_config_and_weight_prefix(config_bytes: &[u8]) -> Result<(Config, Option<&'static str>)> {
    if let Ok(cfg) = serde_json::from_slice::<Config>(config_bytes) {
        return Ok((cfg, None));
    }

    if let Ok(cfg) = serde_json::from_slice::<Qwen3VLConfig>(config_bytes) {
        return Ok((cfg.text_config, Some("model.language_model")));
    }

    Err(candle_core::Error::Msg(
        "Failed to parse config as Qwen3 or Qwen3-VL text config".into(),
    ))
}

#[derive(Deserialize, Debug, Clone)]
struct Qwen3VLFullConfig {
    text_config: Config,
    vision_config: VisionConfig,
    image_token_id: u32,
    #[allow(dead_code)]
    vision_start_token_id: u32,
    #[allow(dead_code)]
    vision_end_token_id: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct Qwen3VLPreprocessorConfig {
    min_pixels: usize,
    max_pixels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    rescale_factor: f32,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

struct PreparedImage {
    pixel_values: Vec<f32>,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    num_llm_tokens: usize,
}

fn scalar_f32(device: &Device, v: f32) -> Result<Tensor> {
    Tensor::from_slice(&[v], (1,), device)?.to_dtype(DType::F32)
}

fn scalar_f64_as_f32(device: &Device, v: f64) -> Result<Tensor> {
    scalar_f32(device, v as f32)
}

#[cfg(feature = "hf-hub")]
fn hf_repo(repo_id: &str) -> Result<ApiRepo> {
    crate::common::pull_from_hf(repo_id.to_string(), crate::get_cache_dir().into(), true)
        .map_err(map_err)
}

fn map_err<E: std::fmt::Display>(err: E) -> candle_core::Error {
    candle_core::Error::Msg(err.to_string())
}

fn build_attention_mask_4d(attention_mask_2d: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = attention_mask_2d.dims2()?;
    let device = attention_mask_2d.device();
    let mask_value = -1e4f32;

    let causal = {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[i * seq_len + j] = mask_value;
            }
        }
        Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?
    };

    let pad_mask_expanded = attention_mask_2d.unsqueeze(1)?.unsqueeze(2)?;
    let pad_mask_expanded = pad_mask_expanded.expand((batch_size, 1, seq_len, seq_len))?;
    let pad_mask_f32 = pad_mask_expanded.to_dtype(DType::F32)?;
    let ones = Tensor::ones_like(&pad_mask_f32)?;
    let inverted_mask = ones.sub(&pad_mask_f32)?;
    let mask_val_t = Tensor::new(&[mask_value], device)?;
    let pad_additive = inverted_mask.broadcast_mul(&mask_val_t)?;

    let causal_broadcast = causal.broadcast_as((batch_size, 1, seq_len, seq_len))?;
    causal_broadcast.add(&pad_additive)
}

fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let sum_sq = xs.sqr()?.sum_keepdim(1)?;
    let eps_tensor = Tensor::new(&[1e-12f32], xs.device())?
        .to_dtype(sum_sq.dtype())?
        .broadcast_as(sum_sq.shape())?;
    let norm = sum_sq.add(&eps_tensor)?.sqrt()?;
    xs.broadcast_div(&norm)
}

fn last_token_pool(hidden: &Tensor, attention_mask_2d: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, _) = hidden.dims3()?;
    let masks = attention_mask_2d.to_vec2::<f32>()?;
    if masks.len() != batch_size {
        return Err(candle_core::Error::Msg(
            "attention mask batch size mismatch".into(),
        ));
    }

    let mut pooled_rows = Vec::with_capacity(batch_size);
    for (batch_idx, row) in masks.iter().enumerate() {
        let last_idx = row.iter().rposition(|&v| v > 0.0).unwrap_or(seq_len - 1);
        pooled_rows.push(hidden.i((batch_idx, last_idx))?);
    }
    let pooled_refs: Vec<&Tensor> = pooled_rows.iter().collect();
    Tensor::stack(&pooled_refs, 0)
}

fn find_token_spans(ids: &[u32], token_id: u32) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == token_id {
            let start = i;
            while i < ids.len() && ids[i] == token_id {
                i += 1;
            }
            spans.push((start, i));
        } else {
            i += 1;
        }
    }
    spans
}

fn round_ties_to_even(value: f64) -> usize {
    let floor = value.floor();
    let frac = value - floor;
    if frac < 0.5 {
        floor as usize
    } else if frac > 0.5 {
        (floor + 1.0) as usize
    } else if (floor as i64) % 2 == 0 {
        floor as usize
    } else {
        (floor + 1.0) as usize
    }
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(candle_core::Error::Msg(
            "Image dimensions must be greater than zero".into(),
        ));
    }
    let aspect = (height.max(width) as f64) / (height.min(width) as f64);
    if aspect > 200.0 {
        return Err(candle_core::Error::Msg(
            "Absolute aspect ratio must be <= 200".into(),
        ));
    }

    // Match Python `round()` behavior used by qwen-vl-utils (ties-to-even).
    let mut h_bar = round_ties_to_even(height as f64 / factor as f64) * factor;
    let mut w_bar = round_ties_to_even(width as f64 / factor as f64) * factor;
    h_bar = h_bar.max(factor);
    w_bar = w_bar.max(factor);

    let area = (height * width) as f64;
    if h_bar * w_bar > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / area).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
    }

    Ok((h_bar, w_bar))
}

fn preprocess_image(
    img: &DynamicImage,
    cfg: &Qwen3VLPreprocessorConfig,
    max_image_tokens: usize,
) -> Result<PreparedImage> {
    if cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 {
        return Err(candle_core::Error::Msg(
            "Expected image_mean and image_std length to be 3".into(),
        ));
    }
    if cfg.patch_size == 0 || cfg.temporal_patch_size == 0 || cfg.merge_size == 0 {
        return Err(candle_core::Error::Msg(
            "patch_size, temporal_patch_size and merge_size must be > 0".into(),
        ));
    }

    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = rgb.dimensions();
    let factor = cfg.patch_size * cfg.merge_size;

    let effective_max_pixels = cfg.max_pixels.min(max_image_tokens * factor * factor);
    let (resized_h, resized_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        factor,
        cfg.min_pixels,
        effective_max_pixels,
    )?;

    let resized = image::imageops::resize(
        &rgb,
        resized_w as u32,
        resized_h as u32,
        FilterType::CatmullRom,
    );

    let grid_t = 1usize;
    let grid_h = resized_h / cfg.patch_size;
    let grid_w = resized_w / cfg.patch_size;
    let merge = cfg.merge_size;

    if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
        return Err(candle_core::Error::Msg(
            "grid_h and grid_w must be divisible by merge_size".into(),
        ));
    }

    let channels = 3usize;
    let patch_dim = channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let total_patch_tokens = grid_t * grid_h * grid_w;
    let mut out = Vec::with_capacity(total_patch_tokens * patch_dim);

    for t in 0..grid_t {
        let _ = t;
        for gh_block in 0..(grid_h / merge) {
            for gw_block in 0..(grid_w / merge) {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let gh = gh_block * merge + mh;
                        let gw = gw_block * merge + mw;

                        for c in 0..channels {
                            for _tp in 0..cfg.temporal_patch_size {
                                for ph in 0..cfg.patch_size {
                                    for pw in 0..cfg.patch_size {
                                        let y = gh * cfg.patch_size + ph;
                                        let x = gw * cfg.patch_size + pw;
                                        let pixel = resized.get_pixel(x as u32, y as u32).0[c];
                                        let mut value = pixel as f32;
                                        value *= cfg.rescale_factor;
                                        value = (value - cfg.image_mean[c]) / cfg.image_std[c];
                                        out.push(value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let num_llm_tokens = total_patch_tokens / (merge * merge);
    Ok(PreparedImage {
        pixel_values: out,
        grid_t: grid_t as u32,
        grid_h: grid_h as u32,
        grid_w: grid_w as u32,
        num_llm_tokens,
    })
}

fn build_vl_prompt(text: Option<&str>, include_image: bool, instruction: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(instruction);
    prompt.push_str("<|im_end|>\n<|im_start|>user\n");
    if include_image {
        prompt.push_str("<|vision_start|><|image_pad|><|vision_end|>");
    }
    if let Some(text) = text {
        prompt.push_str(text);
    }
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

fn expand_image_token_placeholders(prompt: &str, num_image_tokens: usize) -> Result<String> {
    if num_image_tokens == 0 {
        return Ok(prompt.to_string());
    }
    let image_token = "<|image_pad|>";
    if !prompt.contains(image_token) {
        return Err(candle_core::Error::Msg(
            "Prompt contains no <|image_pad|> placeholder".into(),
        ));
    }
    Ok(prompt.replacen(image_token, &image_token.repeat(num_image_tokens), 1))
}

fn build_image_position_ids(
    encodings: &[tokenizers::Encoding],
    image_spans_per_batch: &[Option<(usize, usize)>],
    prepared_images: &[Option<PreparedImage>],
    merge_size: usize,
    device: &Device,
) -> Result<Tensor> {
    if encodings.is_empty() {
        return Err(candle_core::Error::Msg(
            "encodings cannot be empty when building position ids".into(),
        ));
    }
    if encodings.len() != image_spans_per_batch.len() || encodings.len() != prepared_images.len() {
        return Err(candle_core::Error::Msg(
            "batch size mismatch while building position ids".into(),
        ));
    }

    let batch_size = encodings.len();
    let seq_len = encodings[0].len();
    let mut data = vec![1u32; 3 * batch_size * seq_len];
    let index = |dim: usize, batch: usize, pos: usize| -> usize {
        (dim * batch_size + batch) * seq_len + pos
    };

    for (batch_idx, encoding) in encodings.iter().enumerate() {
        let visible_len = encoding
            .get_attention_mask()
            .iter()
            .filter(|&&m| m != 0)
            .count();

        let Some((start, end)) = image_spans_per_batch[batch_idx] else {
            for pos in 0..visible_len {
                let val = pos as u32;
                data[index(0, batch_idx, pos)] = val;
                data[index(1, batch_idx, pos)] = val;
                data[index(2, batch_idx, pos)] = val;
            }
            continue;
        };

        let prepared = prepared_images[batch_idx].as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "Found image token span for a sample without prepared image".into(),
            )
        })?;
        if end > visible_len {
            return Err(candle_core::Error::Msg(
                "Image token span exceeds visible sequence length".into(),
            ));
        }

        let llm_t = prepared.grid_t as usize;
        let llm_h = prepared.grid_h as usize / merge_size;
        let llm_w = prepared.grid_w as usize / merge_size;
        let image_len = end - start;
        if image_len != llm_t * llm_h * llm_w {
            return Err(candle_core::Error::Msg(format!(
                "Image token span length {} does not match expected LLM grid {}x{}x{}",
                image_len, llm_t, llm_h, llm_w
            )));
        }

        for pos in 0..start {
            let val = pos as u32;
            data[index(0, batch_idx, pos)] = val;
            data[index(1, batch_idx, pos)] = val;
            data[index(2, batch_idx, pos)] = val;
        }

        let mut seq_pos = start;
        for t in 0..llm_t {
            for h in 0..llm_h {
                for w in 0..llm_w {
                    data[index(0, batch_idx, seq_pos)] = (start + t) as u32;
                    data[index(1, batch_idx, seq_pos)] = (start + h) as u32;
                    data[index(2, batch_idx, seq_pos)] = (start + w) as u32;
                    seq_pos += 1;
                }
            }
        }
        if seq_pos != end {
            return Err(candle_core::Error::Msg(
                "Image token position construction consumed the wrong number of tokens".into(),
            ));
        }

        let prefix_max = start.saturating_sub(1);
        let image_max = start + llm_t.max(llm_h).max(llm_w).saturating_sub(1);
        let st_idx = prefix_max.max(image_max) + 1;
        for offset in 0..(visible_len - end) {
            let pos = end + offset;
            let val = (st_idx + offset) as u32;
            data[index(0, batch_idx, pos)] = val;
            data[index(1, batch_idx, pos)] = val;
            data[index(2, batch_idx, pos)] = val;
        }
    }

    Tensor::from_vec(data, (3, batch_size, seq_len), device)
}

fn load_image_from_path(path: &Path) -> Result<DynamicImage> {
    image::ImageReader::open(path)
        .map_err(map_err)?
        .decode()
        .map_err(map_err)
}

pub struct Qwen3RMSNorm {
    weight: Tensor, // [dim]
    eps: f64,
}

impl Qwen3RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Qwen3RMSNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let dev = xs.device();

        let xs_f = xs.to_dtype(DType::F32)?;
        let var = xs_f.powf(2.0)?.mean_keepdim(D::Minus1)?; // [..., 1]

        let eps_t = scalar_f64_as_f32(dev, self.eps)?;
        let var_eps = var.broadcast_add(&eps_t)?;

        // xs * rsqrt(var + eps)  == xs * (1/sqrt(...))
        let inv_rms = var_eps.sqrt()?.recip()?; // [..., 1]
        let normed = xs_f.broadcast_mul(&inv_rms)?; // [..., dim]

        // weight * back_to_input_dtype
        let normed = normed.to_dtype(in_dtype)?;
        let w = self.weight.to_dtype(in_dtype)?;
        normed.broadcast_mul(&w)
    }
}

pub struct Qwen3MLP {
    gate_proj: Linear, // hidden -> intermediate
    up_proj: Linear,   // hidden -> intermediate
    down_proj: Linear, // intermediate -> hidden
    act_fn: Activation,
}

impl Qwen3MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct Qwen3RotaryEmbedding {
    inv_freq: Tensor,      // [dim/2] f32
    attention_factor: f32, // HF: attention_scaling (1.0 for default)
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
}

impl Qwen3RotaryEmbedding {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let base = cfg.rope_theta; // f64 for precision
        let dim = cfg.head_dim(); // head_dim
        assert!(dim.is_multiple_of(2), "head_dim must be even, got {dim}");

        // t = [0,2,4,...,dim-2] as f32
        let t = Tensor::arange_step(0u32, dim as u32, 2u32, device)?.to_dtype(DType::F32)?;

        // exponent = t / dim
        let dim_t = scalar_f32(device, dim as f32)?;
        let exponent = t.broadcast_div(&dim_t)?; // [dim/2]

        // inv_freq = 1 / (base ** exponent) = exp(-ln(base) * exponent)
        let ln_base = (base as f32).ln();
        let ln_base_t = scalar_f32(device, ln_base)?;
        let inv_freq = exponent.broadcast_mul(&ln_base_t.neg()?)?.exp()?; // [dim/2]

        let mut mrope_interleaved = false;
        let mut mrope_section = [24usize, 20usize, 20usize];
        if let Some(rope_scaling) = cfg.rope_scaling.as_ref() {
            if let Some(v) = rope_scaling
                .get("mrope_interleaved")
                .and_then(|v| v.as_bool())
            {
                mrope_interleaved = v;
            }
            if let Some(arr) = rope_scaling.get("mrope_section").and_then(|v| v.as_array()) {
                if arr.len() == 3 {
                    mrope_section = [
                        arr[0].as_u64().unwrap_or(mrope_section[0] as u64) as usize,
                        arr[1].as_u64().unwrap_or(mrope_section[1] as u64) as usize,
                        arr[2].as_u64().unwrap_or(mrope_section[2] as u64) as usize,
                    ];
                }
            }
        }

        Ok(Self {
            inv_freq,
            attention_factor: 1.0,
            mrope_interleaved,
            mrope_section,
        })
    }

    /// position_ids: [B,T] or [3,B,T] (MRoPE)
    /// returns (cos, sin): [B,T,dim] in xs.dtype()
    pub fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = xs.device();
        let inv_freq = self.inv_freq.to_device(dev)?.to_dtype(DType::F32)?;
        let d2 = inv_freq.dims1()?;

        let freqs = if position_ids.rank() == 2 {
            let (b, t) = position_ids.dims2()?;
            let pos = position_ids
                .to_device(dev)?
                .to_dtype(DType::F32)?
                .contiguous()?;

            let inv_freq_expanded = inv_freq
                .reshape((1, d2, 1))?
                .expand((b, d2, 1))?
                .contiguous()?;
            let pos_expanded = pos.reshape((b, 1, t))?.contiguous()?;

            inv_freq_expanded
                .matmul(&pos_expanded)?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            let (dims, b, t) = position_ids.dims3()?;
            if dims != 3 {
                return Err(candle_core::Error::Msg(
                    "Expected position_ids first dimension to be 3 for MRoPE".into(),
                ));
            }

            let inv = inv_freq.to_vec1::<f32>()?;
            let pos = position_ids.to_device(dev)?.to_vec3::<u32>()?;
            let mut freqs = vec![0f32; b * t * d2];

            #[allow(clippy::needless_range_loop)]
            for batch_idx in 0..b {
                for tok_idx in 0..t {
                    let base = (batch_idx * t + tok_idx) * d2;
                    let temporal = pos[0][batch_idx][tok_idx] as f32;
                    for i in 0..d2 {
                        freqs[base + i] = temporal * inv[i];
                    }

                    if self.mrope_interleaved {
                        for (dim, _) in pos.iter().enumerate().take(2 + 1).skip(1) {
                            let pos_dim = pos[dim][batch_idx][tok_idx] as f32;
                            let mut i = dim;
                            let limit = (self.mrope_section[dim] * 3).min(d2);
                            while i < limit {
                                freqs[base + i] = pos_dim * inv[i];
                                i += 3;
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(freqs, (b, t, d2), dev)?
        };

        // emb = cat(freqs, freqs) -> [B,T,dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 2)?;

        let scale = scalar_f32(dev, self.attention_factor)?;
        let cos = emb.cos()?.broadcast_mul(&scale)?;
        let sin = emb.sin()?.broadcast_mul(&scale)?;

        let out_dtype = xs.dtype();
        Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
    }
}

// rotate_half + apply_rotary_pos_emb (HF)
// rotate_half: cat(-x2, x1) along last dim (half-split style)
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x
        .dims()
        .last()
        .copied()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".into()))?;
    assert!(d % 2 == 0, "rotate_half requires even last dim, got {d}");
    let half = d / 2;

    // x1 = x[..., :half], x2 = x[..., half:]
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let nx2 = x2.neg()?;
    Tensor::cat(&[&nx2, &x1], x.rank() - 1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,   // [B, Hq, T, D]
    k: &Tensor,   // [B, Hk, T, D]
    cos: &Tensor, // [B, T, D]
    sin: &Tensor, // [B, T, D]
) -> Result<(Tensor, Tensor)> {
    // unsqueeze_dim=1 -> [B,1,T,D] then broadcast
    let cos_u = cos.unsqueeze(1)?;
    let sin_u = sin.unsqueeze(1)?;

    let q_embed = (q.broadcast_mul(&cos_u)? + rotate_half(q)?.broadcast_mul(&sin_u)?)?;
    let k_embed = (k.broadcast_mul(&cos_u)? + rotate_half(k)?.broadcast_mul(&sin_u)?)?;
    Ok((q_embed, k_embed))
}

// repeat_kv (HF): [B, Nkv, T, D] -> [B, Nh, T, D]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, t, d) = x.dims4()?;
    // [B, Nkv, 1, T, D]
    let x = x.unsqueeze(2)?;
    // broadcast to [B, Nkv, n_rep, T, D]
    let x = x.broadcast_as((b, n_kv, n_rep, t, d))?;
    // reshape to [B, Nkv*n_rep, T, D]
    x.reshape((b, n_kv * n_rep, t, d))
}

// - q_norm/k_norm on head_dim after reshape
// - RoPE on q,k
// - repeat_kv for key/value (GQA)
pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3RMSNorm, // dim = head_dim
    k_norm: Qwen3RMSNorm, // dim = head_dim

    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f32,
}

impl Qwen3Attention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = cfg.num_kv_groups();
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "num_heads must be multiple of num_kv_heads"
        );

        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let q_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, q_out, vb.pp("q_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?
        };
        let k_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, kv_out, vb.pp("k_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?
        };
        let v_proj = if cfg.attention_bias {
            linear(cfg.hidden_size, kv_out, vb.pp("v_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?
        };
        let o_proj = if cfg.attention_bias {
            linear(q_out, cfg.hidden_size, vb.pp("o_proj"))?
        } else {
            linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj"))?
        };

        // q_norm/k_norm are RMSNorm(head_dim)
        let q_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// hidden_states: [B,T,H]
    /// position_embeddings: (cos,sin) both [B,T,D]
    /// attention_mask: additive mask [B,1,T,T] (0 or -inf)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _h) = hidden_states.dims3()?;
        let d = self.head_dim;

        // hidden_shape = [B,T,-1,D]
        // q: [B,T,Nh*D] -> [B,T,Nh,D] -> q_norm -> transpose -> [B,Nh,T,D]
        let q = hidden_states
            .apply(&self.q_proj)?
            .reshape((b, t, self.num_heads, d))?;
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;

        // k: [B,T,Nkv*D] -> [B,T,Nkv,D] -> k_norm -> [B,Nkv,T,D]
        let k = hidden_states
            .apply(&self.k_proj)?
            .reshape((b, t, self.num_kv_heads, d))?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;

        // v: [B,Nkv,T,D]
        let v = hidden_states
            .apply(&self.v_proj)?
            .reshape((b, t, self.num_kv_heads, d))?
            .transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // GQA expand
        let k = repeat_kv(&k, self.num_kv_groups)?; // [B,Nh,T,D]
        let v = repeat_kv(&v, self.num_kv_groups)?; // [B,Nh,T,D]

        // attn_weights = q @ k^T * scaling
        let kt = k.transpose(2, 3)?; // [B,Nh,D,T]
        let mut attn = q.matmul(&kt)?; // [B,Nh,T,T]

        let scale = scalar_f32(attn.device(), self.scaling)?.to_dtype(attn.dtype())?;
        attn = attn.broadcast_mul(&scale)?;

        if let Some(mask) = attention_mask {
            attn = attn.broadcast_add(&mask.to_dtype(attn.dtype())?)?;
        }

        // softmax over last dim
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;

        // attn_output = attn @ v -> [B,Nh,T,D]
        let out = attn.matmul(&v)?;
        // transpose back -> [B,T,Nh,D] -> reshape [B,T,Nh*D] -> o_proj -> [B,T,H]
        let out = out.transpose(1, 2)?.reshape((b, t, self.num_heads * d))?;
        out.apply(&self.o_proj)
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,          // dim=hidden_size
    post_attention_layernorm: Qwen3RMSNorm, // dim=hidden_size
}

impl Qwen3DecoderLayer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_embeddings: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        // Pre-norm
        let residual = hidden_states.clone();
        let hs = hidden_states.apply(&self.input_layernorm)?;
        let hs = self
            .self_attn
            .forward(&hs, position_embeddings, attention_mask)?;
        let hs = (residual + hs)?;

        // MLP
        let residual = hs.clone();
        let hs2 = hs.apply(&self.post_attention_layernorm)?;
        let hs2 = hs2.apply(&self.mlp)?;
        residual + hs2
    }
}

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    cfg: Config,
    device: Device,
}

impl Qwen3Model {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(&cfg, vb.pp(format!("layers.{i}")))?);
        }

        let norm = Qwen3RMSNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = Qwen3RotaryEmbedding::new(&cfg, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            cfg,
            device,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_with_inputs_embeds(
        &self,
        inputs_embeds: &Tensor,
        attention_mask_4d: Option<&Tensor>,
        deepstack_additions: Option<&[Tensor]>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = inputs_embeds.dims3()?;
        let mut hs = inputs_embeds.clone();

        let default_position_ids = if position_ids.is_none() {
            let pos_1d = Tensor::arange(0u32, t as u32, hs.device())?;
            Some(pos_1d.unsqueeze(0)?.expand((b, t))?.contiguous()?)
        } else {
            None
        };
        let position_ids = if let Some(position_ids) = position_ids {
            position_ids
        } else {
            default_position_ids.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("missing default position ids".to_string())
            })?
        };

        // position_embeddings = (cos,sin) once
        let (cos, sin) = self.rotary_emb.forward(&hs, position_ids)?;

        // layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hs = layer.forward(&hs, attention_mask_4d, (&cos, &sin))?;
            if let Some(additions) = deepstack_additions {
                if let Some(visual_add) = additions.get(layer_idx) {
                    hs = hs.add(visual_add)?;
                }
            }
        }

        // final norm
        hs.apply(&self.norm)
    }

    /// input_ids: [B,T]
    /// attention_mask: optionally [B,T] (1=token,0=pad) OR already expanded additive mask [B,1,T,T]
    /// Here we assume additive mask [B,1,T,T] for simplicity (HF does that).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask_4d: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hs = self.embed_tokens(input_ids)?;
        self.forward_with_inputs_embeds(&hs, attention_mask_4d, None, None)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// High-level embedding wrapper around [`Qwen3Model`] that handles tokenization,
/// batching, mean-pooling, and L2-normalization.
pub struct Qwen3TextEmbedding {
    model: Qwen3Model,
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3TextEmbedding {
    /// Build from an already loaded model and tokenizer.
    pub fn new(model: Qwen3Model, tokenizer: tokenizers::Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load from a Hugging Face repo.
    ///
    /// Supported model families:
    /// - Qwen3 text embedding checkpoints (e.g. `Qwen/Qwen3-Embedding-0.6B`)
    /// - Qwen3-VL embedding checkpoints in text-only mode
    ///   (e.g. `Qwen/Qwen3-VL-Embedding-2B`)
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let repo = hf_repo(repo_id)?;

        // Load config
        let cfg_path: PathBuf = repo
            .get("config.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cfg_bytes =
            std::fs::read(&cfg_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let (cfg, weight_prefix) = parse_config_and_weight_prefix(&cfg_bytes)?;

        // Load weights (single or sharded)
        let weight_files: Vec<PathBuf> = crate::common::safetensors_weight_files(&repo)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let vb = match weight_prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let model = Qwen3Model::new(cfg, vb)?;

        // Load tokenizer
        let tok_path: PathBuf = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Use LEFT padding so the last position is always the real last token.
        // This simplifies last-token pooling and matches the official implementation.
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        // Note: Unlike some other embedding models, Qwen3-Embedding does NOT
        // add an EOS token. The last_token_pool simply takes the last real token.

        Ok(Self { model, tokenizer })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Embed a batch of texts, returning normalized embeddings.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(|s| s.as_ref()).collect::<Vec<_>>(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        let mut input_ids_vec: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            input_ids_vec.extend(enc.get_ids().iter().copied());
            attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
        }

        let device = self.model.device();
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;

        let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)?;

        // Forward pass -> [B, T, H]
        let hidden = self.model.forward(&input_ids, Some(&attention_mask_4d))?;

        // Last token pooling: with left padding, the last position is always the real last token
        let pooled = hidden.i((.., seq_len - 1))?; // [B, H]

        // L2 normalize (add epsilon for numerical stability)
        let normalized = l2_normalize(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let normalized = normalized.to_dtype(DType::F32)?;
        let data = normalized.to_vec2::<f32>()?;
        Ok(data)
    }
}

/// Multimodal embedding wrapper for Qwen3-VL embedding checkpoints.
///
/// This supports text-only and image-only inputs. Image+text mixed inputs can be
/// added later with the same backend.
pub struct Qwen3VLEmbedding {
    model: Qwen3Model,
    vision: Qwen3VLVisionModel,
    tokenizer: tokenizers::Tokenizer,
    preprocessor: Qwen3VLPreprocessorConfig,
    image_token_id: u32,
    default_instruction: String,
    max_image_tokens: usize,
}

impl Qwen3VLEmbedding {
    /// Load a Qwen3-VL embedding model from Hugging Face.
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams};

        let repo = hf_repo(repo_id)?;

        let cfg_path: PathBuf = repo.get("config.json").map_err(map_err)?;
        let cfg_bytes = std::fs::read(&cfg_path).map_err(map_err)?;
        let cfg: Qwen3VLFullConfig = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let preproc_path: PathBuf = repo.get("preprocessor_config.json").map_err(map_err)?;
        let preprocessor_bytes = std::fs::read(&preproc_path).map_err(map_err)?;
        let preprocessor: Qwen3VLPreprocessorConfig =
            serde_json::from_slice(&preprocessor_bytes).map_err(map_err)?;

        let weight_files: Vec<PathBuf> = crate::common::safetensors_weight_files(&repo)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = Qwen3Model::new(cfg.text_config, vb.pp("model").pp("language_model"))?;
        let vision = Qwen3VLVisionModel::new(&cfg.vision_config, vb.pp("model").pp("visual"))?;

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path).map_err(map_err)?;
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        let max_image_tokens = {
            let base_prompt = build_vl_prompt(None, true, "Represent the user's input.");
            let enc = tokenizer.encode(base_prompt, false).map_err(map_err)?;
            let img_count = enc
                .get_ids()
                .iter()
                .filter(|&&id| id == cfg.image_token_id)
                .count();
            let overhead = enc.get_ids().len() - img_count;
            max_length.saturating_sub(overhead)
        };

        Ok(Self {
            model,
            vision,
            tokenizer,
            preprocessor,
            image_token_id: cfg.image_token_id,
            default_instruction: "Represent the user's input.".to_string(),
            max_image_tokens,
        })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Embed a batch of texts using Qwen3-VL prompt formatting.
    pub fn embed_texts<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        let text_inputs: Vec<Option<String>> =
            texts.iter().map(|t| Some(t.as_ref().to_string())).collect();
        let image_inputs: Vec<Option<DynamicImage>> = (0..texts.len()).map(|_| None).collect();
        self.embed_internal(text_inputs, image_inputs)
    }

    /// Embed a batch of image paths.
    pub fn embed_images<S: AsRef<Path>>(&self, images: &[S]) -> Result<Vec<Vec<f32>>> {
        let mut image_inputs = Vec::with_capacity(images.len());
        for path in images {
            image_inputs.push(Some(load_image_from_path(path.as_ref())?));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();
        self.embed_internal(text_inputs, image_inputs)
    }

    /// Embed a batch of image bytes.
    pub fn embed_image_bytes(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        let mut image_inputs = Vec::with_capacity(images.len());
        for bytes in images {
            let image = image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .map_err(map_err)?
                .decode()
                .map_err(map_err)?;
            image_inputs.push(Some(image));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();
        self.embed_internal(text_inputs, image_inputs)
    }

    fn embed_internal(
        &self,
        texts: Vec<Option<String>>,
        images: Vec<Option<DynamicImage>>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.len() != images.len() {
            return Err(candle_core::Error::Msg(
                "texts and images must have the same batch size".into(),
            ));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut prepared_images = Vec::with_capacity(images.len());
        for image in &images {
            prepared_images.push(match image {
                Some(img) => Some(preprocess_image(
                    img,
                    &self.preprocessor,
                    self.max_image_tokens,
                )?),
                None => None,
            });
        }

        let mut prompts = Vec::with_capacity(texts.len());
        for (text, prepared) in texts.iter().zip(prepared_images.iter()) {
            let mut prompt = build_vl_prompt(
                text.as_deref(),
                prepared.is_some(),
                &self.default_instruction,
            );
            if let Some(prepared) = prepared {
                prompt = expand_image_token_placeholders(&prompt, prepared.num_llm_tokens)?;
            }
            prompts.push(prompt);
        }

        let encodings = self
            .tokenizer
            .encode_batch(prompts, true)
            .map_err(map_err)?;
        let batch_size = encodings.len();
        let seq_len = encodings[0].len();
        let mut image_spans_per_batch = Vec::with_capacity(batch_size);
        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let Some(prepared) = prepared_images[batch_idx].as_ref() else {
                image_spans_per_batch.push(None);
                continue;
            };

            let spans = find_token_spans(encoding.get_ids(), self.image_token_id);
            if spans.len() != 1 {
                return Err(candle_core::Error::Msg(
                    "Expected exactly one image token span per image input".into(),
                ));
            }
            let (start, end) = spans[0];
            let span_len = end - start;
            if span_len != prepared.num_llm_tokens {
                return Err(candle_core::Error::Msg(format!(
                    "Image token span mismatch: prompt has {}, preprocessor expects {}",
                    span_len, prepared.num_llm_tokens
                )));
            }
            image_spans_per_batch.push(Some((start, end)));
        }

        let mut input_ids_vec = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len);
        for encoding in &encodings {
            input_ids_vec.extend(encoding.get_ids().iter().copied());
            attention_mask_vec.extend(encoding.get_attention_mask().iter().map(|&m| m as f32));
        }

        let device = self.model.device();
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;
        let mut inputs_embeds = self.model.embed_tokens(&input_ids)?;
        let hidden_size = self.model.config().hidden_size;
        let mut deepstack_additions: Option<Vec<Tensor>> = None;
        let mut position_ids: Option<Tensor> = None;

        let num_images = prepared_images.iter().filter(|p| p.is_some()).count();
        if num_images > 0 {
            let patch_dim = 3
                * self.preprocessor.temporal_patch_size
                * self.preprocessor.patch_size
                * self.preprocessor.patch_size;

            let mut pixel_values = Vec::new();
            let mut grid_thw = Vec::new();
            for prepared in prepared_images.iter().flatten() {
                pixel_values.extend_from_slice(&prepared.pixel_values);
                grid_thw.extend_from_slice(&[prepared.grid_t, prepared.grid_h, prepared.grid_w]);
            }

            let num_patch_tokens = pixel_values.len() / patch_dim;
            let pixel_values =
                Tensor::from_vec(pixel_values, (num_patch_tokens, patch_dim), device)?;
            let image_grid_thw = Tensor::from_vec(grid_thw, (num_images, 3), device)?;
            position_ids = Some(build_image_position_ids(
                &encodings,
                &image_spans_per_batch,
                &prepared_images,
                self.preprocessor.merge_size,
                device,
            )?);

            let (image_embeds, deepstack_image_embeds) =
                self.vision.forward(&pixel_values, &image_grid_thw)?;
            let mut offset = 0usize;

            for (batch_idx, image_span) in image_spans_per_batch.iter().enumerate() {
                let Some((start, end)) = image_span else {
                    continue;
                };
                let span_len = end - start;

                let image_chunk = image_embeds.narrow(0, offset, span_len)?;
                offset += span_len;
                inputs_embeds = inputs_embeds.slice_assign(
                    &[batch_idx..batch_idx + 1, *start..*end, 0..hidden_size],
                    &image_chunk.unsqueeze(0)?,
                )?;
            }

            if offset != image_embeds.dim(0)? {
                return Err(candle_core::Error::Msg(
                    "Unconsumed image embeddings remain after token injection".into(),
                ));
            }

            if !deepstack_image_embeds.is_empty() {
                let mut per_layer_additions = Vec::with_capacity(deepstack_image_embeds.len());
                for deepstack_layer in deepstack_image_embeds {
                    let mut addition = Tensor::zeros(
                        (batch_size, seq_len, hidden_size),
                        deepstack_layer.dtype(),
                        device,
                    )?;
                    let mut deep_offset = 0usize;
                    for (batch_idx, image_span) in image_spans_per_batch.iter().enumerate() {
                        let Some((start, end)) = image_span else {
                            continue;
                        };
                        let span_len = end - start;
                        let chunk = deepstack_layer.narrow(0, deep_offset, span_len)?;
                        deep_offset += span_len;
                        addition = addition.slice_assign(
                            &[batch_idx..batch_idx + 1, *start..*end, 0..hidden_size],
                            &chunk.unsqueeze(0)?,
                        )?;
                    }
                    if deep_offset != deepstack_layer.dim(0)? {
                        return Err(candle_core::Error::Msg(
                            "Unconsumed deepstack image embeddings remain after token injection"
                                .into(),
                        ));
                    }
                    per_layer_additions.push(addition);
                }
                deepstack_additions = Some(per_layer_additions);
            }
        }

        let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)?;
        let hidden = self.model.forward_with_inputs_embeds(
            &inputs_embeds,
            Some(&attention_mask_4d),
            deepstack_additions.as_deref(),
            position_ids.as_ref(),
        )?;
        let pooled = last_token_pool(&hidden, &attention_mask_2d)?;
        let normalized = l2_normalize(&pooled)?.to_dtype(DType::F32)?;
        normalized.to_vec2::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expand_image_token_placeholders, find_token_spans, parse_config_and_weight_prefix,
        round_ties_to_even,
    };

    #[test]
    fn mrope_matches_plain_rope_for_identical_positions() {
        use super::Qwen3RotaryEmbedding;
        use candle_core::{DType, Device, Tensor};

        let config = r#"{
            "attention_bias": false,
            "attention_dropout": 0.0,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": false},
            "rope_theta": 10000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "vocab_size": 1000
        }"#;
        let (cfg, _) = parse_config_and_weight_prefix(config.as_bytes()).unwrap();
        let device = Device::Cpu;
        let rotary = Qwen3RotaryEmbedding::new(&cfg, &device).unwrap();

        let (batch, tokens) = (2usize, 9usize);
        let xs = Tensor::zeros((batch, tokens, cfg.head_dim()), DType::F32, &device).unwrap();
        let positions: Vec<u32> = (0..batch).flat_map(|_| 0..tokens as u32).collect();
        let plain = Tensor::from_vec(positions.clone(), (batch, tokens), &device).unwrap();
        let mrope = Tensor::from_vec(
            [positions.clone(), positions.clone(), positions].concat(),
            (3, batch, tokens),
            &device,
        )
        .unwrap();

        let (cos_plain, sin_plain) = rotary.forward(&xs, &plain).unwrap();
        let (cos_mrope, sin_mrope) = rotary.forward(&xs, &mrope).unwrap();

        let max_abs_diff = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        assert!(max_abs_diff(&cos_plain, &cos_mrope) < 1e-5, "cos differs");
        assert!(max_abs_diff(&sin_plain, &sin_mrope) < 1e-5, "sin differs");

        let late_cos = cos_mrope
            .get(0)
            .unwrap()
            .get(tokens - 1)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(late_cos.iter().any(|&c| (c - 1.0).abs() > 1e-3));
    }

    #[test]
    fn parses_qwen3_config_without_prefix() {
        let config = r#"{
            "attention_bias": false,
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": null,
            "rope_theta": 1000000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151669
        }"#;

        let (cfg, prefix) = parse_config_and_weight_prefix(config.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(prefix, None);
    }

    #[test]
    fn parses_qwen3_vl_text_config_with_language_model_prefix() {
        let config = r#"{
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 262144,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                },
                "rope_theta": 5000000,
                "tie_word_embeddings": true,
                "use_cache": true,
                "vocab_size": 151936
            },
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653
        }"#;

        let (cfg, prefix) = parse_config_and_weight_prefix(config.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(prefix, Some("model.language_model"));
    }

    #[test]
    fn finds_token_spans() {
        let ids = vec![1u32, 42, 42, 7, 42, 42, 42, 9];
        let spans = find_token_spans(&ids, 42);
        assert_eq!(spans, vec![(1, 3), (4, 7)]);
    }

    #[test]
    fn expands_image_placeholders() {
        let prompt = "<|vision_start|><|image_pad|><|vision_end|>";
        let expanded = expand_image_token_placeholders(prompt, 3).unwrap();
        assert_eq!(
            expanded,
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[test]
    fn rounds_half_to_even_like_python() {
        assert_eq!(round_ties_to_even(12.5), 12);
        assert_eq!(round_ties_to_even(13.5), 14);
        assert_eq!(round_ties_to_even(9.5625), 10);
    }
}
