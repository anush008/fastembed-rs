#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};
#[cfg(feature = "qwen3-vl")]
use image::{imageops::FilterType, DynamicImage};
use serde::Deserialize;
#[cfg(feature = "qwen3-vl")]
use std::io::Cursor;
use std::io::{Read, Seek};
use std::path::Path;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;

#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;

use crate::models::qwen3_qk_norm_rope::rms_norm_rope_thd;
use crate::models::qwen3_sdpa_bthd::sdpa_bthd;
use crate::models::qwen3_silu_mul::silu_mul;
#[cfg(feature = "qwen3-vl")]
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

#[cfg(any(feature = "hf-hub", test))]
#[derive(Deserialize)]
struct Qwen3VLConfig {
    text_config: Config,
}

#[cfg(any(feature = "hf-hub", test))]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

/// The mask `build_attention_mask_4d` builds, `[B, 1, T, T]` (causal plus left padding), in `dtype`,
/// for the fused attention kernel, which reads its mask in the model's dtype. It is one broadcast
/// add of a `[T, T]` causal part and a `[B, T]` padding part, so no other `[B, 1, T, T]` tensor is
/// built on the way.
fn build_attention_mask_fused(attention_mask_2d: &Tensor, dtype: DType) -> Result<Tensor> {
    let (batch_size, seq_len) = attention_mask_2d.dims2()?;
    let device = attention_mask_2d.device();
    let mask_value = -1e4f64;
    let positions = Tensor::arange(0f32, seq_len as f32, device)?;
    let causal = positions
        .reshape((1, seq_len))?
        .broadcast_gt(&positions.reshape((seq_len, 1))?)?
        .to_dtype(DType::F32)?
        .affine(mask_value, 0.0)?
        .to_dtype(dtype)?
        .reshape((1, 1, seq_len, seq_len))?;
    // (1 - mask) * mask_value: 0 on a token, mask_value on padding.
    let padding = attention_mask_2d
        .to_dtype(DType::F32)?
        .affine(-mask_value, mask_value)?
        .to_dtype(dtype)?
        .reshape((batch_size, 1, 1, seq_len))?;
    causal.broadcast_add(&padding)
}

/// Each row of `xs` `[B, H]` divided by its L2 norm, in f32 whatever dtype `xs` is in: in bf16
/// the sum of squares, the norm and every quotient would each round to 8 bits, and the vectors
/// would come back up to about 0.5% off unit length.
fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.to_dtype(DType::F32)?;
    let sum_sq = xs.sqr()?.sum_keepdim(1)?;
    let eps_tensor = Tensor::new(&[1e-12f32], xs.device())?.broadcast_as(sum_sq.shape())?;
    let norm = sum_sq.add(&eps_tensor)?.sqrt()?;
    xs.broadcast_div(&norm)
}

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
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

#[cfg(feature = "qwen3-vl")]
fn load_image_from_path(path: &Path) -> Result<DynamicImage> {
    image::ImageReader::open(path)
        .map_err(map_err)?
        .decode()
        .map_err(map_err)
}

pub struct Qwen3RMSNorm {
    weight: Tensor, // [dim], in the dtype the weights were loaded in
    eps: f64,
}

impl Qwen3RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

/// A dense matrix from safetensors, or a quantized one from a GGUF file.
pub enum Projection {
    Dense(Linear),
    Quantized(QMatMul),
}

impl Module for Projection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Projection::Dense(linear) => linear.forward(xs),
            // candle's quantized matmul rejects strided input, and off the CPU it also ignores a
            // view's start offset.
            Projection::Quantized(matrix)
                if xs.is_contiguous()
                    && (xs.layout().start_offset() == 0 || xs.device().is_cpu()) =>
            {
                matrix.forward(xs)
            }
            Projection::Quantized(matrix) => matrix.forward(&xs.force_contiguous()?),
        }
    }
}

struct GgufWeights<'a, R: Read + Seek> {
    content: &'a gguf_file::Content,
    reader: &'a mut R,
    device: &'a Device,
}

impl<R: Read + Seek> GgufWeights<'_, R> {
    fn tensor(&mut self, name: &str, dims: &[usize], device: &Device) -> Result<QTensor> {
        let tensor = self.content.tensor(self.reader, name, device)?;
        if tensor.shape().dims() != dims {
            candle_core::bail!(
                "GGUF tensor `{name}` has shape {:?}, expected {dims:?}",
                tensor.shape().dims()
            );
        }
        Ok(tensor)
    }

    fn projection(&mut self, name: &str, out_dim: usize, in_dim: usize) -> Result<Projection> {
        let device = self.device;
        let matrix = self.tensor(&format!("{name}.weight"), &[out_dim, in_dim], device)?;
        Ok(Projection::Quantized(QMatMul::from_qtensor(matrix)?))
    }

    fn norm(&mut self, name: &str, dim: usize, eps: f64) -> Result<Qwen3RMSNorm> {
        let device = self.device;
        let weight = self
            .tensor(&format!("{name}.weight"), &[dim], device)?
            .dequantize(device)?;
        Ok(Qwen3RMSNorm::from_weight(weight, eps))
    }

    fn metadata(&self, key: &str) -> Result<&gguf_file::Value> {
        self.content
            .metadata
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("GGUF metadata has no `{key}`")))
    }

    fn metadata_usize(&self, key: &str) -> Result<usize> {
        Ok(self.metadata(key)?.to_u64()? as usize)
    }

    fn config(&self) -> Result<Config> {
        let architecture = self.metadata("general.architecture")?.to_string()?;
        if architecture != "qwen3" {
            candle_core::bail!("GGUF architecture is `{architecture}`, not `qwen3`");
        }
        if self.content.tensor_infos.contains_key("blk.0.attn_q.bias") {
            candle_core::bail!("GGUF Qwen3 with attention biases is not supported");
        }
        let get = |key: &str| self.metadata_usize(&format!("qwen3.{key}"));
        let hidden_size = get("embedding_length")?;
        let num_attention_heads = get("attention.head_count")?;
        // llama.cpp's fallback when the file has no key_length.
        let head_dim = match self.metadata("qwen3.attention.key_length") {
            Ok(value) => value.to_u64()? as usize,
            Err(_) => hidden_size / num_attention_heads,
        };
        let table = self
            .content
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| candle_core::Error::Msg("GGUF has no `token_embd.weight`".into()))?;
        let (vocab_size, width) = table.shape.dims2()?;
        if width != hidden_size {
            candle_core::bail!("GGUF `token_embd.weight` is {width} wide, expected {hidden_size}");
        }
        Ok(Config {
            attention_bias: false,
            attention_dropout: 0.0,
            head_dim: Some(head_dim),
            hidden_act: Activation::Silu,
            hidden_size,
            intermediate_size: get("feed_forward_length")?,
            max_position_embeddings: get("context_length")?,
            num_attention_heads,
            num_hidden_layers: get("block_count")?,
            num_key_value_heads: get("attention.head_count_kv")?,
            rms_norm_eps: f64::from(
                self.metadata("qwen3.attention.layer_norm_rms_epsilon")?
                    .to_f32()?,
            ),
            rope_scaling: None,
            rope_theta: f64::from(self.metadata("qwen3.rope.freq_base")?.to_f32()?),
            sliding_window: None,
            tie_word_embeddings: true,
            use_cache: false,
            use_sliding_window: false,
            vocab_size,
            max_window_layers: 0,
        })
    }
}

impl Module for Qwen3RMSNorm {
    /// `xs * rsqrt(mean(xs^2) + eps) * weight` as one kernel. On Metal it sums the squares in
    /// f32 and rounds once, at the end, to the input's dtype.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // A no-op when the input is in the weights' dtype, which is how the model runs.
        let weight = self.weight.to_dtype(xs.dtype())?;
        candle_nn::ops::rms_norm(&xs.contiguous()?, &weight, self.eps as f32)
    }
}

pub struct Qwen3MLP {
    gate_proj: Projection, // hidden -> intermediate
    up_proj: Projection,   // hidden -> intermediate
    down_proj: Projection, // intermediate -> hidden
    act_fn: Activation,
}

impl Qwen3MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: Projection::Dense(gate_proj),
            up_proj: Projection::Dense(up_proj),
            down_proj: Projection::Dense(down_proj),
            act_fn: cfg.hidden_act,
        })
    }

    fn from_gguf<R: Read + Seek>(
        cfg: &Config,
        weights: &mut GgufWeights<'_, R>,
        prefix: &str,
    ) -> Result<Self> {
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: weights.projection(&format!("{prefix}.ffn_gate"), inter, hidden)?,
            up_proj: weights.projection(&format!("{prefix}.ffn_up"), inter, hidden)?,
            down_proj: weights.projection(&format!("{prefix}.ffn_down"), hidden, inter)?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?;
        let up = xs.apply(&self.up_proj)?;
        // silu(gate) * up is one kernel on Metal (`silu_mul`); another activation takes two.
        let hidden = match self.act_fn {
            Activation::Silu => silu_mul(&gate, &up)?,
            act => (gate.apply(&act)? * up)?,
        };
        hidden.apply(&self.down_proj)
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

    /// The (cos, sin) `forward` returns for positions `0..seq_len` in every row, as `[T, D/2]`
    /// each (the half `forward` repeats), in `dtype`: the tables candle's fused rope kernel reads.
    pub(crate) fn plain_tables(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let inv_freq = self.inv_freq.to_device(device)?.to_dtype(DType::F32)?;
        let d2 = inv_freq.dims1()?;
        let positions = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = positions
            .reshape((seq_len, 1))?
            .broadcast_mul(&inv_freq.reshape((1, d2))?)?;
        let scale = scalar_f32(device, self.attention_factor)?;
        let cos = freqs.cos()?.broadcast_mul(&scale)?.to_dtype(dtype)?;
        let sin = freqs.sin()?.broadcast_mul(&scale)?.to_dtype(dtype)?;
        Ok((cos, sin))
    }
}

/// The rotary tables every layer applies to q and k.
#[derive(Clone, Copy)]
pub(crate) enum RotaryTables<'a> {
    /// `[B, T, D]` cos and sin from `Qwen3RotaryEmbedding::forward`, for any position ids
    /// (MRoPE included), applied by `apply_rotary_pos_emb`.
    PerRow(&'a Tensor, &'a Tensor),
    /// `[T, D/2]` cos and sin from `Qwen3RotaryEmbedding::plain_tables`, for positions `0..T` in
    /// every row, applied by candle's fused rope kernel.
    Plain(&'a Tensor, &'a Tensor),
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

/// HF's apply_rotary_pos_emb for one of q or k: `x` `[B, heads, T, D]`, cos and sin `[B, T, D]`.
fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // unsqueeze_dim=1 -> [B,1,T,D] then broadcast
    let cos_u = cos.unsqueeze(1)?;
    let sin_u = sin.unsqueeze(1)?;
    x.broadcast_mul(&cos_u)? + rotate_half(x)?.broadcast_mul(&sin_u)?
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

/// How attention is masked: by an additive mask, not at all, causally over a batch with no
/// padding, or by a padded batch's `[B, 1, T, T]` mask in the model's dtype
/// (`build_attention_mask_fused`). The last two run candle's fused attention kernel on Metal,
/// which never builds the `[B, heads, T, T]` table.
#[derive(Clone, Copy)]
pub(crate) enum AttentionMask<'a> {
    None,
    Additive(&'a Tensor),
    Causal,
    Fused(&'a Tensor),
}

/// The positions a layer computes its output for. Every layer needs k and v at every position,
/// but the embedding keeps only the last position's vector (left padding puts every row's last
/// real token there), so the last layer computes q, attention, o_proj and the MLP for it alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rows {
    All,
    Last,
}

// - q_norm/k_norm on head_dim after reshape
// - RoPE on q,k
// - repeat_kv for key/value (GQA)
pub struct Qwen3Attention {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
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
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;

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

        Self::from_parts(
            cfg,
            [q_proj, k_proj, v_proj, o_proj].map(Projection::Dense),
            q_norm,
            k_norm,
        )
    }

    fn from_gguf<R: Read + Seek>(
        cfg: &Config,
        weights: &mut GgufWeights<'_, R>,
        prefix: &str,
    ) -> Result<Self> {
        let (hidden, head_dim, eps) = (cfg.hidden_size, cfg.head_dim(), cfg.rms_norm_eps);
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        let projections = [
            weights.projection(&format!("{prefix}.attn_q"), q_out, hidden)?,
            weights.projection(&format!("{prefix}.attn_k"), kv_out, hidden)?,
            weights.projection(&format!("{prefix}.attn_v"), kv_out, hidden)?,
            weights.projection(&format!("{prefix}.attn_output"), hidden, q_out)?,
        ];
        let q_norm = weights.norm(&format!("{prefix}.attn_q_norm"), head_dim, eps)?;
        let k_norm = weights.norm(&format!("{prefix}.attn_k_norm"), head_dim, eps)?;
        Self::from_parts(cfg, projections, q_norm, k_norm)
    }

    /// `projections` is q, k, v, o.
    fn from_parts(
        cfg: &Config,
        projections: [Projection; 4],
        q_norm: Qwen3RMSNorm,
        k_norm: Qwen3RMSNorm,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        if !num_heads.is_multiple_of(num_kv_heads) {
            candle_core::bail!("num_heads must be a multiple of num_kv_heads");
        }
        let head_dim = cfg.head_dim();
        let [q_proj, k_proj, v_proj, o_proj] = projections;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: cfg.num_kv_groups(),
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// hidden_states: [B,T,H]
    /// rotary: the cos and sin tables, `RotaryTables`
    /// attention_mask: additive mask [B,1,T,T] (0 or -inf)
    /// `AttentionMask::Causal` and `AttentionMask::Fused` need a Metal device:
    /// `Qwen3Model::forward_causal` and `Qwen3Model::forward_padded` pass them only there.
    /// `rows` picks the query positions: every one (`[B,T,H]` out), or the last alone
    /// (`[B,1,H]` out), which attends to every key as the last row of the full attention does.
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        rotary: RotaryTables<'_>,
        mask: AttentionMask<'_>,
        rows: Rows,
    ) -> Result<Tensor> {
        let (b, t, _h) = hidden_states.dims3()?;
        let d = self.head_dim;
        // The query positions, `tq` of them, ending at the last.
        let (q_start, tq) = match rows {
            Rows::All => (0, t),
            Rows::Last => (t - 1, 1),
        };

        // hidden_shape = [B,T,-1,D]: q [B,Tq,Nh,D] and k [B,T,Nkv,D], before q_norm and k_norm
        let q = hidden_states
            .narrow(1, q_start, tq)?
            .apply(&self.q_proj)?
            .reshape((b, tq, self.num_heads, d))?;
        let k = hidden_states
            .apply(&self.k_proj)?
            .reshape((b, t, self.num_kv_heads, d))?;

        // v: [B,Nkv,T,D]
        let v = hidden_states
            .apply(&self.v_proj)?
            .reshape((b, t, self.num_kv_heads, d))?
            .transpose(1, 2)?;

        // q_norm/k_norm and RoPE, then q: [B,Nh,Tq,D] and k: [B,Nkv,T,D]. With the plain tables
        // the two run as one pass over q and one over k (one kernel each on Metal). q's rows of
        // the tables are its own positions.
        let (q, k) = match rotary {
            RotaryTables::Plain(cos, sin) => {
                let norm_rope = |xs: &Tensor, norm: &Qwen3RMSNorm, start: usize, len: usize| {
                    rms_norm_rope_thd(
                        xs,
                        &norm.weight,
                        &cos.narrow(0, start, len)?,
                        &sin.narrow(0, start, len)?,
                        norm.eps as f32,
                    )
                };
                (
                    norm_rope(&q, &self.q_norm, q_start, tq)?.transpose(1, 2)?,
                    norm_rope(&k, &self.k_norm, 0, t)?.transpose(1, 2)?,
                )
            }
            RotaryTables::PerRow(cos, sin) => (
                apply_rotary_pos_emb(
                    &q.apply(&self.q_norm)?.transpose(1, 2)?,
                    &cos.narrow(1, q_start, tq)?,
                    &sin.narrow(1, q_start, tq)?,
                )?,
                apply_rotary_pos_emb(&k.apply(&self.k_norm)?.transpose(1, 2)?, cos, sin)?,
            ),
        };

        // The fused kernel takes k and v with their own heads (GQA). It masks causally itself, or
        // reads a given mask through its strides, so the heads share one `[B, 1, T, T]` copy.
        // The last position alone sees every key, so it needs no causal mask, and a padding mask
        // keeps its last row, copied out (`[B, 1, 1, T]`) because the kernel reads the mask from
        // its buffer's start.
        let fused = match mask {
            AttentionMask::Causal => Some((None, rows == Rows::All)),
            AttentionMask::Fused(mask) => Some((
                Some(mask.narrow(2, q_start, tq)?.contiguous()?.broadcast_as((
                    b,
                    self.num_heads,
                    tq,
                    t,
                ))?),
                false,
            )),
            AttentionMask::None | AttentionMask::Additive(_) => None,
        };
        if let Some((fused_mask, causal)) = fused {
            // q, k and v go in as the transposed views they are: the fused kernel reads them
            // through their batch, head and sequence strides (the head dim stays contiguous), so
            // no copy of any of them is made. It writes `[B, Tq, Nh, D]`, so the reshape for
            // o_proj is a view too.
            let out = sdpa_bthd(&q, &k, &v, fused_mask.as_ref(), causal, self.scaling)?;
            let out = out.reshape((b, tq, self.num_heads * d))?;
            return out.apply(&self.o_proj);
        }
        let attention_mask = match mask {
            AttentionMask::Additive(mask) if mask.rank() == 4 && mask.dim(2)? == t => {
                Some(mask.narrow(2, q_start, tq)?)
            }
            AttentionMask::Additive(mask) => Some(mask.clone()),
            AttentionMask::None | AttentionMask::Causal | AttentionMask::Fused(_) => None,
        };

        // The fused rope leaves q and k transposed views; the matmuls below read them contiguous.
        let q = q.contiguous()?;
        let k = k.contiguous()?;

        // GQA expand
        let k = repeat_kv(&k, self.num_kv_groups)?; // [B,Nh,T,D]
        let v = repeat_kv(&v, self.num_kv_groups)?; // [B,Nh,T,D]

        // attn_weights = q @ k^T * scaling
        let kt = k.transpose(2, 3)?; // [B,Nh,D,T]
        let mut attn = q.matmul(&kt)?; // [B,Nh,Tq,T]

        let scale = scalar_f32(attn.device(), self.scaling)?.to_dtype(attn.dtype())?;
        attn = attn.broadcast_mul(&scale)?;

        if let Some(mask) = attention_mask {
            attn = attn.broadcast_add(&mask.to_dtype(attn.dtype())?)?;
        }

        // softmax over last dim
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // attn_output = attn @ v -> [B,Nh,Tq,D]
        let out = attn.matmul(&v)?;
        // transpose back -> [B,Tq,Nh,D] -> reshape [B,Tq,Nh*D] -> o_proj -> [B,Tq,H]
        let out = out.transpose(1, 2)?.reshape((b, tq, self.num_heads * d))?;
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

    fn from_gguf<R: Read + Seek>(
        cfg: &Config,
        weights: &mut GgufWeights<'_, R>,
        i: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{i}");
        let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        Ok(Self {
            self_attn: Qwen3Attention::from_gguf(cfg, weights, &prefix)?,
            mlp: Qwen3MLP::from_gguf(cfg, weights, &prefix)?,
            input_layernorm: weights.norm(&format!("{prefix}.attn_norm"), hidden, eps)?,
            post_attention_layernorm: weights.norm(&format!("{prefix}.ffn_norm"), hidden, eps)?,
        })
    }

    /// `[B,T,H]` in; `[B,T,H]` out for `Rows::All`, the last position's `[B,1,H]` for `Rows::Last`.
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        mask: AttentionMask<'_>,
        rotary: RotaryTables<'_>,
        rows: Rows,
    ) -> Result<Tensor> {
        // Pre-norm
        let residual = match rows {
            Rows::All => hidden_states.clone(),
            Rows::Last => hidden_states.narrow(1, hidden_states.dim(1)? - 1, 1)?,
        };
        let hs = hidden_states.apply(&self.input_layernorm)?;
        let hs = self.self_attn.forward(&hs, rotary, mask, rows)?;
        let hs = (residual + hs)?;

        // MLP
        let residual = hs.clone();
        let hs2 = hs.apply(&self.post_attention_layernorm)?;
        let hs2 = hs2.apply(&self.mlp)?;
        residual + hs2
    }
}

/// A quantized table stays on the CPU. A lookup dequantizes only the rows it reads.
enum TokenEmbedding {
    Dense(candle_nn::Embedding),
    Quantized { table: Box<QTensor>, device: Device },
}

impl TokenEmbedding {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            TokenEmbedding::Dense(embedding) => embedding.forward(input_ids),
            TokenEmbedding::Quantized { table, device } => table
                .embedding(&input_ids.to_device(&Device::Cpu)?)?
                .to_device(device),
        }
    }

    fn ids_device<'a>(&'a self, model_device: &'a Device) -> &'a Device {
        match self {
            TokenEmbedding::Dense(_) => model_device,
            TokenEmbedding::Quantized { .. } => &Device::Cpu,
        }
    }
}

pub struct Qwen3Model {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    cfg: Config,
    device: Device,
}

impl Qwen3Model {
    /// Loads a llama.cpp `qwen3` GGUF file, such as `Qwen/Qwen3-Embedding-4B-GGUF`. The
    /// matrices stay quantized on `device`, the token table stays quantized on the CPU, and the
    /// model runs in f32.
    pub fn from_gguf<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut weights = GgufWeights {
            content,
            reader,
            device,
        };
        let cfg = weights.config()?;
        let table = weights.tensor(
            "token_embd.weight",
            &[cfg.vocab_size, cfg.hidden_size],
            &Device::Cpu,
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::from_gguf(&cfg, &mut weights, i)?);
        }
        let norm = weights.norm("output_norm", cfg.hidden_size, cfg.rms_norm_eps)?;
        let rotary_emb = Qwen3RotaryEmbedding::new(&cfg, device)?;
        Ok(Self {
            embed_tokens: TokenEmbedding::Quantized {
                table: Box::new(table),
                device: device.clone(),
            },
            layers,
            norm,
            rotary_emb,
            cfg,
            device: device.clone(),
        })
    }

    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let embed_tokens = TokenEmbedding::Dense(candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?);

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
        let mask = attention_mask_4d.map_or(AttentionMask::None, AttentionMask::Additive);
        self.forward_embeds_masked(
            inputs_embeds,
            mask,
            deepstack_additions,
            position_ids,
            Rows::All,
        )
    }

    /// `rows` is the positions the last layer computes, and so the positions returned: `[B,T,H]`
    /// for `Rows::All`, the last position's `[B,1,H]` for `Rows::Last`.
    fn forward_embeds_masked(
        &self,
        inputs_embeds: &Tensor,
        mask: AttentionMask<'_>,
        deepstack_additions: Option<&[Tensor]>,
        position_ids: Option<&Tensor>,
        rows: Rows,
    ) -> Result<Tensor> {
        let (hs, _) = self.forward_embeds_tapped(
            inputs_embeds,
            mask,
            deepstack_additions,
            position_ids,
            rows,
            &[],
        )?;
        Ok(hs)
    }

    /// `forward_embeds_masked`, also returning the last position's hidden state `[B, 1, H]`
    /// after each layer count in `taps` (1 is after the first layer), in the order given: the
    /// residual stream before the final norm. A tap at the last layer is its output, unnormed.
    fn forward_embeds_tapped(
        &self,
        inputs_embeds: &Tensor,
        mask: AttentionMask<'_>,
        deepstack_additions: Option<&[Tensor]>,
        position_ids: Option<&Tensor>,
        rows: Rows,
        taps: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        if rows == Rows::Last && deepstack_additions.is_some() {
            candle_core::bail!("the deepstack additions need every position of every layer")
        }
        if let Some(&bad) = taps.iter().find(|&&l| l == 0 || l > self.layers.len()) {
            candle_core::bail!(
                "layer {bad} is not between 1 and the model's {} layers",
                self.layers.len()
            )
        }
        let mut tapped: Vec<Option<Tensor>> = vec![None; taps.len()];
        let (_, t, _) = inputs_embeds.dims3()?;
        let mut hs = inputs_embeds.clone();

        // (cos, sin) once. With no position ids every row's positions are 0..T, which the fused
        // rope kernel takes as one `[T, D/2]` table; given ids (MRoPE) take the per-row tables.
        let (cos, sin) = match position_ids {
            Some(position_ids) => self.rotary_emb.forward(&hs, position_ids)?,
            None => self.rotary_emb.plain_tables(t, hs.dtype(), hs.device())?,
        };
        let rotary = match position_ids {
            Some(_) => RotaryTables::PerRow(&cos, &sin),
            None => RotaryTables::Plain(&cos, &sin),
        };

        // layers
        let last_layer = self.layers.len().saturating_sub(1);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_rows = if layer_idx == last_layer {
                rows
            } else {
                Rows::All
            };
            hs = layer.forward(&hs, mask, rotary, layer_rows)?;
            if let Some(additions) = deepstack_additions {
                if let Some(visual_add) = additions.get(layer_idx) {
                    hs = hs.add(visual_add)?;
                }
            }
            for (slot, _) in taps.iter().enumerate().filter(|(_, &l)| l == layer_idx + 1) {
                // A copy, so the tap does not keep the whole `[B, T, H]` buffer alive.
                let last = hs.narrow(1, hs.dim(1)? - 1, 1)?.force_contiguous()?;
                tapped[slot] = Some(last);
            }
        }

        let tapped = tapped
            .into_iter()
            .map(|t| t.ok_or_else(|| candle_core::Error::Msg("a layer tap was not taken".into())))
            .collect::<Result<Vec<_>>>()?;
        // final norm
        Ok((hs.apply(&self.norm)?, tapped))
    }

    /// input_ids: `[B,T]`
    /// attention_mask: optionally `[B,T]` (1=token,0=pad) OR already expanded additive mask `[B,1,T,T]`
    /// Here we assume additive mask `[B,1,T,T]` for simplicity (HF does that).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask_4d: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hs = self.embed_tokens(input_ids)?;
        self.forward_with_inputs_embeds(&hs, attention_mask_4d, None, None)
    }

    /// `forward` with causal attention over a batch that has no padding. On Metal this runs the
    /// fused attention kernel, which never builds the `[B, heads, T, T]` table or the mask; on any
    /// other device it builds the same causal mask `forward` would get for an unpadded batch.
    /// `rows` as in `forward_embeds_masked`.
    pub(crate) fn forward_causal(&self, input_ids: &Tensor, rows: Rows) -> Result<Tensor> {
        Ok(self.forward_causal_tapped(input_ids, rows, &[])?.0)
    }

    /// `forward_causal` with the layer taps of `forward_embeds_tapped`.
    pub(crate) fn forward_causal_tapped(
        &self,
        input_ids: &Tensor,
        rows: Rows,
        taps: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let hs = self.embed_tokens(input_ids)?;
        if self.device.is_metal() {
            return self.forward_embeds_tapped(&hs, AttentionMask::Causal, None, None, rows, taps);
        }
        let (b, t) = input_ids.dims2()?;
        let no_padding = Tensor::ones((b, t), DType::F32, &self.device)?;
        let mask = build_attention_mask_4d(&no_padding)?;
        self.forward_embeds_tapped(&hs, AttentionMask::Additive(&mask), None, None, rows, taps)
    }

    /// `forward` over a padded batch, `attention_mask_2d` being `[B, T]` (1 a token, 0 padding).
    /// On Metal this runs the fused attention kernel with the padding mask, which never builds the
    /// `[B, heads, T, T]` table; on any other device it is `forward` with the mask
    /// `build_attention_mask_4d` builds.
    /// `rows` as in `forward_embeds_masked`; with left padding the last position is every row's
    /// last real token.
    pub(crate) fn forward_padded(
        &self,
        input_ids: &Tensor,
        attention_mask_2d: &Tensor,
        rows: Rows,
    ) -> Result<Tensor> {
        Ok(self
            .forward_padded_tapped(input_ids, attention_mask_2d, rows, &[])?
            .0)
    }

    /// `forward_padded` with the layer taps of `forward_embeds_tapped`.
    pub(crate) fn forward_padded_tapped(
        &self,
        input_ids: &Tensor,
        attention_mask_2d: &Tensor,
        rows: Rows,
        taps: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let hs = self.embed_tokens(input_ids)?;
        if self.device.is_metal() {
            let mask = build_attention_mask_fused(attention_mask_2d, hs.dtype())?;
            return self.forward_embeds_tapped(
                &hs,
                AttentionMask::Fused(&mask),
                None,
                None,
                rows,
                taps,
            );
        }
        let mask = build_attention_mask_4d(attention_mask_2d)?;
        self.forward_embeds_tapped(&hs, AttentionMask::Additive(&mask), None, None, rows, taps)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Left padding keeps the last real token at the last position, where the pooling reads it.
/// Qwen3-Embedding adds no EOS token.
fn embedding_tokenizer(path: &Path, max_length: usize) -> Result<tokenizers::Tokenizer> {
    use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams};

    let mut tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| candle_core::Error::Msg(format!("{}: {e}", path.display())))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Left,
        ..Default::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(map_err)?;
    Ok(tokenizer)
}

/// How many bytes of a long text [`encode_texts`] reads first for each token it may keep. Qwen3's
/// tokenizer takes about 5 bytes per token of English or Chinese and 1 of digits, so 16 leaves
/// room for most text. Long runs of one punctuation mark can take 50 bytes a token or more;
/// such a text comes up short and is encoded whole, as before.
const PREFIX_BYTES_PER_TOKEN: usize = 16;

/// `tokenizer.encode_batch_fast(texts, true)`, without encoding the part of a long text that
/// truncation would throw away. When the tokenizer truncates on the right to `max_length`, a
/// text longer than `max_length * PREFIX_BYTES_PER_TOKEN` bytes is encoded only up to a space
/// near that size (see [`token_prefix`]); if that prefix still runs past `max_length` tokens,
/// its kept tokens are the whole text's, and otherwise the whole text is encoded. The fast
/// encode leaves out the byte offsets of each token, which nothing here reads; the ids and the
/// attention mask are the same.
fn encode_texts<S: AsRef<str>>(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[S],
) -> Result<Vec<tokenizers::Encoding>> {
    let limit = match tokenizer.get_truncation() {
        Some(t) if t.direction == tokenizers::TruncationDirection::Right => {
            Some(t.max_length.saturating_mul(PREFIX_BYTES_PER_TOKEN))
        }
        _ => None,
    };
    let sources: Vec<&str> = texts
        .iter()
        .map(|t| match limit {
            Some(limit) => token_prefix(t.as_ref(), limit),
            None => t.as_ref(),
        })
        .collect();
    let encode = |sources: Vec<&str>| tokenizer.encode_batch_fast(sources, true).map_err(map_err);
    let mut encodings = encode(sources.clone())?;
    // A prefix truncation cut nothing from may have left tokens out: encode those texts whole,
    // and only those.
    let short: Vec<usize> = (0..texts.len())
        .filter(|&i| {
            sources[i].len() < texts[i].as_ref().len() && encodings[i].get_overflowing().is_empty()
        })
        .collect();
    if short.is_empty() {
        return Ok(encodings);
    }
    let whole = encode(short.iter().map(|&i| texts[i].as_ref()).collect())?;
    for (&i, encoding) in short.iter().zip(whole) {
        encodings[i] = encoding;
    }
    // The two passes were padded apart. Padding only ever adds pad tokens, and no text got
    // shorter, so padding the joined batch again gives what one pass over it would have.
    if let Some(params) = tokenizer.get_padding() {
        tokenizers::utils::padding::pad_encodings(&mut encodings, params).map_err(map_err)?;
    }
    Ok(encodings)
}

/// The longest prefix of `text` of at most `limit` bytes that ends just before a space which
/// follows a character other than whitespace, or all of `text` when it is no longer than
/// `limit` or has no such space. Qwen3's pre-tokenizer splits text into pieces before it
/// encodes them, and no piece runs from such a character across such a space, so the prefix's
/// pieces, and with them its tokens, are the first ones of the whole text.
fn token_prefix(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    let bytes = text.as_bytes();
    for end in (1..=limit).rev() {
        // A space byte always starts a character, so `end` is a character boundary.
        if bytes[end] == b' ' {
            let before = text[..end].chars().next_back();
            if before.is_some_and(|c| !c.is_whitespace()) {
                return &text[..end];
            }
        }
    }
    text
}

/// High-level embedding wrapper around [`Qwen3Model`] that handles tokenization,
/// batching, mean-pooling, and L2-normalization.
pub struct Qwen3TextEmbedding {
    model: Qwen3Model,
    tokenizer: tokenizers::Tokenizer,
    /// Keep this many leading dimensions of each vector before normalizing; `None` keeps all.
    output_dim: Option<usize>,
}

impl Qwen3TextEmbedding {
    /// Build from an already loaded model and tokenizer.
    pub fn new(model: Qwen3Model, tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            output_dim: None,
        }
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
        let tokenizer = embedding_tokenizer(&tok_path, max_length)?;

        Ok(Self {
            model,
            tokenizer,
            output_dim: None,
        })
    }

    /// Return vectors of `dim` numbers: each vector's leading `dim` dimensions, L2-normalized
    /// again. Qwen3-Embedding is trained so that these shorter vectors still work (its model card
    /// lists output sizes from 32 up to the hidden size). `dim` equal to the hidden size gives
    /// the full vectors unchanged.
    pub fn with_output_dim(mut self, dim: usize) -> Result<Self> {
        let hidden = self.model.config().hidden_size;
        if dim == 0 || dim > hidden {
            return Err(candle_core::Error::Msg(format!(
                "output dim {dim} must be between 1 and the hidden size {hidden}"
            )));
        }
        self.output_dim = (dim < hidden).then_some(dim);
        Ok(self)
    }

    /// The length of the vectors `embed` returns.
    pub fn output_dim(&self) -> usize {
        self.output_dim.unwrap_or(self.model.config().hidden_size)
    }

    /// Loads a GGUF file, such as `Qwen3-Embedding-4B-Q8_0.gguf`, with the original model's
    /// `tokenizer.json`. See [`Qwen3Model::from_gguf`].
    pub fn from_gguf(
        gguf_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
        max_length: usize,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(gguf_path)
            .map_err(|e| candle_core::Error::Msg(format!("{}: {e}", gguf_path.display())))?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(gguf_path))?;
        let model = Qwen3Model::from_gguf(&content, &mut file, device)?;
        let tokenizer = embedding_tokenizer(tokenizer_path, max_length)?;
        Ok(Self {
            model,
            tokenizer,
            output_dim: None,
        })
    }

    /// [`Self::from_gguf`] with `gguf_file` from `gguf_repo_id` and `tokenizer.json` from
    /// `tokenizer_repo_id`.
    #[cfg(feature = "hf-hub")]
    pub fn from_hf_gguf(
        gguf_repo_id: &str,
        gguf_file: &str,
        tokenizer_repo_id: &str,
        device: &Device,
        max_length: usize,
    ) -> Result<Self> {
        let gguf_path = hf_repo(gguf_repo_id)?.get(gguf_file).map_err(map_err)?;
        let tokenizer_path = hf_repo(tokenizer_repo_id)?
            .get("tokenizer.json")
            .map_err(map_err)?;
        Self::from_gguf(&gguf_path, &tokenizer_path, device, max_length)
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Embed a batch of texts, returning normalized embeddings in the order of `texts`.
    ///
    /// The texts are grouped by token length and each group runs as one batch with no padding,
    /// so every forward takes the causal path and every text's vector is the one it gets
    /// embedded alone, whatever else is in the call.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_tapped(texts, &[])?.0)
    }

    /// The pooled vector of each layer count in `layers` (1 is after the first decoder layer,
    /// `num_hidden_layers` the model's output), every one from the same single pass, returned in
    /// the order of `layers`, each holding one vector per text. The model's output is exactly what
    /// [`Self::embed`] returns. An earlier layer's vector is the last position's residual stream
    /// after that layer (before the final norm, which belongs to the output), L2-normalized.
    /// [`Self::with_output_dim`] shortens the model's output only: an earlier layer's vector keeps
    /// the full hidden size, since the shorter sizes are trained on the output alone.
    pub fn embed_layers<S: AsRef<str>>(
        &self,
        texts: &[S],
        layers: &[usize],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let last = self.model.config().num_hidden_layers;
        if let Some(&bad) = layers.iter().find(|&&l| l == 0 || l > last) {
            return Err(candle_core::Error::Msg(format!(
                "layer {bad} is not between 1 and the model's {last} layers"
            )));
        }
        let taps: Vec<usize> = layers.iter().copied().filter(|&l| l < last).collect();
        let (output, mut tapped) = self.embed_tapped(texts, &taps)?;
        let mut tapped = tapped.drain(..);
        Ok(layers
            .iter()
            .map(|&l| {
                if l == last {
                    output.clone()
                } else {
                    tapped.next().unwrap_or_default()
                }
            })
            .collect())
    }

    /// `embed`, also returning the normalized pooled vectors of the layer counts in `taps`.
    #[allow(clippy::type_complexity)]
    fn embed_tapped<S: AsRef<str>>(
        &self,
        texts: &[S],
        taps: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>)> {
        if texts.is_empty() {
            return Ok((vec![], vec![vec![]; taps.len()]));
        }
        let encodings = encode_texts(&self.tokenizer, texts)?;

        // Each text's tokens without the left padding, grouped by their count.
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        let mut tokens: Vec<&[u32]> = Vec::with_capacity(encodings.len());
        for (i, enc) in encodings.iter().enumerate() {
            let pad = enc
                .get_attention_mask()
                .iter()
                .take_while(|&&m| m == 0)
                .count();
            let ids = &enc.get_ids()[pad..];
            if ids.is_empty() {
                return Err(candle_core::Error::Msg(format!("text {i} has no tokens")));
            }
            groups.entry(ids.len()).or_default().push(i);
            tokens.push(ids);
        }

        let device = self.model.device();
        let mut out = vec![Vec::new(); texts.len()];
        let mut out_tapped = vec![vec![Vec::new(); texts.len()]; taps.len()];
        for (len, members) in groups {
            let ids: Vec<u32> = members
                .iter()
                .flat_map(|&i| tokens[i].iter().copied())
                .collect();
            let input_ids = Tensor::from_vec(
                ids,
                (members.len(), len),
                self.model.embed_tokens.ids_device(device),
            )?;
            // [k, 1, H]: the last layer computes the last position alone.
            let (hidden, tapped) =
                self.model
                    .forward_causal_tapped(&input_ids, Rows::Last, taps)?;
            for (i, v) in members.iter().zip(pool_last(&hidden, self.output_dim)?) {
                out[*i] = v;
            }
            for (slot, hidden) in out_tapped.iter_mut().zip(&tapped) {
                for (i, v) in members.iter().zip(pool_last(hidden, None)?) {
                    slot[*i] = v;
                }
            }
        }
        Ok((out, out_tapped))
    }

    /// Embed every text in one left-padded batch: one forward for the whole call, but each
    /// text's vector depends a little on how much its batch pads it. [`Self::embed`] runs
    /// equal-length groups without padding instead.
    pub fn embed_padded<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = encode_texts(&self.tokenizer, texts)?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();
        if seq_len == 0 {
            return Err(candle_core::Error::Msg("the texts have no tokens".into()));
        }

        let mut input_ids_vec: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            input_ids_vec.extend(enc.get_ids().iter().copied());
            attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
        }

        let device = self.model.device();
        let padded = attention_mask_vec.contains(&0.0);
        let input_ids = Tensor::from_vec(
            input_ids_vec,
            (batch_size, seq_len),
            self.model.embed_tokens.ids_device(device),
        )?;

        // Forward pass -> [B, T, H]. With no padding the mask is purely causal, which the fused
        // kernel applies itself, so neither the mask nor the attention table is built. With
        // padding the kernel reads one `[B, 1, T, T]` mask and still builds no table.
        let (hidden, _) = if padded {
            let attention_mask_2d =
                Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;
            self.model
                .forward_padded_tapped(&input_ids, &attention_mask_2d, Rows::Last, &[])?
        } else {
            self.model
                .forward_causal_tapped(&input_ids, Rows::Last, &[])?
        };

        // Last token pooling: with left padding, the last position is always the real last
        // token, and it is the one position the last layer computed.
        pool_last(&hidden, self.output_dim)
    }
}

/// The L2-normalized last-position rows of `hidden` `[B, 1, H]`, one vector per batch row.
/// With `dim`, each row keeps its leading `dim` dimensions before it is normalized, as the full
/// vector is.
fn pool_last(hidden: &Tensor, dim: Option<usize>) -> Result<Vec<Vec<f32>>> {
    let pooled = hidden.i((.., 0))?;
    let pooled = match dim {
        Some(dim) => pooled.narrow(1, 0, dim)?,
        None => pooled,
    };
    l2_normalize(&pooled)?.to_vec2::<f32>()
}

/// The instruction the Qwen3-Reranker model card uses when none is given.
pub const QWEN3_RERANKER_DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

const QWEN3_RERANKER_BATCH_SIZE: usize = 16;

const QWEN3_RERANKER_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const QWEN3_RERANKER_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Qwen3-Reranker (for example `Qwen/Qwen3-Reranker-0.6B`): the probability that a document
/// answers a query under an instruction, read as the model's "yes" against "no" at the last
/// position, as the model card computes it.
pub struct Qwen3Reranker {
    model: Qwen3Model,
    tokenizer: tokenizers::Tokenizer,
    /// The output embedding rows of "no" and "yes", `[2, H]`, in the model's dtype.
    no_yes: Tensor,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    max_length: usize,
}

impl Qwen3Reranker {
    /// Build from a loaded model, its tokenizer (no padding or truncation set; this sets its own),
    /// and the `[2, H]` output rows of the "no" and "yes" tokens.
    pub fn new(
        model: Qwen3Model,
        mut tokenizer: tokenizers::Tokenizer,
        no_yes: Tensor,
        max_length: usize,
    ) -> Result<Self> {
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(map_err)?;
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(map_err)?
                .get_ids()
                .to_vec())
        };
        let prefix = encode(QWEN3_RERANKER_PREFIX)?;
        let suffix = encode(QWEN3_RERANKER_SUFFIX)?;
        if max_length <= prefix.len() + suffix.len() {
            return Err(candle_core::Error::Msg(format!(
                "max_length {max_length} leaves no room beside the prompt's {} tokens",
                prefix.len() + suffix.len()
            )));
        }
        Ok(Self {
            model,
            tokenizer,
            no_yes,
            prefix,
            suffix,
            max_length,
        })
    }

    /// Load from a Hugging Face repo such as `Qwen/Qwen3-Reranker-0.6B`.
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        let repo = hf_repo(repo_id)?;
        let cfg_path: PathBuf = repo.get("config.json").map_err(map_err)?;
        let cfg_bytes = std::fs::read(&cfg_path).map_err(map_err)?;
        let (cfg, _) = parse_config_and_weight_prefix(&cfg_bytes)?;
        let weight_files = crate::common::safetensors_weight_files(&repo).map_err(map_err)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| candle_core::Error::Msg(format!("{}: {e}", tok_path.display())))?;
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                candle_core::Error::Msg(format!("the tokenizer has no `{token}` token"))
            })
        };
        let ids = Tensor::new(&[id("no")?, id("yes")?], device)?;
        // The output layer, tied to the token table when the config says so.
        let output = if cfg.tie_word_embeddings {
            vb.pp("model.embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?
        } else {
            vb.pp("lm_head")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?
        };
        let no_yes = output.index_select(&ids, 0)?;
        let model = Qwen3Model::new(cfg, vb.pp("model"))?;
        Self::new(model, tokenizer, no_yes, max_length)
    }

    /// The probability of "yes" for each document against `query` under `instruction`
    /// ([`QWEN3_RERANKER_DEFAULT_INSTRUCTION`] when `None`), in the order of `documents`.
    pub fn score<S: AsRef<str>>(
        &self,
        instruction: Option<&str>,
        query: &str,
        documents: &[S],
    ) -> Result<Vec<f32>> {
        let instruction = instruction.unwrap_or(QWEN3_RERANKER_DEFAULT_INSTRUCTION);
        let room = self.max_length - self.prefix.len() - self.suffix.len();
        let mut rows = Vec::with_capacity(documents.len());
        for doc in documents {
            let pair = format!(
                "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {}",
                doc.as_ref()
            );
            let encoding = self.tokenizer.encode(pair, true).map_err(map_err)?;
            let ids = encoding.get_ids();
            let mut row = self.prefix.clone();
            row.extend_from_slice(&ids[..ids.len().min(room)]);
            row.extend_from_slice(&self.suffix);
            rows.push(row);
        }

        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].len());
        let mut scores = vec![0f32; rows.len()];
        for chunk in order.chunks(QWEN3_RERANKER_BATCH_SIZE) {
            let batch: Vec<Vec<u32>> = chunk
                .iter()
                .map(|&i| std::mem::take(&mut rows[i]))
                .collect();
            for (&i, score) in chunk.iter().zip(self.score_rows(&batch)?) {
                scores[i] = score;
            }
        }
        Ok(scores)
    }

    /// P("yes") for each row of token ids, run as one left-padded batch.
    fn score_rows(&self, rows: &[Vec<u32>]) -> Result<Vec<f32>> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let device = self.model.device();
        let t = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(rows.len() * t);
        let mut mask = Vec::with_capacity(rows.len() * t);
        for row in rows {
            let pad = t - row.len();
            ids.extend(std::iter::repeat_n(0u32, pad).chain(row.iter().copied()));
            mask.extend(std::iter::repeat_n(0f32, pad).chain(std::iter::repeat_n(1f32, row.len())));
        }
        let input_ids = Tensor::from_vec(
            ids,
            (rows.len(), t),
            self.model.embed_tokens.ids_device(device),
        )?;
        let hidden = if mask.contains(&0.0) {
            let mask = Tensor::from_vec(mask, (rows.len(), t), device)?;
            self.model.forward_padded(&input_ids, &mask, Rows::Last)?
        } else {
            self.model.forward_causal(&input_ids, Rows::Last)?
        };
        // [B, 2]: the "no" and "yes" logits at the last position, then P("yes").
        let logits = hidden
            .i((.., 0))?
            .matmul(&self.no_yes.to_dtype(hidden.dtype())?.t()?)?
            .to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        probs.i((.., 1))?.to_vec1::<f32>()
    }
}

/// Multimodal embedding wrapper for Qwen3-VL embedding checkpoints.
///
/// This supports text-only and image-only inputs. Image+text mixed inputs can be
/// added later with the same backend.
#[cfg(feature = "qwen3-vl")]
pub struct Qwen3VLEmbedding {
    model: Qwen3Model,
    vision: Qwen3VLVisionModel,
    tokenizer: tokenizers::Tokenizer,
    preprocessor: Qwen3VLPreprocessorConfig,
    image_token_id: u32,
    default_instruction: String,
    max_image_tokens: usize,
}

#[cfg(feature = "qwen3-vl")]
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
        l2_normalize(&pooled)?.to_vec2::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_config_and_weight_prefix;
    #[cfg(feature = "qwen3-vl")]
    use super::{expand_image_token_placeholders, find_token_spans, round_ties_to_even};

    #[test]
    fn fused_mask_matches_the_additive_mask() {
        use super::{build_attention_mask_4d, build_attention_mask_fused};
        use candle_core::{DType, Device, Tensor};

        let device = Device::Cpu;
        // Two left-padded rows: two pads, and none.
        let mask_2d = Tensor::from_vec(
            vec![0f32, 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            (2, 5),
            &device,
        )
        .unwrap();
        let additive = build_attention_mask_4d(&mask_2d).unwrap();
        let fused = build_attention_mask_fused(&mask_2d, DType::F32).unwrap();
        assert_eq!(fused.dims(), additive.dims());
        assert_eq!(
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            additive.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    fn max_abs_diff(a: &candle_core::Tensor, b: &candle_core::Tensor) -> f32 {
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
    }

    fn small_config() -> super::Config {
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
            "rope_scaling": null,
            "rope_theta": 10000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "vocab_size": 1000
        }"#;
        parse_config_and_weight_prefix(config.as_bytes()).unwrap().0
    }

    #[test]
    fn fused_rms_norm_matches_the_f32_formula() {
        use super::Qwen3RMSNorm;
        use candle_core::{DType, Device, Module, Tensor, D};

        let device = Device::Cpu;
        let eps = 1e-6f64;
        let xs = Tensor::randn(0f32, 3.0, (2, 5, 3, 16), &device).unwrap();
        let weight = Tensor::randn(1f32, 0.5, (16,), &device).unwrap();
        let norm = Qwen3RMSNorm {
            weight: weight.clone(),
            eps,
        };

        // The unfused form: xs * 1/sqrt(mean(xs^2) + eps) * weight, in f32.
        let var = xs.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let inv_rms = (var + eps).unwrap().sqrt().unwrap().recip().unwrap();
        let expected = xs
            .broadcast_mul(&inv_rms)
            .unwrap()
            .broadcast_mul(&weight)
            .unwrap();

        let fused = norm.forward(&xs).unwrap();
        assert_eq!(fused.dims(), expected.dims());
        assert_eq!(fused.dtype(), DType::F32);
        assert!(max_abs_diff(&fused, &expected) < 1e-5, "rms norm differs");
    }

    #[test]
    fn fused_rope_matches_rotate_half_for_plain_positions() {
        use super::{apply_rotary_pos_emb, Qwen3RotaryEmbedding};
        use candle_core::{DType, Device, Tensor};

        let cfg = small_config();
        let device = Device::Cpu;
        let rotary = Qwen3RotaryEmbedding::new(&cfg, &device).unwrap();
        let (b, t, d) = (2usize, 9usize, cfg.head_dim());
        let (heads, kv_heads) = (cfg.num_attention_heads, cfg.num_key_value_heads);

        // q and k as the attention holds them after q_norm and k_norm: [B, T, heads, D].
        let q = Tensor::randn(0f32, 1.0, (b, t, heads, d), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, t, kv_heads, d), &device).unwrap();

        let xs = Tensor::zeros((b, t, d), DType::F32, &device).unwrap();
        let positions: Vec<u32> = (0..b).flat_map(|_| 0..t as u32).collect();
        let position_ids = Tensor::from_vec(positions, (b, t), &device).unwrap();
        let (cos, sin) = rotary.forward(&xs, &position_ids).unwrap();
        let q_ref = apply_rotary_pos_emb(&q.transpose(1, 2).unwrap(), &cos, &sin).unwrap();
        let k_ref = apply_rotary_pos_emb(&k.transpose(1, 2).unwrap(), &cos, &sin).unwrap();

        let (cos, sin) = rotary.plain_tables(t, DType::F32, &device).unwrap();
        assert_eq!(cos.dims(), &[t, d / 2]);
        let q_fused = candle_nn::rotary_emb::rope_thd(&q, &cos, &sin)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k_fused = candle_nn::rotary_emb::rope_thd(&k, &cos, &sin)
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        assert!(max_abs_diff(&q_fused, &q_ref) < 1e-5, "q differs");
        assert!(max_abs_diff(&k_fused, &k_ref) < 1e-5, "k differs");
        // A late position is rotated, so the comparison is not of two identities.
        assert!(max_abs_diff(&q_ref, &q.transpose(1, 2).unwrap()) > 1e-2);
    }

    #[test]
    fn plain_rope_tables_match_explicit_positions_through_the_model() {
        use super::{build_attention_mask_4d, Qwen3Model};
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{Init, VarBuilder, VarMap};

        let cfg = small_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // A VarMap starts a plain `get` at zero, which would zero every norm; start them near 1.
        let near_one = Init::Randn {
            mean: 1.0,
            stdev: 0.1,
        };
        for (name, dim) in [
            ("norm", cfg.hidden_size),
            ("layers.0.input_layernorm", cfg.hidden_size),
            ("layers.0.post_attention_layernorm", cfg.hidden_size),
            ("layers.0.self_attn.q_norm", cfg.head_dim()),
            ("layers.0.self_attn.k_norm", cfg.head_dim()),
        ] {
            vb.pp(name).get_with_hints(dim, "weight", near_one).unwrap();
        }
        let model = Qwen3Model::new(cfg, vb).unwrap();

        let (b, t) = (2usize, 7usize);
        let ids: Vec<u32> = (0..(b * t) as u32).map(|i| (i * 13) % 1000).collect();
        let input_ids = Tensor::from_vec(ids, (b, t), &device).unwrap();

        // On the CPU this is the additive-mask attention with the plain `[T, D/2]` tables.
        let plain = model.forward_causal(&input_ids, super::Rows::All).unwrap();

        // The same forward with the positions given, which takes the per-row tables.
        let no_padding = Tensor::ones((b, t), DType::F32, &device).unwrap();
        let mask = build_attention_mask_4d(&no_padding).unwrap();
        let positions = Tensor::arange(0u32, t as u32, &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .expand((b, t))
            .unwrap()
            .contiguous()
            .unwrap();
        let embeds = model.embed_tokens(&input_ids).unwrap();
        let per_row = model
            .forward_with_inputs_embeds(&embeds, Some(&mask), None, Some(&positions))
            .unwrap();

        assert!(
            max_abs_diff(&plain, &per_row) < 1e-4,
            "hidden states differ"
        );
        let zeros = plain.zeros_like().unwrap();
        assert!(
            max_abs_diff(&plain, &zeros) > 1e-2,
            "hidden states are zero"
        );
    }

    /// A random model of `layers` layers whose norms start near 1 (a VarMap starts them at 0).
    fn random_model(
        mut cfg: super::Config,
        layers: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> super::Qwen3Model {
        use candle_nn::{Init, VarBuilder, VarMap};
        cfg.num_hidden_layers = layers;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let near_one = Init::Randn {
            mean: 1.0,
            stdev: 0.1,
        };
        vb.pp("norm")
            .get_with_hints(cfg.hidden_size, "weight", near_one)
            .unwrap();
        for i in 0..layers {
            for (name, dim) in [
                ("input_layernorm", cfg.hidden_size),
                ("post_attention_layernorm", cfg.hidden_size),
                ("self_attn.q_norm", cfg.head_dim()),
                ("self_attn.k_norm", cfg.head_dim()),
            ] {
                vb.pp(format!("layers.{i}.{name}"))
                    .get_with_hints(dim, "weight", near_one)
                    .unwrap();
            }
        }
        super::Qwen3Model::new(cfg, vb).unwrap()
    }

    /// A word-level tokenizer over `w0`..`w99` that splits on whitespace and pads on the left,
    /// as the embedding tokenizer does.
    fn word_tokenizer() -> tokenizers::Tokenizer {
        use std::str::FromStr;
        let vocab: Vec<String> = std::iter::once("\"[UNK]\": 0".to_string())
            .chain((0..100).map(|i| format!("\"w{i}\": {}", i + 1)))
            .collect();
        let json = format!(
            r#"{{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {{"type": "WhitespaceSplit"}},
                "post_processor": null, "decoder": null,
                "model": {{"type": "WordLevel", "vocab": {{{}}}, "unk_token": "[UNK]"}}}}"#,
            vocab.join(", ")
        );
        let mut tokenizer = tokenizers::Tokenizer::from_str(&json).unwrap();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        tokenizer
    }

    /// Texts of 5, 2, 9, 5, 1, 9 and 5 words: three lengths repeat, one stands alone.
    fn mixed_texts() -> Vec<String> {
        [5usize, 2, 9, 5, 1, 9, 5]
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                (0..n)
                    .map(|w| format!("w{}", (i * 31 + w * 7) % 100))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    /// The Qwen3-Embedding tokenizer as `from_hf` builds it, truncating to `max_length`.
    #[cfg(feature = "hf-hub")]
    fn qwen3_tokenizer(max_length: usize) -> tokenizers::Tokenizer {
        let path = super::hf_repo("Qwen/Qwen3-Embedding-0.6B")
            .unwrap()
            .get("tokenizer.json")
            .unwrap();
        super::embedding_tokenizer(&path, max_length).unwrap()
    }

    /// Texts far longer than 64 tokens of 16 bytes, in shapes where a cut could change tokens:
    /// runs of whitespace and newlines, Chinese with and without spaces, combining marks, emoji,
    /// digits, a long run of letters, and dashes (about 50 bytes a token, so its prefix comes up
    /// short).
    #[cfg(feature = "hf-hub")]
    fn long_texts() -> Vec<String> {
        let letters: String = (0..3000)
            .map(|i| (b'a' + (i * 7 % 26) as u8) as char)
            .collect();
        vec![
            "the quick brown fox jumps over the lazy dog ".repeat(200),
            "word  \n\n word\t\tword   other\r\nthing  ".repeat(200),
            "我们今天去公园散步然后吃饭".repeat(200),
            "我们今天去公园散步。 然后吃饭了。 ".repeat(200),
            "cafe\u{301} na\u{303}o  ".repeat(500),
            "👍🏽 ok 👨‍👩‍👧 fine ".repeat(300),
            "1234567890 ".repeat(500),
            format!(
                "{} {letters} {}",
                "short words ".repeat(30),
                "tail ".repeat(800)
            ),
            "end.\n\nstart, then: more!? ".repeat(200),
            "a\u{a0} b\u{3000} c ".repeat(500),
            format!("{} ", "-".repeat(120)).repeat(200),
            // 1024 bytes in, a cut at that byte lands inside a long word whose tokens straddle
            // the 63rd: cut there, the word splits into other tokens before the 63rd. Cut at the
            // space before it, the prefix comes up short and the text is encoded whole.
            format!(
                " {}{}",
                format!("{} ", "-".repeat(32)).repeat(24),
                "internationalization incomprehensibilities counterrevolutionaries \
                 electroencephalography uncharacteristically pneumonoultramicroscopic "
                    .repeat(30)
            ),
        ]
    }

    #[cfg(feature = "hf-hub")]
    #[test]
    fn encode_texts_keeps_the_whole_texts_tokens_past_max_length() {
        let max_length = 64;
        let tokenizer = qwen3_tokenizer(max_length);
        let limit = max_length * super::PREFIX_BYTES_PER_TOKEN;
        let mut texts = long_texts();
        assert!(texts.iter().all(|t| t.len() > 4 * limit));
        // Short texts too, so the batch is padded.
        texts.extend(["hello world".to_string(), "a".to_string()]);

        let whole = tokenizer.encode_batch(texts.clone(), true).unwrap();
        let got = super::encode_texts(&tokenizer, &texts).unwrap();
        assert_eq!(got.len(), texts.len());
        for (i, (g, w)) in got.iter().zip(&whole).enumerate() {
            assert_eq!(g.get_ids(), w.get_ids(), "text {i}: ids");
            assert_eq!(
                g.get_attention_mask(),
                w.get_attention_mask(),
                "text {i}: mask"
            );
        }
        // One at a time, with no padding, as well.
        for text in &texts {
            let got = super::encode_texts(&tokenizer, std::slice::from_ref(text)).unwrap();
            let whole = tokenizer.encode(text.as_str(), true).unwrap();
            assert_eq!(got[0].get_ids(), whole.get_ids());
        }

        // Both paths ran: most long texts were read up to a prefix, and the dashes' prefix came
        // up short, so it was encoded whole.
        let prefixed = texts
            .iter()
            .filter(|t| super::token_prefix(t, limit).len() < t.len())
            .count();
        assert!(prefixed >= 8, "only {prefixed} texts were cut");
        let dashes = texts[10].as_str();
        let prefix = super::token_prefix(dashes, limit);
        assert!(prefix.len() < dashes.len());
        assert!(tokenizer
            .encode(prefix, true)
            .unwrap()
            .get_overflowing()
            .is_empty());

        // The dashes beside a short text and nothing longer: the first pass pads the short text
        // only to the dashes' short prefix, so the joined batch must be padded again to the
        // dashes' whole length.
        let pair = vec![texts[10].clone(), "hello world".to_string()];
        let whole = tokenizer.encode_batch(pair.clone(), true).unwrap();
        let got = super::encode_texts(&tokenizer, &pair).unwrap();
        let prefix_len = tokenizer.encode(prefix, true).unwrap().get_ids().len();
        assert!(whole[0].get_ids().len() > prefix_len);
        for (i, (g, w)) in got.iter().zip(&whole).enumerate() {
            assert_eq!(g.get_ids(), w.get_ids(), "pair text {i}: ids");
            assert_eq!(
                g.get_attention_mask(),
                w.get_attention_mask(),
                "pair text {i}: mask"
            );
        }
    }

    #[test]
    fn encode_texts_encodes_whole_texts_when_nothing_truncates() {
        // The word tokenizer has no truncation, so every text is encoded whole.
        let tokenizer = word_tokenizer();
        let texts = ["w1 w2 w3 ".repeat(400), "w4".to_string()];
        let got = super::encode_texts(&tokenizer, &texts).unwrap();
        let whole = tokenizer.encode_batch(texts.to_vec(), true).unwrap();
        for (g, w) in got.iter().zip(&whole) {
            assert_eq!(g.get_ids(), w.get_ids());
            assert_eq!(g.get_attention_mask(), w.get_attention_mask());
        }
        assert_eq!(got[0].len(), 1200);
    }

    #[test]
    fn token_prefix_ends_before_a_space_that_follows_a_non_space() {
        use super::token_prefix;
        assert_eq!(token_prefix("abc def", 10), "abc def");
        assert_eq!(token_prefix("abc def ghi", 9), "abc def");
        assert_eq!(token_prefix("abc def ghi", 7), "abc def");
        assert_eq!(token_prefix("abc def ghi", 6), "abc");
        // A space after whitespace is skipped, back to the run's first space.
        assert_eq!(token_prefix("abc   def ghi", 6), "abc");
        assert_eq!(token_prefix("abc\n def ghi", 5), "abc\n def ghi");
        // Before a space that follows a multi-byte character.
        assert_eq!(token_prefix("ab日 cd ef", 8), "ab日 cd");
        assert_eq!(token_prefix("ab日 cdefg", 8), "ab日");
        // No such space: the whole text.
        assert_eq!(token_prefix("abcdefghij", 4), "abcdefghij");
        assert_eq!(token_prefix(" abcdefghij", 4), " abcdefghij");
    }

    #[test]
    fn embed_rejects_a_text_with_no_tokens() {
        let model = super::Qwen3TextEmbedding::new(
            random_model(
                small_config(),
                2,
                &candle_core::Device::Cpu,
                candle_core::DType::F32,
            ),
            word_tokenizer(),
        );
        let texts = mixed_texts();
        assert!(model.embed(&["", texts[0].as_ref()]).is_err());
        assert!(model.embed_padded(&[""]).is_err());
    }

    #[test]
    fn embed_gives_each_text_its_vector_alone_in_the_callers_order() {
        let model = super::Qwen3TextEmbedding::new(
            random_model(
                small_config(),
                2,
                &candle_core::Device::Cpu,
                candle_core::DType::F32,
            ),
            word_tokenizer(),
        );
        let texts = mixed_texts();
        let together = model.embed(&texts).unwrap();
        assert_eq!(together.len(), texts.len());
        for (i, text) in texts.iter().enumerate() {
            let alone = model.embed(std::slice::from_ref(text)).unwrap();
            if [1, 4].contains(&i) {
                // The only text of its length runs as a batch of one, exactly as alone.
                assert_eq!(together[i], alone[0], "text {i} differs from alone");
            } else {
                // In a group of equal-length texts the matmuls sum in another order.
                let diff = together[i]
                    .iter()
                    .zip(&alone[0])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(diff < 1e-5, "text {i} differs from alone by {diff}");
            }
        }
        // Different texts give different vectors, so a mixed-up order would show.
        assert_ne!(together[0], together[1]);
        // One padded batch (today's `embed`) does not give a text its vector alone bit for bit.
        let padded = model.embed_padded(&texts).unwrap();
        let alone = model.embed(&texts[1..2]).unwrap();
        assert_ne!(padded[1], alone[0], "padding changed nothing");
    }

    #[test]
    fn output_dim_keeps_the_leading_dimensions_normalized() {
        let model = super::Qwen3TextEmbedding::new(
            random_model(
                small_config(),
                2,
                &candle_core::Device::Cpu,
                candle_core::DType::F32,
            ),
            word_tokenizer(),
        );
        let texts = ["w1 w2 w3", "w4 w5", "w6 w7 w8 w9"];
        let hidden = small_config().hidden_size;
        let full = model.embed(&texts).unwrap();
        assert_eq!(model.output_dim(), hidden);

        // The hidden size itself is the full vector, bit for bit.
        let model = model.with_output_dim(hidden).unwrap();
        assert_eq!(model.embed(&texts).unwrap(), full);

        let dim = 16;
        let model = model.with_output_dim(dim).unwrap();
        assert_eq!(model.output_dim(), dim);
        let short = model.embed(&texts).unwrap();
        for padded in model.embed_padded(&texts).unwrap() {
            assert_eq!(padded.len(), dim);
        }
        for (s, f) in short.iter().zip(&full) {
            assert_eq!(s.len(), dim);
            let norm: f32 = f[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            for (a, b) in s.iter().zip(&f[..dim]) {
                assert!((a - b / norm).abs() < 1e-6, "{a} against {}", b / norm);
            }
        }

        assert!(model.with_output_dim(hidden + 1).is_err());
    }

    #[test]
    fn reranker_scores_each_row_as_yes_against_no_at_its_last_position() {
        use candle_core::{DType, Device, IndexOp, Tensor};

        let cfg = small_config();
        let hidden = cfg.hidden_size;
        let model = random_model(cfg, 2, &Device::Cpu, DType::F32);
        let no_yes = Tensor::randn(0f32, 1.0, (2, hidden), &Device::Cpu).unwrap();
        let reranker =
            super::Qwen3Reranker::new(model, word_tokenizer(), no_yes.clone(), 512).unwrap();

        // Three rows of different lengths, run as one left-padded batch.
        let rows: Vec<Vec<u32>> = vec![(1..9).collect(), (20..23).collect(), (40..52).collect()];
        let together = reranker.score_rows(&rows).unwrap();
        assert_eq!(together.len(), 3);
        for (i, row) in rows.iter().enumerate() {
            // By hand: the row alone, its last position's normed state against the two rows.
            let ids = Tensor::new(row.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let h = reranker
                .model
                .forward_causal(&ids, super::Rows::Last)
                .unwrap()
                .i((0, 0))
                .unwrap();
            let logits = no_yes
                .matmul(&h.unsqueeze(1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let yes = 1.0 / (1.0 + (logits[0] - logits[1]).exp());
            assert!(
                (together[i] - yes).abs() < 1e-5,
                "row {i}: {} against {yes}",
                together[i]
            );
        }
        assert!(together.iter().all(|p| (0.0..=1.0).contains(p)));
        assert_ne!(together[0], together[1]);

        // The prompt wraps every pair, and a max length with no room for a document is refused.
        let scores = reranker.score(None, "w1 w2", &["w3 w4 w5", "w6"]).unwrap();
        assert_eq!(scores.len(), 2);
        let model = random_model(small_config(), 1, &Device::Cpu, DType::F32);
        assert!(super::Qwen3Reranker::new(model, word_tokenizer(), no_yes, 4).is_err());
    }

    #[test]
    fn embed_layers_returns_the_output_and_earlier_layers_from_one_pass() {
        use super::{build_attention_mask_4d, l2_normalize, AttentionMask, RotaryTables, Rows};
        use candle_core::{DType, Device, IndexOp, Tensor};

        let layers = 3;
        let model = super::Qwen3TextEmbedding::new(
            random_model(small_config(), layers, &Device::Cpu, DType::F32),
            word_tokenizer(),
        );
        let texts = ["w1 w2 w3 w4 w5", "w6 w7"];
        let got = model.embed_layers(&texts, &[layers, 1, 2]).unwrap();
        assert_eq!(got.len(), 3);

        // The model's output is `embed`'s, bit for bit.
        assert_eq!(got[0], model.embed(&texts).unwrap());

        // Layer 1 of the unpadded first text is the first layer run by hand, its last position
        // normalized.
        let alone = model.embed_layers(&texts[..1], &[1]).unwrap();
        let ids = Tensor::new(&[[2u32, 3, 4, 5, 6]], &Device::Cpu).unwrap();
        let hs = model.model.embed_tokens(&ids).unwrap();
        let mask =
            build_attention_mask_4d(&Tensor::ones((1, 5), DType::F32, &Device::Cpu).unwrap())
                .unwrap();
        let (cos, sin) = model
            .model
            .rotary_emb
            .plain_tables(5, DType::F32, &Device::Cpu)
            .unwrap();
        let hs = model.model.layers[0]
            .forward(
                &hs,
                AttentionMask::Additive(&mask),
                RotaryTables::Plain(&cos, &sin),
                Rows::All,
            )
            .unwrap();
        let by_hand = l2_normalize(&hs.i((.., 4)).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(alone[0], by_hand);

        // In the padded batch every layer's vector matches the text alone to rounding.
        for (layer, vs) in [layers, 1, 2].iter().zip(&got) {
            for (i, text) in texts.iter().enumerate() {
                let one = model
                    .embed_layers(std::slice::from_ref(text), &[*layer])
                    .unwrap();
                let diff = vs[i]
                    .iter()
                    .zip(&one[0][0])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(diff < 1e-5, "layer {layer} text {i} differs by {diff}");
            }
        }
        // The layers differ, so a tap taken at the wrong layer would show.
        assert_ne!(got[1], got[2]);
        assert_ne!(got[2], got[0]);

        assert!(model.embed_layers(&texts, &[0]).is_err());
        assert!(model.embed_layers(&texts, &[layers + 1]).is_err());

        // A shorter output shortens the model's output, as `embed` does, and leaves earlier
        // layers at the full hidden size.
        let hidden = small_config().hidden_size;
        let model = model.with_output_dim(16).unwrap();
        let short = model.embed_layers(&texts, &[layers, 1]).unwrap();
        assert_eq!(short[0], model.embed(&texts).unwrap());
        assert!(short[0].iter().all(|v| v.len() == 16));
        assert!(short[1].iter().all(|v| v.len() == hidden));
        assert_eq!(short[1], got[1]);
    }

    /// The last layer run for the last position alone gives the last row the full run gives:
    /// causal, left-padded, and with explicit position ids (the per-row tables).
    fn assert_last_row_matches(model: &super::Qwen3Model, tolerance: f32) {
        use super::{AttentionMask, Rows};
        use candle_core::{DType, IndexOp, Tensor};

        let device = model.device().clone();
        let (b, t) = (2usize, 37usize);
        let ids: Vec<u32> = (0..(b * t) as u32).map(|i| (i * 13) % 1000).collect();
        let input_ids = Tensor::from_vec(ids, (b, t), &device).unwrap();
        let check = |all: Tensor, last: Tensor, what: &str| {
            assert_eq!(all.dims(), &[b, t, model.config().hidden_size], "{what}");
            assert_eq!(last.dims(), &[b, 1, model.config().hidden_size], "{what}");
            let all_last = all.i((.., t - 1..t)).unwrap();
            let diff = max_abs_diff(
                &all_last.to_dtype(DType::F32).unwrap(),
                &last.to_dtype(DType::F32).unwrap(),
            );
            assert!(diff <= tolerance, "{what}: the last row differs by {diff}");
            // The last row is not the first: positions really differ, so a wrong row shows.
            let first = all.i((.., 0..1)).unwrap();
            assert!(
                max_abs_diff(
                    &first.to_dtype(DType::F32).unwrap(),
                    &last.to_dtype(DType::F32).unwrap()
                ) > 1e-2,
                "{what}"
            );
        };

        check(
            model.forward_causal(&input_ids, Rows::All).unwrap(),
            model.forward_causal(&input_ids, Rows::Last).unwrap(),
            "causal",
        );

        // The second row left-padded by eleven.
        let mut mask = vec![1f32; b * t];
        mask[t..t + 11].fill(0.0);
        let mask_2d = Tensor::from_vec(mask, (b, t), &device).unwrap();
        check(
            model
                .forward_padded(&input_ids, &mask_2d, Rows::All)
                .unwrap(),
            model
                .forward_padded(&input_ids, &mask_2d, Rows::Last)
                .unwrap(),
            "padded",
        );

        if !device.is_metal() {
            let mask = super::build_attention_mask_4d(&mask_2d).unwrap();
            let positions = Tensor::arange(0u32, t as u32, &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .expand((b, t))
                .unwrap()
                .contiguous()
                .unwrap();
            let embeds = model.embed_tokens(&input_ids).unwrap();
            let run = |rows| {
                model
                    .forward_embeds_masked(
                        &embeds,
                        AttentionMask::Additive(&mask),
                        None,
                        Some(&positions),
                        rows,
                    )
                    .unwrap()
            };
            check(run(Rows::All), run(Rows::Last), "per-row tables");
        }
    }

    #[test]
    fn last_position_alone_matches_the_full_last_layer_on_the_cpu() {
        let model = random_model(
            small_config(),
            2,
            &candle_core::Device::Cpu,
            candle_core::DType::F32,
        );
        assert_last_row_matches(&model, 1e-5);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn last_position_alone_matches_the_full_last_layer_on_metal() {
        // Head dim 64, one the fused attention kernel takes.
        let mut cfg = small_config();
        cfg.head_dim = Some(64);
        cfg.hidden_size = 256;
        let device = candle_core::Device::new_metal(0).unwrap();
        let model = random_model(cfg, 2, &device, candle_core::DType::F32);
        assert_last_row_matches(&model, 1e-4);
    }

    /// The same random head-dim-128 model on the CPU in f32 and on Metal in F32 and BF16: every
    /// pooled output (`embed` over mixed lengths, `embed_padded`, a 32-number output and an
    /// earlier layer's vector) matches the CPU f32 vectors, and every vector it returns is unit
    /// length, whatever dtype the model runs in.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_vectors_match_the_cpu_and_are_unit_length_in_f32_and_bf16() {
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;

        let mut cfg = small_config();
        cfg.head_dim = Some(128);
        cfg.hidden_size = 256;
        cfg.intermediate_size = 512;
        cfg.num_hidden_layers = 2;
        let (_, varmap) = random_model_and_weights(&cfg);
        let weights: std::collections::HashMap<String, candle_core::Tensor> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect();
        let load = |device: &Device, dtype: DType| {
            let vb = VarBuilder::from_tensors(weights.clone(), dtype, device);
            super::Qwen3TextEmbedding::new(
                super::Qwen3Model::new(cfg.clone(), vb).unwrap(),
                word_tokenizer(),
            )
        };

        /// Every pooled output the wrapper returns, in one list of vectors.
        fn outputs(model: super::Qwen3TextEmbedding, texts: &[String]) -> Vec<Vec<f32>> {
            let mut all = model.embed(texts).unwrap();
            all.extend(model.embed_padded(texts).unwrap());
            all.extend(model.embed_layers(texts, &[1]).unwrap().remove(0));
            let short = model.with_output_dim(32).unwrap();
            all.extend(short.embed(texts).unwrap());
            all.extend(short.embed_padded(texts).unwrap());
            all
        }
        let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            dot / (norm(a) * norm(b))
        };

        let texts = mixed_texts();
        let reference = outputs(load(&Device::Cpu, DType::F32), &texts);
        let metal = Device::new_metal(0).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            let got = outputs(load(&metal, dtype), &texts);
            assert_eq!(got.len(), reference.len());
            let (mut worst_norm, mut worst_diff, mut worst_cos) = (0f64, 0f32, 1f64);
            for (g, r) in got.iter().zip(&reference) {
                assert_eq!(g.len(), r.len(), "{dtype:?}");
                worst_norm = worst_norm.max((norm(g) - 1.0).abs());
                let diff = g
                    .iter()
                    .zip(r)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                worst_diff = worst_diff.max(diff);
                worst_cos = worst_cos.min(cosine(g, r));
            }
            eprintln!(
                "{dtype:?}: norm off 1 by at most {worst_norm:e}, max diff {worst_diff:e}, \
                 min cosine {worst_cos}"
            );
            assert!(
                worst_norm < 1e-4,
                "{dtype:?}: a norm is off 1 by {worst_norm:e}"
            );
            if dtype == DType::F32 {
                assert!(worst_diff < 1e-4, "f32: a vector differs by {worst_diff}");
            } else {
                assert!(worst_cos > 0.999, "bf16: a vector has cosine {worst_cos}");
            }
        }
    }

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

    /// A random f32 model on the CPU, and its VarMap for writing the same weights to GGUF.
    fn random_model_and_weights(cfg: &super::Config) -> (super::Qwen3Model, candle_nn::VarMap) {
        use candle_core::{DType, Device};
        use candle_nn::{Init, VarBuilder, VarMap};

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let near_one = Init::Randn {
            mean: 1.0,
            stdev: 0.1,
        };
        vb.pp("norm")
            .get_with_hints(cfg.hidden_size, "weight", near_one)
            .unwrap();
        for i in 0..cfg.num_hidden_layers {
            for (name, dim) in [
                ("input_layernorm", cfg.hidden_size),
                ("post_attention_layernorm", cfg.hidden_size),
                ("self_attn.q_norm", cfg.head_dim()),
                ("self_attn.k_norm", cfg.head_dim()),
            ] {
                vb.pp(format!("layers.{i}.{name}"))
                    .get_with_hints(dim, "weight", near_one)
                    .unwrap();
            }
        }
        (super::Qwen3Model::new(cfg.clone(), vb).unwrap(), varmap)
    }

    /// llama.cpp's `qwen3` layout, with the norms in f32 and everything else in `matrix_dtype`.
    fn gguf_bytes(
        cfg: &super::Config,
        varmap: &candle_nn::VarMap,
        matrix_dtype: candle_core::quantized::GgmlDType,
    ) -> Vec<u8> {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

        let mut renames = vec![
            ("embed_tokens".to_string(), "token_embd".to_string()),
            ("norm".to_string(), "output_norm".to_string()),
        ];
        for i in 0..cfg.num_hidden_layers {
            for (from, to) in [
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_output"),
                ("self_attn.q_norm", "attn_q_norm"),
                ("self_attn.k_norm", "attn_k_norm"),
                ("mlp.gate_proj", "ffn_gate"),
                ("mlp.up_proj", "ffn_up"),
                ("mlp.down_proj", "ffn_down"),
                ("input_layernorm", "attn_norm"),
                ("post_attention_layernorm", "ffn_norm"),
            ] {
                renames.push((format!("layers.{i}.{from}"), format!("blk.{i}.{to}")));
            }
        }
        let data = varmap.data().lock().unwrap();
        assert_eq!(data.len(), renames.len(), "a weight has no GGUF name");
        let tensors: Vec<(String, QTensor)> = renames
            .iter()
            .map(|(from, to)| {
                let t = data[&format!("{from}.weight")].as_tensor();
                let dtype = if t.rank() == 1 {
                    GgmlDType::F32
                } else {
                    matrix_dtype
                };
                (format!("{to}.weight"), QTensor::quantize(t, dtype).unwrap())
            })
            .collect();
        let u32_value = |v: usize| gguf_file::Value::U32(v as u32);
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("qwen3".into()),
            ),
            ("qwen3.attention.key_length", u32_value(cfg.head_dim())),
            ("qwen3.embedding_length", u32_value(cfg.hidden_size)),
            (
                "qwen3.feed_forward_length",
                u32_value(cfg.intermediate_size),
            ),
            (
                "qwen3.context_length",
                u32_value(cfg.max_position_embeddings),
            ),
            (
                "qwen3.attention.head_count",
                u32_value(cfg.num_attention_heads),
            ),
            ("qwen3.block_count", u32_value(cfg.num_hidden_layers)),
            (
                "qwen3.attention.head_count_kv",
                u32_value(cfg.num_key_value_heads),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(cfg.rms_norm_eps as f32),
            ),
            (
                "qwen3.rope.freq_base",
                gguf_file::Value::F32(cfg.rope_theta as f32),
            ),
        ];
        let metadata: Vec<(&str, &gguf_file::Value)> =
            metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensors: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut out = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut out, &metadata, &tensors).unwrap();
        out.into_inner()
    }

    /// Reads a GGUF file, lets `edit` change its metadata, and loads it.
    fn load_gguf_with(
        bytes: Vec<u8>,
        device: &candle_core::Device,
        edit: impl FnOnce(&mut candle_core::quantized::gguf_file::Content),
    ) -> candle_core::Result<super::Qwen3Model> {
        use candle_core::quantized::gguf_file;

        let mut reader = std::io::Cursor::new(bytes);
        let mut content = gguf_file::Content::read(&mut reader).unwrap();
        edit(&mut content);
        super::Qwen3Model::from_gguf(&content, &mut reader, device)
    }

    fn load_gguf(bytes: Vec<u8>, device: &candle_core::Device) -> super::Qwen3Model {
        load_gguf_with(bytes, device, |_| {}).unwrap()
    }

    fn small_model() -> (super::Qwen3Model, candle_nn::VarMap) {
        random_model_and_weights(&small_config())
    }

    fn small_gguf_bytes(
        varmap: &candle_nn::VarMap,
        matrix_dtype: candle_core::quantized::GgmlDType,
    ) -> Vec<u8> {
        gguf_bytes(&small_config(), varmap, matrix_dtype)
    }

    fn small_gguf_model(
        varmap: &candle_nn::VarMap,
        matrix_dtype: candle_core::quantized::GgmlDType,
    ) -> super::Qwen3Model {
        load_gguf(
            small_gguf_bytes(varmap, matrix_dtype),
            &candle_core::Device::Cpu,
        )
    }

    /// The smallest row cosine between two flattened `[.., hidden]` state sets.
    fn min_row_cosine(a: &[f32], b: &[f32]) -> f32 {
        let hidden = small_config().hidden_size;
        a.chunks(hidden)
            .zip(b.chunks(hidden))
            .map(|(x, y)| {
                let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
                let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
                let ny: f32 = y.iter().map(|q| q * q).sum::<f32>().sqrt();
                dot / (nx * ny)
            })
            .fold(f32::MAX, f32::min)
    }

    /// The hidden states of a causal batch and a left-padded batch, flattened, for `rows`.
    fn small_forwards(model: &super::Qwen3Model, rows: super::Rows) -> (Vec<f32>, Vec<f32>) {
        use candle_core::{Device, Tensor};

        let (b, t) = (2usize, 7usize);
        let ids: Vec<u32> = (0..(b * t) as u32).map(|i| (i * 13) % 1000).collect();
        let input_ids = Tensor::from_vec(ids, (b, t), &Device::Cpu).unwrap();
        let causal = model.forward_causal(&input_ids, rows).unwrap();
        let mask: Vec<f32> = (0..b * t).map(|i| if i < 3 { 0.0 } else { 1.0 }).collect();
        let mask = Tensor::from_vec(mask, (b, t), &Device::Cpu).unwrap();
        let padded = model.forward_padded(&input_ids, &mask, rows).unwrap();
        let flat = |x: Tensor| x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (flat(causal), flat(padded))
    }

    #[test]
    fn gguf_reads_the_config_from_its_metadata() {
        use candle_core::quantized::GgmlDType;

        let (_, varmap) = small_model();
        let model = small_gguf_model(&varmap, GgmlDType::F32);
        let (got, want) = (model.config(), small_config());
        assert_eq!(got.hidden_size, want.hidden_size);
        assert_eq!(got.intermediate_size, want.intermediate_size);
        assert_eq!(got.head_dim(), want.head_dim());
        assert_eq!(got.num_attention_heads, want.num_attention_heads);
        assert_eq!(got.num_key_value_heads, want.num_key_value_heads);
        assert_eq!(got.num_hidden_layers, want.num_hidden_layers);
        assert_eq!(got.vocab_size, want.vocab_size);
        assert_eq!(got.rope_theta, want.rope_theta);
        assert_eq!(got.rms_norm_eps as f32, want.rms_norm_eps as f32);
    }

    #[test]
    fn gguf_f32_model_is_bit_identical_to_the_safetensors_model() {
        use super::Rows;
        use candle_core::quantized::GgmlDType;

        let (dense, varmap) = small_model();
        // An f32 matrix stays a plain tensor, which `QMatMul` multiplies as `Linear` does.
        let gguf = small_gguf_model(&varmap, GgmlDType::F32);
        let (dense_causal, dense_padded) = small_forwards(&dense, Rows::All);
        let (gguf_causal, gguf_padded) = small_forwards(&gguf, Rows::All);
        assert_eq!(gguf_causal, dense_causal);
        assert_eq!(gguf_padded, dense_padded);
        assert!(
            dense_causal.iter().any(|v| v.abs() > 1e-2),
            "states are zero"
        );
        // The GGUF path copies the last-position view before it multiplies, so the sums round
        // differently.
        let (dense_causal, dense_padded) = small_forwards(&dense, Rows::Last);
        let (gguf_causal, gguf_padded) = small_forwards(&gguf, Rows::Last);
        for (a, b) in [(&dense_causal, &gguf_causal), (&dense_padded, &gguf_padded)] {
            let diff = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(diff < 1e-5, "the last position differs by {diff}");
        }
    }

    #[test]
    fn gguf_q8_0_model_stays_close_to_the_safetensors_model() {
        use super::Rows;
        use candle_core::quantized::GgmlDType;

        let (dense, varmap) = small_model();
        let (dense_causal, dense_padded) = small_forwards(&dense, Rows::All);
        let q8 = small_gguf_model(&varmap, GgmlDType::Q8_0);
        let (q8_causal, q8_padded) = small_forwards(&q8, Rows::All);
        for (a, b) in [(&dense_causal, &q8_causal), (&dense_padded, &q8_padded)] {
            assert_ne!(a, b, "Q8_0 changed nothing");
            let cosine = min_row_cosine(a, b);
            assert!(cosine > 0.999, "Q8_0 row cosine {cosine}");
        }
    }

    #[test]
    fn gguf_last_position_alone_matches_the_full_last_layer_on_the_cpu() {
        use candle_core::quantized::GgmlDType;

        let mut cfg = small_config();
        cfg.num_hidden_layers = 2;
        let (_, varmap) = random_model_and_weights(&cfg);
        let model = load_gguf(
            gguf_bytes(&cfg, &varmap, GgmlDType::Q8_0),
            &candle_core::Device::Cpu,
        );
        assert_last_row_matches(&model, 1e-5);
        assert_one_row_last_position_matches(&model, &candle_core::Device::Cpu, 1e-5);
    }

    /// With one row, the last position is a contiguous view with an offset.
    fn assert_one_row_last_position_matches(
        model: &super::Qwen3Model,
        device: &candle_core::Device,
        tolerance: f32,
    ) {
        use super::Rows;
        use candle_core::{IndexOp, Tensor};

        let t = 23usize;
        let ids: Vec<u32> = (0..t as u32).map(|i| (i * 17) % 1000).collect();
        let input_ids = Tensor::from_vec(ids, (1, t), device).unwrap();
        let all = model.forward_causal(&input_ids, Rows::All).unwrap();
        let last = model.forward_causal(&input_ids, Rows::Last).unwrap();
        let diff = max_abs_diff(&all.i((.., t - 1..t)).unwrap(), &last);
        assert!(
            diff < tolerance,
            "one row: the last position differs by {diff}"
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn gguf_last_position_alone_matches_the_full_last_layer_on_metal() {
        use candle_core::quantized::GgmlDType;

        // Head dim 64, one the fused attention kernel takes.
        let mut cfg = small_config();
        cfg.head_dim = Some(64);
        cfg.hidden_size = 256;
        cfg.num_hidden_layers = 2;
        let (_, varmap) = random_model_and_weights(&cfg);
        let device = candle_core::Device::new_metal(0).unwrap();
        let model = load_gguf(gguf_bytes(&cfg, &varmap, GgmlDType::Q8_0), &device);
        // The mat-vec and mat-mat kernels round differently.
        assert_last_row_matches(&model, 5e-3);
        assert_one_row_last_position_matches(&model, &device, 5e-3);
    }

    #[test]
    fn gguf_rejects_another_architecture() {
        use candle_core::quantized::{gguf_file, GgmlDType};
        use candle_core::Device;

        let (_, varmap) = small_model();
        let err = load_gguf_with(
            small_gguf_bytes(&varmap, GgmlDType::F32),
            &Device::Cpu,
            |content| {
                content.metadata.insert(
                    "general.architecture".into(),
                    gguf_file::Value::String("llama".into()),
                );
            },
        )
        .err()
        .expect("a llama file loads as qwen3");
        assert!(err.to_string().contains("not `qwen3`"), "{err}");
    }

    #[test]
    fn gguf_rejects_a_tensor_that_does_not_match_the_metadata() {
        use candle_core::quantized::{gguf_file, GgmlDType};
        use candle_core::Device;

        let (_, varmap) = small_model();
        let wrong = small_config().intermediate_size as u32 + 1;
        let err = load_gguf_with(
            small_gguf_bytes(&varmap, GgmlDType::F32),
            &Device::Cpu,
            |content| {
                content.metadata.insert(
                    "qwen3.feed_forward_length".into(),
                    gguf_file::Value::U32(wrong),
                );
            },
        )
        .err()
        .expect("a mismatched ffn_gate loads");
        assert!(err.to_string().contains("blk.0.ffn_gate.weight"), "{err}");
    }

    #[test]
    fn gguf_reads_other_integer_types_and_no_key_length() {
        use candle_core::quantized::{gguf_file, GgmlDType};
        use candle_core::Device;

        let (_, varmap) = small_model();
        let want = small_config();
        let model = load_gguf_with(
            small_gguf_bytes(&varmap, GgmlDType::F32),
            &Device::Cpu,
            |content| {
                content.metadata.remove("qwen3.attention.key_length");
                content.metadata.insert(
                    "qwen3.block_count".into(),
                    gguf_file::Value::U64(want.num_hidden_layers as u64),
                );
            },
        )
        .unwrap();
        assert_eq!(model.config().head_dim(), want.head_dim());
        assert_eq!(model.config().num_hidden_layers, want.num_hidden_layers);
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

    #[cfg(feature = "qwen3-vl")]
    #[test]
    fn finds_token_spans() {
        let ids = vec![1u32, 42, 42, 7, 42, 42, 42, 9];
        let spans = find_token_spans(&ids, 42);
        assert_eq!(spans, vec![(1, 3), (4, 7)]);
    }

    #[cfg(feature = "qwen3-vl")]
    #[test]
    fn expands_image_placeholders() {
        let prompt = "<|vision_start|><|image_pad|><|vision_end|>";
        let expanded = expand_image_token_placeholders(prompt, 3).unwrap();
        assert_eq!(
            expanded,
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[cfg(feature = "qwen3-vl")]
    #[test]
    fn rounds_half_to_even_like_python() {
        assert_eq!(round_ties_to_even(12.5), 12);
        assert_eq!(round_ties_to_even(13.5), 14);
        assert_eq!(round_ties_to_even(9.5625), 10);
    }
}
