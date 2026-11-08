//! BERT-family text embeddings on the candle backend, through candle's fused
//! attention kernel.
//!
//! `candle-transformers`' BERT runs attention as a sequence of per-head tensor
//! ops; on Metal that leaves the GPU mostly idle (measured ~3 TFLOP/s against
//! ~9 for MLX on the same checkpoint). This encoder keeps the same math but
//! hands attention to `candle_nn::ops::sdpa` — the fused Metal kernel the Qwen3
//! path already uses — and flattens the projections to 2D matmuls.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::{Tokenizer, TruncationParams};

fn map_err<E: std::fmt::Display>(err: E) -> candle_core::Error {
    candle_core::Error::Msg(err.to_string())
}

#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
}

impl BertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

struct BertLayer {
    qkv: Linear,
    out_proj: Linear,
    /// Projection bias, LayerNorm weight and LayerNorm bias, one `[3, h]` row stack.
    attn_wb: Tensor,
    intermediate: Linear,
    up_bias: Tensor,
    output: Linear,
    /// Same stack for the MLP output projection.
    out_wb: Tensor,
    eps: f64,
}

impl BertLayer {
    fn load(vb: VarBuilder, cfg: &BertConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        // One `[h, 3h]` projection instead of three `[h, h]` ones: a single wider
        // matmul feeds the fused attention directly and saves two kernel launches
        // per layer.
        let get = |name: &str, shape: (usize, usize)| vb.get(shape, name);
        let q_w = get("attention.self.query.weight", (h, h))?;
        let k_w = get("attention.self.key.weight", (h, h))?;
        let v_w = get("attention.self.value.weight", (h, h))?;
        let q_b = vb.get((h,), "attention.self.query.bias")?;
        let k_b = vb.get((h,), "attention.self.key.bias")?;
        let v_b = vb.get((h,), "attention.self.value.bias")?;
        let qkv = Linear::new(
            Tensor::cat(&[q_w, k_w, v_w], 0)?,
            Some(Tensor::cat(&[q_b, k_b, v_b], 0)?),
        );
        let stack = |names: [&str; 3]| -> Result<Tensor> {
            let mut rows = Vec::with_capacity(3);
            for (name, width) in [(names[0], h), (names[1], h), (names[2], h)] {
                rows.push(vb.get((width,), name)?);
            }
            Tensor::cat(&rows, 0)?.reshape((3, h))
        };
        let attn_wb = stack([
            "attention.output.dense.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.LayerNorm.bias",
        ])?;
        let out_wb = stack([
            "output.dense.bias",
            "output.LayerNorm.weight",
            "output.LayerNorm.bias",
        ])?;
        let up_bias = vb.get((cfg.intermediate_size,), "intermediate.dense.bias")?;
        Ok(Self {
            qkv,
            out_proj: linear_no_bias(h, h, vb.pp("attention.output.dense"))?,
            attn_wb,
            intermediate: linear_no_bias(h, cfg.intermediate_size, vb.pp("intermediate.dense"))?,
            up_bias,
            output: linear_no_bias(cfg.intermediate_size, h, vb.pp("output.dense"))?,
            out_wb,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let h = heads * head_dim;
        let flat = xs.reshape((b * t, h))?;
        let qkv = self.qkv.forward(&flat)?.reshape((b, t, 3 * h))?;
        let q = qkv
            .narrow(2, 0, h)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?;
        let k = qkv
            .narrow(2, h, h)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?;
        let v = qkv
            .narrow(2, 2 * h, h)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?;
        let attn = attention(&q, &k, &v, mask, scale)?.reshape((b * t, h))?;
        let attn = self.out_proj.forward(&attn)?;
        let xs = crate::models::bert_fused_ops::add_layernorm(
            &attn,
            &xs.reshape((b * t, h))?,
            &self.attn_wb,
            self.eps,
        )?;
        let flat = xs.reshape((b * t, h))?;
        let up = self.intermediate.forward(&flat)?;
        let up = crate::models::bert_fused_ops::bias_gelu(&up, &self.up_bias)?;
        let mlp = self.output.forward(&up)?;
        let fused = crate::models::bert_fused_ops::add_layernorm(
            &mlp,
            &xs.reshape((b * t, h))?,
            &self.out_wb,
            self.eps,
        )?;
        fused.reshape((b, t, h))
    }
}

/// Fused attention on Metal, plain tensor math elsewhere (candle's `sdpa` has no
/// CPU kernel, and the reference path is what the parity test compares against).
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    if q.device().is_metal() {
        crate::models::qwen3_sdpa_bthd::sdpa_bthd(q, k, v, mask, false, scale)
    } else {
        let q = q.contiguous()?;
        let k = k.transpose(2, 3)?.contiguous()?;
        let v = v.contiguous()?;
        let mut scores = q.matmul(&k)?.affine(scale as f64, 0.0)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.matmul(&v)?.transpose(1, 2)?.contiguous()
    }
}

/// A BERT encoder used as a text embedder (CLS pooling + L2 normalization).
pub struct BertTextEmbedding {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embeddings_ln: LayerNorm,
    layers: Vec<BertLayer>,
    cfg: BertConfig,
    tokenizer: Option<Tokenizer>,
    max_length: usize,
    batch_size: usize,
    pad_id: u32,
}

impl BertTextEmbedding {
    /// Load a BERT-family checkpoint (`config.json`, `tokenizer.json`, safetensors) from a
    /// Hugging Face repo. `dtype` applies to the weights.
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        let repo =
            crate::common::pull_from_hf(repo_id.to_string(), crate::get_cache_dir().into(), true)
                .map_err(map_err)?;
        let cfg_path: PathBuf = repo.get("config.json").map_err(map_err)?;
        let cfg_bytes = std::fs::read(&cfg_path).map_err(map_err)?;
        let cfg: BertConfig = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let weight_files = crate::common::safetensors_weight_files(&repo).map_err(map_err)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };

        // HF BERT checkpoints prefix every tensor with `bert.`; accept a bare
        // prefix too (some converted sentence-transformer repos drop it).
        let load = |vb: VarBuilder| -> Result<Self> { Self::load_weights(vb, &cfg, dtype) };
        let mut this = match load(vb.clone().pp("bert")) {
            Ok(m) => m,
            Err(_) => load(vb)?,
        };

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(map_err)?;
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(map_err)?;
        this.tokenizer = Some(tokenizer);
        this.max_length = max_length;
        this.pad_id = pad_id;
        Ok(this)
    }

    fn load_weights(vb: VarBuilder, cfg: &BertConfig, dtype: DType) -> Result<Self> {
        let h = cfg.hidden_size;
        let word_embeddings = embedding(cfg.vocab_size, h, vb.pp("embeddings.word_embeddings"))?;
        let position_embeddings = embedding(
            cfg.max_position_embeddings,
            h,
            vb.pp("embeddings.position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            cfg.type_vocab_size,
            h,
            vb.pp("embeddings.token_type_embeddings"),
        )?;
        let embeddings_ln = layer_norm(h, cfg.layer_norm_eps, vb.pp("embeddings.LayerNorm"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(BertLayer::load(vb.pp(format!("encoder.layer.{i}")), cfg)?);
        }
        let _ = dtype;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_ln,
            layers,
            cfg: cfg.clone(),
            tokenizer: None,
            max_length: 0,
            batch_size: 16,
            pad_id: 0,
        })
    }

    /// Texts embedded per forward pass (padded to the batch's longest row).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Embed texts in order, returning normalized vectors of `hidden_size` floats.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        let device = self.word_embeddings.embeddings().device();
        let inputs: Vec<&str> = texts.iter().map(|t| t.as_ref()).collect();
        let encodings = self
            .tokenizer
            .as_ref()
            .expect("tokenizer is set by from_hf")
            .encode_batch(inputs, true)
            .map_err(map_err)?;
        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].get_ids().len());
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); encodings.len()];
        for chunk in order.chunks(self.batch_size) {
            let seq = chunk
                .iter()
                .map(|&i| encodings[i].get_ids().len())
                .max()
                .unwrap_or(0);
            if seq == 0 {
                continue;
            }
            let batch = chunk.len();
            let mut ids = Vec::with_capacity(batch * seq);
            let mut mask = Vec::with_capacity(batch * seq);
            for &i in chunk {
                let e = &encodings[i];
                let n = e.get_ids().len();
                ids.extend_from_slice(e.get_ids());
                ids.extend(std::iter::repeat_n(self.pad_id, seq - n));
                mask.extend(std::iter::repeat_n(1u32, n));
                mask.extend(std::iter::repeat_n(0u32, seq - n));
            }
            let input_ids = Tensor::from_vec(ids, (batch, seq), device)?;
            let token_type_ids = Tensor::zeros((batch, seq), DType::U32, device)?;
            let attention_mask = Tensor::from_vec(mask, (batch, seq), device)?;
            let padded = chunk.iter().any(|&i| encodings[i].get_ids().len() < seq);
            let hidden = self.forward(&input_ids, &token_type_ids, &attention_mask, padded)?;
            let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
            let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
            for (row, &i) in cls
                .broadcast_div(&norm)?
                .to_vec2::<f32>()?
                .iter()
                .zip(chunk)
            {
                out[i] = row.clone();
            }
        }
        Ok(out)
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
        padded: bool,
    ) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        let positions = Tensor::arange(0u32, t as u32, device)?
            .unsqueeze(0)?
            .broadcast_as((b, t))?;
        let mut xs = self.word_embeddings.forward(input_ids)?;
        xs = (xs + self.position_embeddings.forward(&positions)?)?;
        xs = (xs + self.token_type_embeddings.forward(token_type_ids)?)?;
        let xs = self.embeddings_ln.forward(&xs)?;

        // Additive padding mask `[b, 1, t, t]`, broadcast over heads. The fused
        // kernel reads it through strides, so nothing is materialized per head.
        // A batch with no padding needs no mask at all, which keeps the kernel on
        // its unmasked path (and the single-query case on it too).
        let additive = if !padded {
            None
        } else {
            let mask_f = attention_mask.to_dtype(DType::F32)?;
            let additive = (mask_f.ones_like()? - &mask_f)?.affine(f32::MIN as f64, 0.0)?;
            Some(
                additive
                    .reshape((b, 1, 1, t))?
                    .to_dtype(xs.dtype())?
                    .broadcast_as((b, self.cfg.num_attention_heads, t, t))?,
            )
        };
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                additive.as_ref(),
                self.cfg.num_attention_heads,
                self.cfg.head_dim(),
                1.0 / (self.cfg.head_dim() as f32).sqrt(),
            )?;
        }
        Ok(xs)
    }
}
