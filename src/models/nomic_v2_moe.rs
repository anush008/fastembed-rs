//! nomic-embed-text-v2-moe: First general-purpose MoE embedding model.
//!
//! 475M total / 305M active params, 8 experts with top-2 routing.
//! NomicBert architecture with MoE layers on alternating transformer blocks.
//!
//! This module provides [`NomicV2MoeTextEmbedding`] which handles tokenization,
//! forward pass, mean pooling, and L2 normalization, entirely via candle-nn.
//! No ONNX runtime required.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Config (deserialized from config.json)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug, Clone)]
pub struct NomicConfig {
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(alias = "n_inner")]
    pub intermediate_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_moe_top_k")]
    pub moe_top_k: usize,
    #[serde(default = "default_moe_every_n_layers")]
    pub moe_every_n_layers: usize,
    #[serde(default = "default_rotary_emb_base")]
    pub rotary_emb_base: f64,
    #[serde(default = "default_rotary_emb_fraction")]
    pub rotary_emb_fraction: f64,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_true")]
    pub qkv_proj_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_fc1_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_fc2_bias: bool,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
}

fn default_type_vocab_size() -> usize {
    1
}
fn default_num_experts() -> usize {
    8
}
fn default_moe_top_k() -> usize {
    2
}
fn default_moe_every_n_layers() -> usize {
    2
}
fn default_rotary_emb_base() -> f64 {
    10000.0
}
fn default_rotary_emb_fraction() -> f64 {
    1.0
}
fn default_layer_norm_epsilon() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_pad_token_id() -> usize {
    1
}

impl NomicConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rotary_emb_fraction) as usize
    }
}

// ---------------------------------------------------------------------------
// Rotary Positional Embedding
// ---------------------------------------------------------------------------

struct NomicRotaryEmbedding {
    inv_freq: Tensor, // [rotary_dim/2]
}

impl NomicRotaryEmbedding {
    fn new(cfg: &NomicConfig, device: &Device) -> Result<Self> {
        let rotary_dim = cfg.rotary_dim();
        let base = cfg.rotary_emb_base;
        assert!(
            rotary_dim.is_multiple_of(2),
            "rotary_dim must be even, got {rotary_dim}"
        );

        let t = Tensor::arange_step(0u32, rotary_dim as u32, 2u32, device)?.to_dtype(DType::F32)?;
        let dim_f = Tensor::new(&[rotary_dim as f32], device)?;
        let exponent = t.broadcast_div(&dim_f)?;
        let ln_base = (base as f32).ln();
        let ln_base_t = Tensor::new(&[ln_base], device)?;
        let inv_freq = exponent.broadcast_mul(&ln_base_t.neg()?)?.exp()?;

        Ok(Self { inv_freq })
    }

    /// Returns (cos, sin) each of shape [1, seq_len, rotary_dim].
    fn forward(&self, seq_len: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?; // [T, 1]
        let inv_freq = self.inv_freq.to_device(device)?.unsqueeze(0)?; // [1, D/2]
        let freqs = positions.matmul(&inv_freq)?; // [T, D/2]
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // [T, D]
        let cos = emb.cos()?.unsqueeze(0)?.to_dtype(dtype)?; // [1, T, D]
        let sin = emb.sin()?.unsqueeze(0)?.to_dtype(dtype)?; // [1, T, D]
        Ok((cos, sin))
    }
}

/// Apply non-interleaved rotary embedding to q/k.
/// `rotary_dim` is the number of dimensions to rotate (may be <= head_dim).
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    if rotary_dim >= last_dim {
        apply_rotary_full(x, cos, sin)
    } else {
        let x_rot = x.narrow(D::Minus1, 0, rotary_dim)?;
        let x_pass = x.narrow(D::Minus1, rotary_dim, last_dim - rotary_dim)?;
        let x_rot = apply_rotary_full(&x_rot, cos, sin)?;
        Tensor::cat(&[&x_rot, &x_pass], x.rank() - 1)
    }
}

fn apply_rotary_full(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?; // [1, 1, T, D]
    let sin = sin.unsqueeze(1)?;
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], x.rank() - 1)?;
    x.broadcast_mul(&cos)?
        .broadcast_add(&rotated.broadcast_mul(&sin)?)
}

// ---------------------------------------------------------------------------
// NomicBert Embeddings (word + token_type, NO norm, emb_ln is separate)
// ---------------------------------------------------------------------------

struct NomicEmbeddings {
    word_embeddings: candle_nn::Embedding,
    token_type_embeddings: Option<candle_nn::Embedding>,
}

impl NomicEmbeddings {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let word_embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;
        let token_type_embeddings = if cfg.type_vocab_size > 0 {
            Some(candle_nn::embedding(
                cfg.type_vocab_size,
                cfg.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let mut embeddings = self.word_embeddings.forward(input_ids)?;
        if let (Some(tte), Some(tti)) = (&self.token_type_embeddings, token_type_ids) {
            embeddings = (embeddings + tte.forward(tti)?)?;
        }
        Ok(embeddings)
    }
}

// ---------------------------------------------------------------------------
// NomicBert Attention (combined Wqkv, RoPE, bidirectional)
// ---------------------------------------------------------------------------

struct NomicAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f64,
}

impl NomicAttention {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let wqkv = if cfg.qkv_proj_bias {
            linear(hidden, 3 * hidden, vb.pp("Wqkv"))?
        } else {
            candle_nn::linear_no_bias(hidden, 3 * hidden, vb.pp("Wqkv"))?
        };
        let out_proj = if cfg.qkv_proj_bias {
            linear(hidden, hidden, vb.pp("out_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden, hidden, vb.pp("out_proj"))?
        };

        let head_dim = cfg.head_dim();
        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            wqkv,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            rotary_dim: cfg.rotary_dim(),
            scale,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = hidden_states.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let qkv = hidden_states.apply(&self.wqkv)?;
        let qkv = qkv.reshape((b, t, 3, h, d))?;

        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;

        let q = apply_rotary_emb(&q, cos, sin, self.rotary_dim)?;
        let k = apply_rotary_emb(&k, cos, sin, self.rotary_dim)?;

        let mut attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        if let Some(mask) = attention_mask {
            attn = attn.broadcast_add(&mask.to_dtype(attn.dtype())?)?;
        }

        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * d))?;
        out.apply(&self.out_proj)
    }
}

// ---------------------------------------------------------------------------
// Standard MLP (for non-MoE layers)
// ---------------------------------------------------------------------------

struct NomicBertMLP {
    fc1: Linear,
    fc2: Linear,
}

impl NomicBertMLP {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = if cfg.mlp_fc1_bias {
            linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?
        };
        let fc2 = if cfg.mlp_fc2_bias {
            linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?
        } else {
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?
        };
        Ok(Self { fc1, fc2 })
    }
}

impl Module for NomicBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

// ---------------------------------------------------------------------------
// MoE: Router + Stacked Expert MLPs (megablocks convention)
//
// Weight keys:
//   mlp.router.layer.weight   [num_experts, hidden_size]
//   mlp.experts.mlp.w1        [num_experts * intermediate, hidden_size]
//   mlp.experts.mlp.w2        [num_experts * intermediate, hidden_size]
//   mlp.experts.bias           [hidden_size]
//
// Note: Expert dispatch uses CPU-side token-by-expert accumulation. This is
// efficient for CPU inference but would benefit from scatter/gather kernels
// for GPU backends.
// ---------------------------------------------------------------------------

struct NomicRouter {
    gate: Linear,
    top_k: usize,
}

impl NomicRouter {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("layer"))?;
        Ok(Self {
            gate,
            top_k: cfg.moe_top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = xs.apply(&self.gate)?;
        let weights = candle_nn::ops::softmax(&logits, D::Minus1)?;

        let seq_len = logits.dim(0)?;
        let weights_vec = weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        let mut top_indices_vec = Vec::with_capacity(seq_len * self.top_k);
        let mut top_weights_vec = Vec::with_capacity(seq_len * self.top_k);

        for token_weights in &weights_vec {
            let mut indexed: Vec<(usize, f32)> =
                token_weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

            for &(idx, w) in indexed.iter().take(self.top_k) {
                top_indices_vec.push(idx as u32);
                top_weights_vec.push(w);
            }
        }

        let device = xs.device();
        let top_indices = Tensor::from_vec(top_indices_vec, (seq_len, self.top_k), device)?;
        let top_weights = Tensor::from_vec(top_weights_vec, (seq_len, self.top_k), device)?
            .to_dtype(xs.dtype())?;

        Ok((top_weights, top_indices))
    }
}

struct NomicMoELayer {
    router: NomicRouter,
    w1: Tensor,   // [num_experts, intermediate_size, hidden_size]
    w2: Tensor,   // [num_experts, intermediate_size, hidden_size] (megablocks: no transpose)
    bias: Tensor, // [hidden_size] shared output bias
    num_experts: usize,
}

impl NomicMoELayer {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let router = NomicRouter::new(cfg, vb.pp("router"))?;

        let experts_vb = vb.pp("experts");
        let mlp_vb = experts_vb.pp("mlp");

        let w1_flat = mlp_vb.get(
            (cfg.num_experts * cfg.intermediate_size, cfg.hidden_size),
            "w1",
        )?;
        let w2_flat = mlp_vb.get(
            (cfg.num_experts * cfg.intermediate_size, cfg.hidden_size),
            "w2",
        )?;
        let bias = experts_vb.get((cfg.hidden_size,), "bias")?;

        let w1 = w1_flat.reshape((cfg.num_experts, cfg.intermediate_size, cfg.hidden_size))?;
        let w2 = w2_flat.reshape((cfg.num_experts, cfg.intermediate_size, cfg.hidden_size))?;

        Ok(Self {
            router,
            w1,
            w2,
            bias,
            num_experts: cfg.num_experts,
        })
    }

    /// xs: [total_tokens, hidden_size]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden) = xs.dims2()?;
        let (top_weights, top_indices) = self.router.forward(xs)?;

        let mut output_vec = vec![0.0f32; seq_len * hidden];

        let top_indices_vec = top_indices.to_vec2::<u32>()?;
        let top_weights_vec = top_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        for expert_idx in 0..self.num_experts {
            let mut assigned_tokens: Vec<usize> = Vec::new();
            let mut assigned_weights: Vec<f32> = Vec::new();

            for (token_idx, (indices, weights)) in top_indices_vec
                .iter()
                .zip(top_weights_vec.iter())
                .enumerate()
            {
                for (&idx, &w) in indices.iter().zip(weights.iter()) {
                    if idx as usize == expert_idx {
                        assigned_tokens.push(token_idx);
                        assigned_weights.push(w);
                    }
                }
            }

            if assigned_tokens.is_empty() {
                continue;
            }

            let indices_t = Tensor::from_vec(
                assigned_tokens
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<_>>(),
                (assigned_tokens.len(),),
                xs.device(),
            )?;
            let selected = xs.index_select(&indices_t, 0)?;

            // Expert forward: up = gelu(selected @ w1[i].T), down = up @ w2[i]
            let w1_i = self.w1.i(expert_idx)?;
            let w2_i = self.w2.i(expert_idx)?;

            let up = selected.matmul(&w1_i.t()?)?.gelu_erf()?;
            let down = up.matmul(&w2_i)?; // megablocks: no transpose on w2

            let weights_t =
                Tensor::from_vec(assigned_weights, (assigned_tokens.len(), 1), xs.device())?
                    .to_dtype(xs.dtype())?;
            let weighted = down.broadcast_mul(&weights_t)?;

            let weighted_vec = weighted.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            for (local_idx, &global_idx) in assigned_tokens.iter().enumerate() {
                for (j, val) in weighted_vec[local_idx].iter().enumerate() {
                    output_vec[global_idx * hidden + j] += val;
                }
            }
        }

        let output =
            Tensor::from_vec(output_vec, (seq_len, hidden), xs.device())?.to_dtype(xs.dtype())?;
        output.broadcast_add(&self.bias.to_dtype(xs.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// MLP enum: Standard or MoE
// ---------------------------------------------------------------------------

enum NomicMLP {
    Standard(NomicBertMLP),
    MoE(NomicMoELayer),
}

impl NomicMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            NomicMLP::Standard(mlp) => mlp.forward(xs),
            NomicMLP::MoE(moe) => {
                let shape = xs.dims().to_vec();
                let hidden = *shape.last().unwrap();
                let total_tokens: usize = shape[..shape.len() - 1].iter().product();
                let flat = xs.reshape((total_tokens, hidden))?;
                let out = moe.forward(&flat)?;
                out.reshape(shape.as_slice())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transformer Block, POST-NORM (prenorm=false in config)
//
// Flow: attn(x) + x → norm1 → mlp + prev → norm2
// Dropout (resid_pdrop=0.0) is a no-op at inference and omitted.
// ---------------------------------------------------------------------------

struct NomicBertBlock {
    attn: NomicAttention,
    mlp: NomicMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl NomicBertBlock {
    fn new(cfg: &NomicConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let attn = NomicAttention::new(cfg, vb.pp("attn"))?;
        let is_moe = cfg.num_experts > 1 && layer_idx % cfg.moe_every_n_layers == 1;
        let mlp = if is_moe {
            NomicMLP::MoE(NomicMoELayer::new(cfg, vb.pp("mlp"))?)
        } else {
            NomicMLP::Standard(NomicBertMLP::new(cfg, vb.pp("mlp"))?)
        };
        let norm1 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm2"))?;

        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Post-norm: attn → residual add → norm1
        let attn_out = self.attn.forward(hidden_states, cos, sin, attention_mask)?;
        let hidden_states = (attn_out + hidden_states)?.apply(&self.norm1)?;

        // Post-norm: mlp → residual add → norm2
        let mlp_out = self.mlp.forward(&hidden_states)?;
        (mlp_out + hidden_states)?.apply(&self.norm2)
    }
}

// ---------------------------------------------------------------------------
// NomicBert Encoder, no final norm, blocks handle normalization internally
// ---------------------------------------------------------------------------

struct NomicBertEncoder {
    layers: Vec<NomicBertBlock>,
}

impl NomicBertEncoder {
    fn new(cfg: &NomicConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(NomicBertBlock::new(cfg, i, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hs = hidden_states.clone();
        for layer in &self.layers {
            hs = layer.forward(&hs, cos, sin, attention_mask)?;
        }
        Ok(hs)
    }
}

// ---------------------------------------------------------------------------
// NomicBertModel
// ---------------------------------------------------------------------------

pub struct NomicBertModel {
    embeddings: NomicEmbeddings,
    emb_ln: LayerNorm,
    encoder: NomicBertEncoder,
    rotary: NomicRotaryEmbedding,
    device: Device,
}

impl NomicBertModel {
    pub fn new(cfg: NomicConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let embeddings = NomicEmbeddings::new(&cfg, vb.pp("embeddings"))?;
        let emb_ln = layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("emb_ln"))?;
        let encoder = NomicBertEncoder::new(&cfg, vb.pp("encoder"))?;
        let rotary = NomicRotaryEmbedding::new(&cfg, &device)?;
        Ok(Self {
            embeddings,
            emb_ln,
            encoder,
            rotary,
            device,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;

        let emb = self.embeddings.forward(input_ids, token_type_ids)?;
        let hidden_states = emb.apply(&self.emb_ln)?;

        // Rotary position embeddings
        let (cos, sin) = self
            .rotary
            .forward(t, &self.device, hidden_states.dtype())?;

        // Bidirectional attention mask [B, 1, 1, T]
        let mask_4d = build_bidirectional_mask(attention_mask)?;

        // Encoder (post-norm blocks, no final norm needed)
        self.encoder
            .forward(&hidden_states, &cos, &sin, Some(&mask_4d))
    }
}

fn build_bidirectional_mask(attention_mask: &Tensor) -> Result<Tensor> {
    let mask_f32 = attention_mask.to_dtype(DType::F32)?;
    let ones = Tensor::ones_like(&mask_f32)?;
    let inverted = ones.sub(&mask_f32)?;
    let mask_value = Tensor::new(&[-10000.0f32], attention_mask.device())?;
    let additive = inverted.broadcast_mul(&mask_value)?;
    additive.unsqueeze(1)?.unsqueeze(2)
}

// ---------------------------------------------------------------------------
// Mean pooling + L2 normalization
// ---------------------------------------------------------------------------

fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask_f32 = attention_mask.to_dtype(hidden.dtype())?.unsqueeze(2)?;
    let masked = hidden.broadcast_mul(&mask_f32)?;
    let summed = masked.sum(1)?;
    let count = mask_f32.sum(1)?.clamp(1e-9, f64::MAX)?;
    summed.broadcast_div(&count)
}

fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let sum_sq = xs.sqr()?.sum_keepdim(D::Minus1)?;
    let eps = Tensor::new(&[1e-12f32], xs.device())?.broadcast_as(sum_sq.shape())?;
    let norm = sum_sq.to_dtype(DType::F32)?.add(&eps)?.sqrt()?;
    xs.to_dtype(DType::F32)?.broadcast_div(&norm)
}

// ---------------------------------------------------------------------------
// Public API: NomicV2MoeTextEmbedding
// ---------------------------------------------------------------------------

pub struct NomicV2MoeTextEmbedding {
    model: NomicBertModel,
    tokenizer: tokenizers::Tokenizer,
    cfg: NomicConfig,
}

impl NomicV2MoeTextEmbedding {
    pub fn new(model: NomicBertModel, tokenizer: tokenizers::Tokenizer, cfg: NomicConfig) -> Self {
        Self {
            model,
            tokenizer,
            cfg,
        }
    }

    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let repo =
            crate::common::pull_from_hf(repo_id.to_string(), crate::get_cache_dir().into(), true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let cfg_path: PathBuf = repo
            .get("config.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cfg: NomicConfig = serde_json::from_slice(
            &std::fs::read(&cfg_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
        )
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let weight_files: Vec<PathBuf> = crate::common::safetensors_weight_files(&repo)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = NomicBertModel::new(cfg.clone(), vb)?;

        let tok_path: PathBuf = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_id: cfg.pad_token_id as u32,
            pad_token: "<pad>".to_string(),
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        Ok(Self {
            model,
            tokenizer,
            cfg,
        })
    }

    pub fn config(&self) -> &NomicConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.model.device
    }

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

        let device = &self.model.device;
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask = Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;

        let token_type_ids = Tensor::zeros((batch_size, seq_len), DType::U32, device)?;

        let hidden = self
            .model
            .forward(&input_ids, &attention_mask, Some(&token_type_ids))?;

        let pooled = mean_pool(&hidden, &attention_mask)?;
        let normalized = l2_normalize(&pooled)?;

        normalized.to_vec2::<f32>()
    }
}
