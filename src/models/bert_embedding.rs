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
use tokenizers::Tokenizer;

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
    // Kept as JSON: a value of another type than expected counts as no value.
    #[serde(default)]
    pub model_type: Option<serde_json::Value>,
    #[serde(default)]
    pub pad_token_id: Option<serde_json::Value>,
}

impl BertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The padding id that RoBERTa-style position ids count from: token `i` of a text has
    /// position `padding id + 1 + i`. `None` for BERT, which counts from 0.
    fn position_padding_idx(&self) -> Option<u32> {
        let model_type = self.model_type.as_ref().and_then(|value| value.as_str());
        matches!(model_type, Some("roberta" | "xlm-roberta" | "camembert")).then(|| {
            let padding_idx = self.pad_token_id.as_ref().and_then(|value| value.as_u64());
            padding_idx
                .and_then(|idx| u32::try_from(idx).ok())
                .unwrap_or(1)
        })
    }

    /// The longest text that the position table has rows for.
    fn max_tokens(&self) -> usize {
        match self.position_padding_idx() {
            Some(padding_idx) => self
                .max_position_embeddings
                .saturating_sub(padding_idx as usize + 1),
            None => self.max_position_embeddings,
        }
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
        // The projection bias of `block`, then the weight and the bias of its LayerNorm.
        // Checkpoints converted from TensorFlow name the last two `gamma` and `beta`.
        let stack = |block: &str| -> Result<Tensor> {
            let norm = vb.pp(block).pp("LayerNorm");
            let name = |names: [&'static str; 2]| {
                let [name, alias] = names;
                if !norm.contains_tensor(name) && norm.contains_tensor(alias) {
                    alias
                } else {
                    name
                }
            };
            let rows = [
                vb.get((h,), &format!("{block}.dense.bias"))?,
                norm.get((h,), name(["weight", "gamma"]))?,
                norm.get((h,), name(["bias", "beta"]))?,
            ];
            Tensor::cat(&rows, 0)?.reshape((3, h))
        };
        let attn_wb = stack("attention.output")?;
        let out_wb = stack("output")?;
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
    /// Hugging Face repo. `dtype` applies to the weights. `max_length` is capped at the
    /// longest text that the position table holds.
    ///
    /// A RoBERTa, XLM-RoBERTa, or CamemBERT checkpoint, by the `model_type` of its config,
    /// gets the position ids of that architecture.
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

        let weight_files = crate::common::safetensors_weight_files(&repo)
            .map_err(|e| candle_core::Error::Msg(crate::common::error_with_cause(e)))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let this = Self::load(vb, &cfg, dtype)?;

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(map_err)?;
        this.with_tokenizer(tokenizer, max_length)
    }

    fn load(vb: VarBuilder, cfg: &BertConfig, dtype: DType) -> Result<Self> {
        // HF BERT checkpoints prefix every tensor with `bert.`; accept a bare
        // prefix too (some converted sentence-transformer repos drop it).
        match Self::load_weights(vb.pp("bert"), cfg, dtype) {
            Ok(model) => Ok(model),
            Err(prefixed) => Self::load_weights(vb.clone(), cfg, dtype).map_err(|bare| {
                // The error of the layout that the file has.
                if vb.contains_tensor("bert.embeddings.word_embeddings.weight") {
                    prefixed
                } else {
                    bare
                }
            }),
        }
    }

    fn with_tokenizer(mut self, mut tokenizer: Tokenizer, max_length: usize) -> Result<Self> {
        let pad_id = match self.cfg.position_padding_idx() {
            Some(padding_idx) => padding_idx,
            None => tokenizer.token_to_id("[PAD]").unwrap_or(0),
        };
        tokenizer.with_padding(None);
        let max_length = set_truncation(&mut tokenizer, max_length, &self.cfg)?;
        self.tokenizer = Some(tokenizer);
        self.max_length = max_length;
        self.pad_id = pad_id;
        Ok(self)
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
            let positions = position_ids(&ids, seq, self.cfg.position_padding_idx());
            let input_ids = Tensor::from_vec(ids, (batch, seq), device)?;
            let positions = Tensor::from_vec(positions, (batch, seq), device)?;
            let token_type_ids = Tensor::zeros((batch, seq), DType::U32, device)?;
            let attention_mask = Tensor::from_vec(mask, (batch, seq), device)?;
            let padded = chunk.iter().any(|&i| encodings[i].get_ids().len() < seq);
            let hidden = self.forward(
                &input_ids,
                &positions,
                &token_type_ids,
                &attention_mask,
                padded,
            )?;
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
        positions: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
        padded: bool,
    ) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let mut xs = self.word_embeddings.forward(input_ids)?;
        xs = (xs + self.position_embeddings.forward(positions)?)?;
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

/// The position ids of `ids`, which holds rows of `seq` tokens. BERT counts the tokens of a
/// row from 0. RoBERTa counts the tokens that are not padding from `padding_idx + 1`, and
/// padding has the position `padding_idx`.
fn position_ids(ids: &[u32], seq: usize, padding_idx: Option<u32>) -> Vec<u32> {
    let mut positions = Vec::with_capacity(ids.len());
    for row in ids.chunks(seq) {
        match padding_idx {
            None => positions.extend(0..row.len() as u32),
            Some(padding_idx) => {
                let mut position = padding_idx;
                positions.extend(row.iter().map(|&id| {
                    if id == padding_idx {
                        padding_idx
                    } else {
                        position += 1;
                        position
                    }
                }));
            }
        }
    }
    positions
}

/// Sets `tokenizer` to truncate at `max_length`, capped at the position count of the model,
/// and returns the length in use.
fn set_truncation(tokenizer: &mut Tokenizer, max_length: usize, cfg: &BertConfig) -> Result<usize> {
    let max_length = max_length.min(cfg.max_tokens());
    let truncation = crate::common::truncation_to(tokenizer, max_length).map_err(map_err)?;
    tokenizer
        .with_truncation(Some(truncation))
        .map_err(map_err)?;
    Ok(max_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const TEXTS: [&str; 3] = ["w1 w2 w3 w4 w5", "w6", "w7 w8 w9"];

    fn config(model_type: Option<&str>) -> BertConfig {
        BertConfig {
            vocab_size: 14,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 18,
            type_vocab_size: 1,
            layer_norm_eps: 1e-12,
            model_type: model_type.map(Into::into),
            pad_token_id: model_type.map(|_| 1.into()),
        }
    }

    /// Random weights for `cfg`, under the names that `transformers` saves them with.
    fn tensors(cfg: &BertConfig) -> HashMap<String, Tensor> {
        let (h, inner) = (cfg.hidden_size, cfg.intermediate_size);
        let mut shapes = vec![
            (
                "embeddings.word_embeddings.weight".to_string(),
                vec![cfg.vocab_size, h],
            ),
            (
                "embeddings.position_embeddings.weight".to_string(),
                vec![cfg.max_position_embeddings, h],
            ),
            (
                "embeddings.token_type_embeddings.weight".to_string(),
                vec![cfg.type_vocab_size, h],
            ),
            ("embeddings.LayerNorm.weight".to_string(), vec![h]),
            ("embeddings.LayerNorm.bias".to_string(), vec![h]),
        ];
        for layer in 0..cfg.num_hidden_layers {
            for (name, shape) in [
                ("attention.self.query.weight", vec![h, h]),
                ("attention.self.query.bias", vec![h]),
                ("attention.self.key.weight", vec![h, h]),
                ("attention.self.key.bias", vec![h]),
                ("attention.self.value.weight", vec![h, h]),
                ("attention.self.value.bias", vec![h]),
                ("attention.output.dense.weight", vec![h, h]),
                ("attention.output.dense.bias", vec![h]),
                ("attention.output.LayerNorm.weight", vec![h]),
                ("attention.output.LayerNorm.bias", vec![h]),
                ("intermediate.dense.weight", vec![inner, h]),
                ("intermediate.dense.bias", vec![inner]),
                ("output.dense.weight", vec![h, inner]),
                ("output.dense.bias", vec![h]),
                ("output.LayerNorm.weight", vec![h]),
                ("output.LayerNorm.bias", vec![h]),
            ] {
                shapes.push((format!("encoder.layer.{layer}.{name}"), shape));
            }
        }
        shapes
            .into_iter()
            .map(|(name, shape)| {
                let tensor = Tensor::randn(0f32, 1.0, shape, &Device::Cpu).unwrap();
                (name, tensor)
            })
            .collect()
    }

    /// A word-level tokenizer over `w0`..`w9` with the special token ids of RoBERTa.
    fn tokenizer() -> Tokenizer {
        let words: Vec<String> = (0..10).map(|i| format!("\"w{i}\": {}", i + 4)).collect();
        format!(
            r#"{{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": {{"type": "WhitespaceSplit"}},
                "post_processor": {{"type": "BertProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]}},
                "decoder": null,
                "model": {{"type": "WordLevel", "unk_token": "<unk>",
                    "vocab": {{"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, {}}}}}}}"#,
            words.join(", ")
        )
        .parse()
        .unwrap()
    }

    fn embedder(cfg: &BertConfig, tensors: HashMap<String, Tensor>) -> Result<BertTextEmbedding> {
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        BertTextEmbedding::load(vb, cfg, DType::F32)?.with_tokenizer(tokenizer(), 64)
    }

    fn assert_same(got: &[Vec<f32>], want: &[Vec<f32>]) {
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want) {
            assert_eq!(got.len(), want.len());
            for (got, want) in got.iter().zip(want) {
                assert!((got - want).abs() < 1e-6, "{got} is not {want}");
            }
        }
    }

    #[test]
    fn layer_norm_tensors_named_gamma_and_beta_load() {
        let cfg = config(None);
        let tensors = tensors(&cfg);
        let want = embedder(&cfg, tensors.clone())
            .unwrap()
            .embed(&TEXTS)
            .unwrap();
        assert!(want.iter().flatten().all(|x| x.is_finite()));

        // The names of `google-bert/bert-base-uncased`.
        let renamed = tensors
            .into_iter()
            .map(|(name, tensor)| {
                let name = name
                    .replace("LayerNorm.weight", "LayerNorm.gamma")
                    .replace("LayerNorm.bias", "LayerNorm.beta");
                (format!("bert.{name}"), tensor)
            })
            .collect();
        let got = embedder(&cfg, renamed).unwrap().embed(&TEXTS).unwrap();
        assert_same(&got, &want);
    }

    #[test]
    fn a_missing_tensor_is_reported_under_the_prefix_of_the_file() {
        let cfg = config(None);
        let mut bare = tensors(&cfg);
        let mut prefixed: HashMap<String, Tensor> = bare
            .iter()
            .map(|(name, tensor)| (format!("bert.{name}"), tensor.clone()))
            .collect();

        prefixed.remove("bert.encoder.layer.0.output.dense.bias");
        let err = embedder(&cfg, prefixed).err().unwrap().to_string();
        assert!(
            err.contains("cannot find tensor bert.encoder.layer.0.output.dense.bias"),
            "{err}"
        );

        bare.remove("encoder.layer.0.output.dense.bias");
        let err = embedder(&cfg, bare).err().unwrap().to_string();
        assert!(
            err.contains("cannot find tensor encoder.layer.0.output.dense.bias"),
            "{err}"
        );
    }

    #[test]
    fn position_ids_count_from_zero_for_bert_and_from_the_padding_id_for_roberta() {
        let ids = [0, 5, 6, 2, 0, 5, 2, 1];
        assert_eq!(position_ids(&ids, 4, None), [0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(position_ids(&ids, 4, Some(1)), [2, 3, 4, 5, 2, 3, 4, 1]);
        // A padding token inside a text takes no position, as in `transformers`.
        assert_eq!(position_ids(&[0, 1, 5, 2], 4, Some(1)), [2, 1, 3, 4]);
    }

    #[test]
    fn a_roberta_model_reads_the_positions_after_the_padding_id() {
        let roberta = config(Some("roberta"));
        let tensors = tensors(&roberta);

        // The same weights as a BERT model: its position table starts at the first token.
        let bert = BertConfig {
            max_position_embeddings: roberta.max_position_embeddings - 2,
            ..config(None)
        };
        let mut shifted = tensors.clone();
        let table = "embeddings.position_embeddings.weight";
        let rows = tensors[table]
            .narrow(0, 2, bert.max_position_embeddings)
            .unwrap();
        shifted.insert(table.to_string(), rows);

        let bert = embedder(&bert, shifted).unwrap();
        let roberta = embedder(&roberta, tensors).unwrap();
        assert_same(
            &roberta.embed(&TEXTS).unwrap(),
            &bert.embed(&TEXTS).unwrap(),
        );

        // The position table has rows for 16 tokens after the padding id.
        assert_eq!(roberta.max_length(), 16);
        let long = "w1 ".repeat(40);
        assert_same(
            &roberta.embed(&[&long]).unwrap(),
            &bert.embed(&[&long]).unwrap(),
        );
    }

    #[test]
    fn the_config_names_the_models_with_roberta_positions() {
        let config = |extra: &str| -> BertConfig {
            serde_json::from_str(&format!(
                r#"{{"vocab_size": 4, "hidden_size": 4, "num_hidden_layers": 1,
                    "num_attention_heads": 1, "intermediate_size": 4,
                    "max_position_embeddings": 514, "type_vocab_size": 1,
                    "layer_norm_eps": 1e-5{extra}}}"#
            ))
            .unwrap()
        };
        for model_type in ["roberta", "xlm-roberta", "camembert"] {
            let cfg = config(&format!(
                r#", "model_type": "{model_type}", "pad_token_id": 1"#
            ));
            assert_eq!(cfg.position_padding_idx(), Some(1));
            assert_eq!(cfg.max_tokens(), 512);
        }
        // A padding id that is no index means the default of these models, as no id does.
        for pad_token_id in ["null", "-1", "1.0", "\"1\""] {
            let cfg = config(&format!(
                r#", "model_type": "roberta", "pad_token_id": {pad_token_id}"#
            ));
            assert_eq!(cfg.position_padding_idx(), Some(1));
        }

        for extra in [
            "",
            r#", "model_type": "bert", "pad_token_id": 0"#,
            r#", "model_type": "bert", "pad_token_id": -1"#,
            r#", "model_type": 7, "pad_token_id": 1.0"#,
        ] {
            let cfg = config(extra);
            assert_eq!(cfg.position_padding_idx(), None);
            assert_eq!(cfg.max_tokens(), 514);
        }
    }

    #[test]
    fn truncation_is_capped_at_the_position_count_and_keeps_room_for_special_tokens() {
        let cfg = BertConfig {
            vocab_size: 4,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            intermediate_size: 4,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            model_type: None,
            pad_token_id: None,
        };
        let mut tokenizer: Tokenizer = r#"{
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 2], "cls": ["[CLS]", 1]},
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}
            }
        }"#
        .parse()
        .unwrap();

        assert_eq!(set_truncation(&mut tokenizer, 4096, &cfg).unwrap(), 8);
        let encoding = tokenizer.encode("hello ".repeat(20), true).unwrap();
        assert_eq!(encoding.get_ids(), [1, 3, 3, 3, 3, 3, 3, 2]);

        assert_eq!(set_truncation(&mut tokenizer, 5, &cfg).unwrap(), 5);
        assert!(set_truncation(&mut tokenizer, 1, &cfg).is_err());
    }
}
