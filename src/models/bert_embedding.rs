//! BERT-family text embeddings on the candle backend.
//!
//! `mixedbread-ai/mxbai-embed-large-v1` (and any BERT checkpoint with CLS pooling) can
//! therefore run on the candle `Device::new_metal` path, like the Qwen3 models — no ONNX
//! Runtime, no CoreML partition/fallback. Pooling is CLS followed by L2 normalization,
//! matching the model card; the query-side prefix, if a caller wants one, is applied by
//! the caller (this type never rewrites documents).

use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::{Tokenizer, TruncationParams};

fn map_err<E: std::fmt::Display>(err: E) -> candle_core::Error {
    candle_core::Error::Msg(err.to_string())
}

/// A BERT encoder used as a text embedder (CLS pooling + L2 normalization).
pub struct BertTextEmbedding {
    model: BertModel,
    tokenizer: Tokenizer,
    max_length: usize,
    batch_size: usize,
    pad_id: u32,
}

impl BertTextEmbedding {
    /// Load a BERT-family checkpoint (`config.json`, `tokenizer.json`, safetensors) from a
    /// Hugging Face repo. `dtype` applies to the weights: `DType::F16` keeps the Metal
    /// buffers small, `DType::F32` is the reference path.
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        let repo = crate::common::pull_from_hf(
            repo_id.to_string(),
            crate::get_cache_dir().into(),
            true,
        )
        .map_err(map_err)?;

        let cfg_path: PathBuf = repo.get("config.json").map_err(map_err)?;
        let cfg_bytes = std::fs::read(&cfg_path).map_err(map_err)?;
        let cfg: Config = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let weight_files = crate::common::safetensors_weight_files(&repo).map_err(map_err)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = BertModel::load(vb, &cfg)?;

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(map_err)?;
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        // Padding is applied per length-bucket in `embed`, not by the tokenizer: a
        // batch padded to its own longest row would waste a third of the compute when
        // the corpus mixes short and long chunks.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(map_err)?;

        Ok(Self {
            model,
            tokenizer,
            max_length,
            batch_size: 16,
            pad_id,
        })
    }

    /// Texts embedded per forward pass. Padding is per batch, so a larger batch trades
    /// throughput for padding waste on a length-skewed corpus.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Embed texts in order, returning normalized vectors of `hidden_size` floats.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        let device = &self.model.device;
        let inputs: Vec<&str> = texts.iter().map(|t| t.as_ref()).collect();
        let encodings = self.tokenizer.encode_batch(inputs, true).map_err(map_err)?;
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
                ids.extend(std::iter::repeat(self.pad_id).take(seq - n));
                mask.extend(std::iter::repeat(1u32).take(n));
                mask.extend(std::iter::repeat(0u32).take(seq - n));
            }
            let input_ids = Tensor::from_vec(ids, (batch, seq), device)?;
            let token_type_ids = Tensor::zeros((batch, seq), DType::U32, device)?;
            let attention_mask = Tensor::from_vec(mask, (batch, seq), device)?;
            let hidden = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
            let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
            for (row, &i) in cls.broadcast_div(&norm)?.to_vec2::<f32>()?.iter().zip(chunk) {
                out[i] = row.clone();
            }
        }
        Ok(out)
    }
}
