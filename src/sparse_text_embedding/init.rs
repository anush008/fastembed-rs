use ort::session::Session;
use std::collections::{HashMap, HashSet};
use tokenizers::Tokenizer;

use crate::{
    init::{HasMaxLength, InitOptionsWithLength},
    models::sparse::SparseModel,
    TokenizerFiles,
};

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for SparseModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the SparseTextEmbedding model
pub type SparseInitOptions = InitOptionsWithLength<SparseModel>;

/// Struct for "bring your own" sparse embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedSparseModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    /// Model family of the checkpoint, selects the post-processing of the ONNX outputs.
    pub model: SparseModel,
    /// `idf.json` bytes for inference-free models, enables [`SparseTextEmbedding::query_embed`].
    pub idf_file: Option<Vec<u8>>,
}

impl UserDefinedSparseModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles, model: SparseModel) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
            model,
            idf_file: None,
        }
    }

    pub fn with_idf_file(mut self, idf_file: Vec<u8>) -> Self {
        self.idf_file = Some(idf_file);
        self
    }
}

/// Rust representation of the SparseTextEmbedding model
pub struct SparseTextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) model: SparseModel,
    /// Ids of the tokenizer's special tokens, excluded from every embedding produced by
    /// the inference-free models.
    pub(crate) special_token_ids: HashSet<usize>,
    /// Token id to IDF weight, read from the `idf.json` shipped with inference-free models.
    /// [`None`] for models which do not have a separate query representation.
    pub(crate) token_id_to_idf: Option<HashMap<usize, f32>>,
}
