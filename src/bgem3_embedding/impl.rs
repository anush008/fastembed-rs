#[cfg(feature = "hf-hub")]
use crate::common::{init_session_builder, load_tokenizer_hf_hub};
use crate::{
    common::{encode_batch, load_tokenizer, Error, Result},
    models::bgem3::{models_list, Bgem3Model},
    text_embedding::InitOptionsUserDefined,
    ModelInfo, SparseEmbedding, TokenizerFiles,
};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ort::session::Session;
use std::collections::HashMap;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg(feature = "hf-hub")]
use super::Bgem3InitOptions;
use super::{Bgem3Embedding, Bgem3EmbeddingOutput, UserDefinedBgem3Model, DEFAULT_BATCH_SIZE};

impl Bgem3Embedding {
    /// Try to generate a new Bgem3Embedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: Bgem3InitOptions) -> Result<Self> {
        let Bgem3InitOptions {
            max_length,
            model_name,
            cache_dir,
            show_download_progress,
            execution_providers,
            intra_threads,
            session_config,
        } = options;

        let model_repo = Bgem3Embedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = Bgem3Embedding::get_model_info(&model_name);
        let model_file_name = &model_info.model_file;
        let model_file_reference =
            model_repo
                .get(model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        // Download additional files if needed
        for file in &model_info.additional_files {
            model_repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })?;
        }

        let session = init_session_builder(execution_providers, intra_threads, session_config)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session, model_name))
    }

    /// Create a Bgem3Embedding instance from model files provided by the user.
    pub fn try_new_from_user_defined(
        model: UserDefinedBgem3Model,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let (mut session_builder, max_length) = options.into_session_builder()?;
        let session = session_builder.commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session, Bgem3Model::default()))
    }

    /// Create a Bgem3Embedding instance from an ONNX model on disk.
    ///
    /// `model_path` is a `.onnx` file or a directory containing `model.onnx`.
    /// Supports split external data files (model.onnx + model.onnx_data).
    pub fn try_new_from_path(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_files: TokenizerFiles,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let model_file = if model_path.is_dir() {
            model_path.join("model.onnx")
        } else {
            model_path.to_path_buf()
        };

        let (mut session_builder, max_length) = options.into_session_builder()?;
        let session = session_builder.commit_from_file(model_file)?;

        let tokenizer = load_tokenizer(tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session, Bgem3Model::default()))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, session: Session, model: Bgem3Model) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            model,
        }
    }

    /// Return the Bgem3Embedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: Bgem3Model,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<Bgem3Model>> {
        models_list()
    }

    /// Get ModelInfo from Bgem3Model
    pub fn get_model_info(model: &Bgem3Model) -> ModelInfo<Bgem3Model> {
        Bgem3Embedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    /// Method to generate sentence embeddings for a collection of texts.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Bgem3EmbeddingOutput> {
        let texts = texts.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let mut all_dense = Vec::with_capacity(texts.len());
        let mut all_sparse = Vec::with_capacity(texts.len());
        let mut all_colbert = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let inputs = batch.iter().map(|text| text.as_ref()).collect();
            let mut encoded = encode_batch(&self.tokenizer, inputs)?;
            let session_inputs = encoded.session_inputs(self.need_token_type_ids)?;
            let (current_batch_size, encoding_length) = encoded.input_ids.dim();
            let inputs_ids_array = &encoded.input_ids;
            let attention_mask_array = &encoded.attention_mask;

            let outputs = self
                .session
                .run(session_inputs)
                .map_err(|e| Error::OrtSession(e.to_string()))?;
            if outputs.len() < 3 {
                return Err(Error::Other(format!(
                    "BGE-M3 expects the model to return 3 outputs (dense, sparse, colbert), got {}",
                    outputs.len()
                )));
            }

            // gpahal/bge-m3-onnx-int8 returns outputs in this exact order:
            // outputs[0] -> dense_vecs: [batch_size, 1024]
            // outputs[1] -> sparse_vecs: [batch_size, seq_len, 1]
            // outputs[2] -> colbert_vecs: [batch_size, seq_len - 1, 1024]

            // Dense vecs
            let dense_output = &outputs[0];
            let (dense_shape, dense_data) = dense_output
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::TensorExtraction(e.to_string()))?;
            let dense_shape: Vec<usize> = dense_shape.iter().map(|&d| d as usize).collect();
            let dense_view = ndarray::ArrayViewD::from_shape(dense_shape.as_slice(), dense_data)
                .map_err(|e| Error::InvalidShape(e.to_string()))?;

            for row in dense_view.rows() {
                all_dense.push(row.to_vec());
            }

            // Sparse vecs
            let sparse_output = &outputs[1];
            let (sparse_shape, sparse_data) = sparse_output
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::TensorExtraction(e.to_string()))?;
            let sparse_shape: Vec<usize> = sparse_shape.iter().map(|&d| d as usize).collect();
            if sparse_shape.len() != 3 {
                return Err(Error::Other(format!(
                    "BGE-M3 sparse output must be rank-3 [batch, seq_len, 1], got shape {sparse_shape:?}"
                )));
            }
            let sparse_view = ndarray::ArrayViewD::from_shape(sparse_shape.as_slice(), sparse_data)
                .map_err(|e| Error::InvalidShape(e.to_string()))?;

            // Special tokens to skip: XLM-RoBERTa: CLS=0, PAD=1, EOS=2, UNK=3
            const SPECIAL_TOKENS: [i64; 4] = [0, 1, 2, 3];

            for batch_idx in 0..current_batch_size {
                let mut token_weights: HashMap<usize, f32> = HashMap::new();
                for seq_idx in 0..encoding_length {
                    if attention_mask_array[[batch_idx, seq_idx]] == 0 {
                        continue;
                    }

                    let token_id = inputs_ids_array[[batch_idx, seq_idx]];
                    if SPECIAL_TOKENS.contains(&token_id) {
                        continue;
                    }

                    let weight = sparse_view[[batch_idx, seq_idx, 0]];
                    if weight > 0.0 {
                        token_weights
                            .entry(token_id as usize)
                            .and_modify(|w| *w = w.max(weight))
                            .or_insert(weight);
                    }
                }

                let mut indices: Vec<_> = token_weights.keys().copied().collect();
                indices.sort_unstable();
                let values: Vec<_> = indices.iter().map(|i| token_weights[i]).collect();

                all_sparse.push(SparseEmbedding { values, indices });
            }

            // ColBERT vecs
            let colbert_output = &outputs[2];
            let (colbert_shape, colbert_data) = colbert_output
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::TensorExtraction(e.to_string()))?;
            let colbert_shape: Vec<usize> = colbert_shape.iter().map(|&d| d as usize).collect();
            if colbert_shape.len() != 3 || colbert_shape[1] >= encoding_length {
                return Err(Error::Other(format!(
                    "BGE-M3 colbert output must be rank-3 [batch, seq_len - 1, dim] with seq_len - 1 < {encoding_length}, got shape {colbert_shape:?}"
                )));
            }
            let colbert_view =
                ndarray::ArrayViewD::from_shape(colbert_shape.as_slice(), colbert_data)
                    .map_err(|e| Error::InvalidShape(e.to_string()))?;

            // Shape of colbert_view is [batch_size, seq_len - 1, 1024]
            let colbert_seq_len = colbert_shape[1]; // seq_len - 1

            for batch_idx in 0..current_batch_size {
                let mut doc_colbert = Vec::new();
                for seq_idx in 0..colbert_seq_len {
                    if attention_mask_array[[batch_idx, seq_idx + 1]] == 1 {
                        let token_vector = colbert_view.slice(ndarray::s![batch_idx, seq_idx, ..]);
                        doc_colbert.push(token_vector.to_vec());
                    }
                }
                all_colbert.push(doc_colbert);
            }
        }

        Ok(Bgem3EmbeddingOutput {
            dense: all_dense,
            sparse: all_sparse,
            colbert: all_colbert,
        })
    }
}
