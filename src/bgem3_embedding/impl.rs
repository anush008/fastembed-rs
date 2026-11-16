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
use ndarray::{Array2, ArrayViewD};
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
            environment,
            execution_providers,
            intra_threads,
            session_config,
            auto_device,
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

        let session = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
            auto_device,
        )?
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
            let current_batch_size = encoded.input_ids.nrows();

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

            let dense = output_view(&outputs[0])?;
            all_dense.extend(dense_vectors(&dense, current_batch_size)?);

            let sparse = output_view(&outputs[1])?;
            all_sparse.extend(sparse_embeddings(
                &sparse,
                &encoded.input_ids,
                &encoded.attention_mask,
            )?);

            let colbert = output_view(&outputs[2])?;
            all_colbert.extend(colbert_vectors(&colbert, &encoded.attention_mask)?);
        }

        Ok(Bgem3EmbeddingOutput {
            dense: all_dense,
            sparse: all_sparse,
            colbert: all_colbert,
        })
    }
}

fn output_view<'a>(output: &'a ort::value::DynValue) -> Result<ArrayViewD<'a, f32>> {
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::TensorExtraction(e.to_string()))?;
    let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    ArrayViewD::from_shape(shape, data).map_err(|e| Error::InvalidShape(e.to_string()))
}

/// One dense vector per text.
fn dense_vectors(dense: &ArrayViewD<f32>, texts: usize) -> Result<Vec<Vec<f32>>> {
    let vectors: Vec<Vec<f32>> = dense.rows().into_iter().map(|row| row.to_vec()).collect();
    if vectors.len() != texts {
        return Err(Error::Other(format!(
            "BGE-M3 dense output must have one row per text, got shape {:?} for {texts} texts",
            dense.shape()
        )));
    }
    Ok(vectors)
}

/// Per text, the largest positive weight of each token id, special tokens left out.
fn sparse_embeddings(
    sparse: &ArrayViewD<f32>,
    input_ids: &Array2<i64>,
    attention_mask: &Array2<i64>,
) -> Result<Vec<SparseEmbedding>> {
    // Special tokens to skip: XLM-RoBERTa: CLS=0, PAD=1, EOS=2, UNK=3
    const SPECIAL_TOKENS: [i64; 4] = [0, 1, 2, 3];

    let shape = sparse.shape();
    if shape.len() != 3 {
        return Err(Error::Other(format!(
            "BGE-M3 sparse output must be rank-3 [batch, seq_len, 1], got shape {shape:?}"
        )));
    }
    let (batch_size, encoding_length) = input_ids.dim();

    (0..batch_size)
        .map(|batch_idx| {
            let mut token_weights: HashMap<usize, f32> = HashMap::new();
            for seq_idx in 0..encoding_length {
                if attention_mask[[batch_idx, seq_idx]] == 0 {
                    continue;
                }

                let token_id = input_ids[[batch_idx, seq_idx]];
                if SPECIAL_TOKENS.contains(&token_id) {
                    continue;
                }

                let weight = *sparse.get([batch_idx, seq_idx, 0]).ok_or_else(|| {
                    Error::Other(format!(
                        "BGE-M3 sparse output of shape {shape:?} has no weight for token {seq_idx} of text {batch_idx}"
                    ))
                })?;
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

            Ok(SparseEmbedding { values, indices })
        })
        .collect()
}

/// Per text, the ColBERT vectors of its unmasked tokens after the first.
fn colbert_vectors(
    colbert: &ArrayViewD<f32>,
    attention_mask: &Array2<i64>,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let shape = colbert.shape();
    let (batch_size, encoding_length) = attention_mask.dim();
    if shape.len() != 3 || shape[1] >= encoding_length {
        return Err(Error::Other(format!(
            "BGE-M3 colbert output must be rank-3 [batch, seq_len - 1, dim] with seq_len - 1 < {encoding_length}, got shape {shape:?}"
        )));
    }
    // Shape of colbert is [batch_size, seq_len - 1, 1024]
    let colbert_seq_len = shape[1];

    (0..batch_size)
        .map(|batch_idx| {
            let mut doc_colbert = Vec::new();
            for seq_idx in 0..colbert_seq_len {
                if attention_mask[[batch_idx, seq_idx + 1]] == 1 {
                    if batch_idx >= shape[0] {
                        return Err(Error::Other(format!(
                            "BGE-M3 colbert output of shape {shape:?} has no vectors for text {batch_idx}"
                        )));
                    }
                    let token_vector = colbert.slice(ndarray::s![batch_idx, seq_idx, ..]);
                    doc_colbert.push(token_vector.to_vec());
                }
            }
            Ok(doc_colbert)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array1, Array2, Array3};

    #[test]
    fn dense_vectors_takes_one_row_per_text() {
        let flat = Array2::from_shape_vec((2, 3), (0..6).map(|v| v as f32).collect()).unwrap();
        let rows = dense_vectors(&flat.view().into_dyn(), 2).unwrap();
        assert_eq!(rows, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        // Shapes 7.7.0 read one row per text from.
        let nested = Array3::<f32>::zeros((2, 1, 3)).into_dyn();
        assert_eq!(dense_vectors(&nested.view(), 2).unwrap().len(), 2);
        let single = Array1::<f32>::zeros(3).into_dyn();
        assert_eq!(dense_vectors(&single.view(), 1).unwrap().len(), 1);

        let per_token = Array3::<f32>::zeros((2, 4, 3)).into_dyn();
        assert!(dense_vectors(&per_token.view(), 2).is_err());
    }

    #[test]
    fn sparse_embeddings_keeps_the_largest_positive_weight_per_token() {
        // <s>=0 and </s>=2 are special, 1 is padding.
        let input_ids = arr2(&[[0i64, 9, 7, 9, 2], [0, 7, 2, 1, 1]]);
        let attention_mask = arr2(&[[1i64, 1, 1, 1, 1], [1, 1, 1, 0, 0]]);
        let sparse = Array3::from_shape_vec(
            (2, 5, 1),
            vec![5.0, 0.2, -0.1, 0.6, 5.0, 5.0, 0.3, 5.0, 9.0, 9.0],
        )
        .unwrap()
        .into_dyn();

        let out = sparse_embeddings(&sparse.view(), &input_ids, &attention_mask).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].indices, [9]);
        assert_eq!(out[0].values, [0.6]);
        assert_eq!(out[1].indices, [7]);
        assert_eq!(out[1].values, [0.3]);
    }

    #[test]
    fn sparse_embeddings_errors_where_a_token_has_no_weight() {
        let input_ids = arr2(&[[0i64, 9, 2], [0, 7, 2]]);
        let attention_mask = arr2(&[[1i64, 1, 1], [1, 1, 1]]);

        let rank2 = Array2::<f32>::zeros((2, 3)).into_dyn();
        assert!(sparse_embeddings(&rank2.view(), &input_ids, &attention_mask).is_err());
        let one_text = Array3::<f32>::zeros((1, 3, 1)).into_dyn();
        assert!(sparse_embeddings(&one_text.view(), &input_ids, &attention_mask).is_err());
        let no_width = Array3::<f32>::zeros((2, 3, 0)).into_dyn();
        assert!(sparse_embeddings(&no_width.view(), &input_ids, &attention_mask).is_err());

        // The last token has no weight, and it is special, so nothing is read there.
        let last_token_missing = Array3::<f32>::ones((2, 2, 1)).into_dyn();
        let out =
            sparse_embeddings(&last_token_missing.view(), &input_ids, &attention_mask).unwrap();
        assert_eq!(out[0].indices, [9]);
        // Rows and positions past the input are left unread, as before.
        let larger = Array3::<f32>::ones((3, 6, 2)).into_dyn();
        assert!(sparse_embeddings(&larger.view(), &input_ids, &attention_mask).is_ok());
    }

    #[test]
    fn colbert_vectors_takes_the_unmasked_tokens_after_the_first() {
        let attention_mask = arr2(&[[1i64, 1, 1], [1, 1, 0]]);
        let colbert = Array3::from_shape_vec((2, 2, 2), (0..8).map(|v| v as f32).collect())
            .unwrap()
            .into_dyn();

        let out = colbert_vectors(&colbert.view(), &attention_mask).unwrap();
        assert_eq!(out[0], [[0.0, 1.0], [2.0, 3.0]]);
        assert_eq!(out[1], [[4.0, 5.0]]);
    }

    #[test]
    fn colbert_vectors_errors_on_shapes_it_cannot_read() {
        let attention_mask = arr2(&[[1i64, 1, 1], [1, 1, 0]]);

        let rank2 = Array2::<f32>::zeros((2, 2)).into_dyn();
        assert!(colbert_vectors(&rank2.view(), &attention_mask).is_err());
        let full_length = Array3::<f32>::zeros((2, 3, 2)).into_dyn();
        assert!(colbert_vectors(&full_length.view(), &attention_mask).is_err());
        let one_text = Array3::<f32>::zeros((1, 2, 2)).into_dyn();
        assert!(colbert_vectors(&one_text.view(), &attention_mask).is_err());
        // An extra row is left unread, as before.
        let larger = Array3::<f32>::zeros((3, 2, 2)).into_dyn();
        assert_eq!(
            colbert_vectors(&larger.view(), &attention_mask)
                .unwrap()
                .len(),
            2
        );
    }
}
