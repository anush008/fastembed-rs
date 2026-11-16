#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::{
        check_pair_truncation, encode_batch, init_session_builder, load_tokenizer, Error, Result,
    },
    models::reranking::reranker_model_list,
    RerankerModel, RerankerModelInfo,
};
use ndarray::{s, ArrayViewD};
use ort::session::Session;
use tokenizers::Tokenizer;

#[cfg(feature = "hf-hub")]
use super::RerankInitOptions;
use super::{
    OnnxSource, RerankInitOptionsUserDefined, RerankResult, TextRerank, UserDefinedRerankingModel,
    DEFAULT_BATCH_SIZE,
};

impl TextRerank {
    fn new(tokenizer: Tokenizer, session: Session) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
        }
    }

    pub fn get_model_info(model: &RerankerModel) -> RerankerModelInfo {
        TextRerank::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    pub fn list_supported_models() -> Vec<RerankerModelInfo> {
        reranker_model_list()
    }

    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: RerankInitOptions) -> Result<TextRerank> {
        use super::RerankInitOptions;
        use crate::common::pull_from_hf;

        let RerankInitOptions {
            max_length,
            model_name,
            environment,
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
            auto_device,
        } = options;

        let model_repo = pull_from_hf(model_name.to_string(), cache_dir, show_download_progress)?;

        let model_info = TextRerank::get_model_info(&model_name);
        let model_file_reference =
            model_repo
                .get(&model_info.model_file)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_info.model_file.clone(),
                    source: Box::new(e),
                })?;
        for additional_file in &model_info.additional_files {
            model_repo
                .get(additional_file)
                .map_err(|e| Error::ModelRetrieval {
                    file: additional_file.clone(),
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
        check_pair_truncation(&tokenizer)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Create a TextRerank instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' reranking models
    pub fn try_new_from_user_defined(
        model: UserDefinedRerankingModel,
        options: RerankInitOptionsUserDefined,
    ) -> Result<Self> {
        let RerankInitOptionsUserDefined {
            environment,
            execution_providers,
            max_length,
            intra_threads,
            disable_cpu_fallback,
            dimension_overrides,
            session_config,
            auto_device,
        } = options;

        let mut session_builder = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
            auto_device,
        )?;
        let builder_error = |err: ort::Error<ort::session::builder::SessionBuilder>| {
            Error::OrtBuilder(err.to_string())
        };
        if disable_cpu_fallback {
            session_builder = session_builder
                .with_disable_cpu_fallback()
                .map_err(builder_error)?;
        }
        for (name, size) in dimension_overrides {
            session_builder = session_builder
                .with_dimension_override(name, size)
                .map_err(builder_error)?;
        }
        let session = match &model.onnx_source {
            OnnxSource::Memory(bytes) => session_builder.commit_from_memory(bytes)?,
            OnnxSource::File(path) => session_builder.commit_from_file(path)?,
        };

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        check_pair_truncation(&tokenizer)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Rerank documents using the reranker model and returns the results sorted by score in descending order.
    ///
    /// Accepts a query and a collection of documents implementing [`AsRef<str>`].
    pub fn rerank<S: AsRef<str> + Send + Sync>(
        &mut self,
        query: S,
        documents: impl AsRef<[S]>,
        return_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let documents = documents.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }
        let q = query.as_ref();

        let mut scores: Vec<f32> = Vec::with_capacity(documents.len());
        for batch in documents.chunks(batch_size) {
            let inputs = batch.iter().map(|d| (q, d.as_ref())).collect();
            let mut encoded = encode_batch(&self.tokenizer, inputs)?;
            let session_inputs = encoded.session_inputs(self.need_token_type_ids)?;

            let outputs = self
                .session
                .run(session_inputs)
                .map_err(|e| Error::OrtSession(e.to_string()))?;
            let outputs = outputs
                .get("logits")
                .ok_or_else(|| Error::OutputKeyMissing {
                    key: "logits".into(),
                })?
                .try_extract_array::<f32>()
                .map_err(|e| {
                    Error::TensorExtraction(format!("Failed to extract logits tensor: {e}"))
                })?;
            scores.extend(logits_to_scores(&outputs, batch.len())?);
        }

        // Return top_n_result of type Vec<RerankResult> ordered by score in descending order, don't use binary heap
        let mut top_n_result: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                document: return_documents.then(|| documents[index].as_ref().to_string()),
                score,
                index,
            })
            .collect();
        top_n_result.sort_by(|a, b| a.score.total_cmp(&b.score).reverse());
        Ok(top_n_result)
    }
}

/// One score per document from `[batch]` or `[batch, 1]` logits.
fn logits_to_scores(logits: &ArrayViewD<f32>, documents: usize) -> Result<Vec<f32>> {
    let scores: Vec<f32> = match logits.shape() {
        [_] => logits.iter().copied().collect(),
        [_, width] if *width > 0 => logits.slice(s![.., 0]).iter().copied().collect(),
        shape => {
            return Err(Error::InvalidShape(format!(
                "Expected logits of shape [batch] or [batch, 1], got {shape:?}"
            )))
        }
    };
    if scores.len() != documents {
        return Err(Error::InvalidShape(format!(
            "Expected one score per document, got logits of shape {:?} for {documents} documents",
            logits.shape()
        )));
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array3};

    #[test]
    fn logits_to_scores_takes_batch_and_batch_by_one() {
        let flat = arr1(&[0.5f32, -1.0]).into_dyn();
        assert_eq!(logits_to_scores(&flat.view(), 2).unwrap(), [0.5, -1.0]);
        let column = arr2(&[[0.5f32], [-1.0]]).into_dyn();
        assert_eq!(logits_to_scores(&column.view(), 2).unwrap(), [0.5, -1.0]);
        let wide = arr2(&[[0.5f32, 9.0], [-1.0, 9.0]]).into_dyn();
        assert_eq!(logits_to_scores(&wide.view(), 2).unwrap(), [0.5, -1.0]);
    }

    #[test]
    fn logits_to_scores_rejects_other_shapes() {
        let rank3 = Array3::<f32>::zeros((2, 1, 1)).into_dyn();
        assert!(logits_to_scores(&rank3.view(), 2).is_err());
        let scalar = ndarray::arr0(1f32).into_dyn();
        assert!(logits_to_scores(&scalar.view(), 1).is_err());
        let empty_width = ndarray::Array2::<f32>::zeros((2, 0)).into_dyn();
        assert!(logits_to_scores(&empty_width.view(), 2).is_err());
        let short = arr1(&[0.5f32]).into_dyn();
        assert!(logits_to_scores(&short.view(), 2).is_err());
    }
}
