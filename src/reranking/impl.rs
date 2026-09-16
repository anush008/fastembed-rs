#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::{encode_batch, init_session_builder, load_tokenizer, Error, Result},
    models::reranking::reranker_model_list,
    RerankerModel, RerankerModelInfo,
};
use ndarray::s;
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
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
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

        let session = init_session_builder(execution_providers, intra_threads, session_config)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
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
            execution_providers,
            max_length,
            intra_threads,
            disable_cpu_fallback,
            dimension_overrides,
            session_config,
        } = options;

        let mut session_builder =
            init_session_builder(execution_providers, intra_threads, session_config)?;
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
            scores.extend(outputs.slice(s![.., 0]).iter().copied());
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
