//! The definition of the main struct for text embeddings - [`TextEmbedding`].

#[cfg(feature = "hf-hub")]
use crate::common::HubRepo;
#[cfg(feature = "hf-hub")]
use crate::common::{init_session_builder, load_tokenizer_hf_hub};
use crate::{
    common::{encode_batch, load_tokenizer, Error, Result},
    models::{text_embedding::models_list, ModelTrait},
    pooling::Pooling,
    Embedding, EmbeddingModel, EmbeddingOutput, ModelInfo, OutputKey, QuantizationMode,
    SingleBatchOutput,
};
use ort::session::Session;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg(feature = "hf-hub")]
use super::TextInitOptions;
use super::{
    output, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel, DEFAULT_BATCH_SIZE,
};

impl TextEmbedding {
    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: TextInitOptions) -> Result<Self> {
        let TextInitOptions {
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

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = TextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference =
            model_repo
                .get(model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        for file in &model_info.additional_files {
            model_repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })?;
        }

        let post_processing = TextEmbedding::get_default_pooling_method(&model_name);

        let session = init_session_builder(
            environment,
            execution_providers,
            intra_threads,
            session_config,
            auto_device,
        )?
        .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            post_processing,
            TextEmbedding::get_quantization_mode(&model_name),
            model_info.output_key.clone(),
        ))
    }

    /// Create a TextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let (mut session_builder, max_length) = options.into_session_builder()?;
        for external_initializer_file in model.external_initializers {
            session_builder = session_builder
                .with_external_initializer_file_in_memory(
                    external_initializer_file.file_name,
                    external_initializer_file.buffer.into(),
                )
                .map_err(|err| Error::OrtBuilder(err.to_string()))?;
        }
        let session = session_builder.commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            model.pooling,
            model.quantization,
            model.output_key,
        ))
    }

    /// Private method to return an instance
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        post_process: Option<Pooling>,
        quantization: QuantizationMode,
        output_key: Option<OutputKey>,
    ) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Self {
            tokenizer,
            session,
            need_token_type_ids,
            pooling: post_process,
            quantization,
            output_key,
        }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<HubRepo> {
        use crate::common::pull_from_hf;

        let model_code = TextEmbedding::get_model_info(&model)?.model_code.clone();
        pull_from_hf(model_code, cache_dir, show_download_progress)
    }

    pub fn get_default_pooling_method(model_name: &EmbeddingModel) -> Option<Pooling> {
        match model_name {
            EmbeddingModel::AllMiniLML6V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            EmbeddingModel::BGEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEM3 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),
            EmbeddingModel::AllMpnetBaseV2 => Some(Pooling::Mean),

            EmbeddingModel::ModernBertEmbedLarge => Some(Pooling::Mean),

            EmbeddingModel::MultilingualE5Base => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Small => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Large => Some(Pooling::Mean),

            EmbeddingModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            EmbeddingModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            EmbeddingModel::GTEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15Q => Some(Pooling::Cls),

            EmbeddingModel::ClipVitB32 => Some(Pooling::Mean),

            EmbeddingModel::JinaEmbeddingsV2BaseCode => Some(Pooling::Mean),
            EmbeddingModel::JinaEmbeddingsV2BaseEN => Some(Pooling::Mean),

            EmbeddingModel::EmbeddingGemma300M => Some(Pooling::Mean),
            EmbeddingModel::EmbeddingGemma300MQ4 => Some(Pooling::Mean),
            EmbeddingModel::EmbeddingGemma300MQ => Some(Pooling::Mean),

            EmbeddingModel::SnowflakeArcticEmbedXS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedXSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedM => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLong => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedL => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedLQ => Some(Pooling::Cls),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// A dynamically quantized model fits its value range to each batch, so
    /// [`TextEmbedding::embed`] runs all the texts of a call as one batch. The `Q` variants of
    /// these models are dynamically quantized:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - snowflake/snowflake-arctic-embed (xs, s, m, m-long, l)
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(model_name: &EmbeddingModel) -> QuantizationMode {
        match model_name {
            EmbeddingModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::BGEBaseENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGELargeENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGESmallENV15Q => QuantizationMode::Static,
            EmbeddingModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
            EmbeddingModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTELargeENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedXSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedLQ => QuantizationMode::Dynamic,
            // Int8 weights with float activations: batching-safe, so not Dynamic
            EmbeddingModel::EmbeddingGemma300MQ => QuantizationMode::Static,
            // 4-bit static quantization: batching-safe, so not Dynamic
            EmbeddingModel::EmbeddingGemma300MQ4 => QuantizationMode::None,
            EmbeddingModel::AllMiniLML6V2
            | EmbeddingModel::AllMiniLML12V2
            | EmbeddingModel::AllMpnetBaseV2
            | EmbeddingModel::BGEBaseENV15
            | EmbeddingModel::BGELargeENV15
            | EmbeddingModel::BGESmallENV15
            | EmbeddingModel::BGESmallZHV15
            | EmbeddingModel::BGELargeZHV15
            | EmbeddingModel::BGEM3
            | EmbeddingModel::NomicEmbedTextV1
            | EmbeddingModel::NomicEmbedTextV15
            | EmbeddingModel::ParaphraseMLMiniLML12V2
            | EmbeddingModel::ParaphraseMLMpnetBaseV2
            | EmbeddingModel::ModernBertEmbedLarge
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::MultilingualE5Base
            | EmbeddingModel::MultilingualE5Large
            | EmbeddingModel::MxbaiEmbedLargeV1
            | EmbeddingModel::GTEBaseENV15
            | EmbeddingModel::GTELargeENV15
            | EmbeddingModel::ClipVitB32
            | EmbeddingModel::JinaEmbeddingsV2BaseCode
            | EmbeddingModel::JinaEmbeddingsV2BaseEN
            | EmbeddingModel::EmbeddingGemma300M
            | EmbeddingModel::SnowflakeArcticEmbedXS
            | EmbeddingModel::SnowflakeArcticEmbedS
            | EmbeddingModel::SnowflakeArcticEmbedM
            | EmbeddingModel::SnowflakeArcticEmbedMLong
            | EmbeddingModel::SnowflakeArcticEmbedL => QuantizationMode::None,
        }
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<&ModelInfo<EmbeddingModel>> {
        EmbeddingModel::get_model_info(model).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Model {model:?} not found. Please check if the model is supported \
                by the current version."
            ))
        })
    }

    /// Method to generate the raw session outputs wrapped in an [`EmbeddingOutput`]
    /// instance, which can be used to extract the embeddings with default or custom
    /// methods as well as output key precedence.
    ///
    /// # Note
    ///
    /// This is a lower level method than [`TextEmbedding::embed`], and is useful
    /// when you need to extract the session outputs in a custom way.
    ///
    /// If you want to extract the embeddings directly, use [`TextEmbedding::embed`].
    ///
    /// If you want to use the raw session outputs, use [`EmbeddingOutput::into_raw`]
    /// on the output of this method.
    ///
    /// If you want to choose a different export key or customize the way the batch
    /// arrays are aggregated, you can define your own array transformer
    /// and use it on [`EmbeddingOutput::export_with_transformer`] to extract the
    /// embeddings with your custom output type.
    pub fn transform<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
        let texts = texts.as_ref();
        let Some(batch_size) = self.batch_size_for(texts.len(), batch_size)? else {
            return Ok(EmbeddingOutput::new([]));
        };

        let batches = texts
            .chunks(batch_size)
            .map(|batch| self.run_batch(batch))
            .collect::<Result<Vec<_>>>()?;

        Ok(EmbeddingOutput::new(batches))
    }

    /// The number of texts in one session run. `None` when there is no text to run.
    fn batch_size_for(&self, texts: usize, batch_size: Option<usize>) -> Result<Option<usize>> {
        if texts == 0 && batch_size != Some(0) {
            return Ok(None);
        }
        // Determine the batch size according to the quantization method used.
        // Default if not specified
        let batch_size = match self.quantization {
            QuantizationMode::Dynamic => {
                if let Some(batch_size) = batch_size {
                    if batch_size < texts {
                        Err(Error::InvalidArgument(
                            "Dynamic quantization cannot be used with batching. \
                            This is due to the dynamic quantization process adjusting \
                            the data range to fit each batch, making the embeddings \
                            incompatible across batches. Try specifying a batch size \
                            of `None`, or use a model with static or no quantization."
                                .into(),
                        ))
                    } else {
                        Ok(texts)
                    }
                } else {
                    Ok(texts)
                }
            }
            _ => Ok(batch_size.unwrap_or(DEFAULT_BATCH_SIZE)),
        }?;
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }
        Ok(Some(batch_size))
    }

    fn run_batch<S: AsRef<str> + Send + Sync>(&mut self, batch: &[S]) -> Result<SingleBatchOutput> {
        let inputs = batch.iter().map(|text| text.as_ref()).collect();
        let mut encoded = encode_batch(&self.tokenizer, inputs)?;
        let session_inputs = encoded.session_inputs(self.need_token_type_ids)?;

        let outputs_map = self
            .session
            .run(session_inputs)
            .map_err(|e| Error::OrtSession(e.to_string()))?
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Ok(SingleBatchOutput {
            outputs: outputs_map,
            attention_mask_array: encoded.attention_mask,
        })
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    ///
    /// The output is a [`Vec`] of [`Embedding`]s.
    ///
    /// # Note
    ///
    /// This method is a higher level method than [`TextEmbedding::transform`] by utilizing
    /// the default output precedence and array transformer for the [`TextEmbedding`] model.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let texts = texts.as_ref();
        let Some(batch_size) = self.batch_size_for(texts.len(), batch_size)? else {
            return Ok(Vec::new());
        };
        let precedence = match &self.output_key {
            Some(output_key) => vec![output_key.clone()],
            None => output::OUTPUT_TYPE_PRECEDENCE.to_vec(),
        };
        let transformer =
            output::transformer_with_precedence(precedence.as_slice(), self.pooling.clone());

        // Each batch is pooled before the next one runs, so one batch of model output is held
        // at a time.
        let mut embeddings = Vec::with_capacity(texts.len());
        for (index, batch) in texts.chunks(batch_size).enumerate() {
            let output = self.run_batch(batch)?;
            match transformer(std::slice::from_ref(&output)) {
                Ok(vectors) => embeddings.extend(vectors),
                Err(error) => {
                    // An error in running a later batch comes first, as it does when every
                    // batch runs before the first one is pooled.
                    drop(output);
                    for later in texts.chunks(batch_size).skip(index + 1) {
                        self.run_batch(later)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_variants_have_explicit_pooling() {
        for variant in crate::models::text_embedding::all_variants() {
            assert!(
                TextEmbedding::get_default_pooling_method(&variant).is_some(),
                "{variant:?}"
            );
            let _ = TextEmbedding::get_quantization_mode(&variant);
        }
    }

    fn model(quantization: QuantizationMode) -> TextEmbedding {
        use crate::common::tests::{bert_tokenizer_files, relu_model_bytes};

        let model = UserDefinedEmbeddingModel::new(relu_model_bytes(), bert_tokenizer_files())
            .with_quantization(quantization);
        TextEmbedding::try_new_from_user_defined(model, Default::default()).unwrap()
    }

    #[test]
    fn no_texts_embed_to_nothing_whatever_the_quantization() {
        for quantization in [QuantizationMode::None, QuantizationMode::Dynamic] {
            let mut model = model(quantization);
            let no_texts: [&str; 0] = [];
            for batch_size in [None, Some(1)] {
                assert!(model.embed(no_texts, batch_size).unwrap().is_empty());
                let output = model.transform(no_texts, batch_size).unwrap();
                assert!(output.into_raw().is_empty());
            }
            assert!(matches!(
                model.embed(no_texts, Some(0)),
                Err(Error::InvalidArgument(_))
            ));
        }
    }

    /// A model that embeds each text to its token ids: `[0, 2, .., 1]`, one 2 for each word.
    fn token_ids_model(tokens: Option<u8>, output_key: Option<OutputKey>) -> TextEmbedding {
        use crate::common::tests::{bert_tokenizer_files, token_ids_model_bytes};

        let mut model =
            UserDefinedEmbeddingModel::new(token_ids_model_bytes(tokens), bert_tokenizer_files());
        model.output_key = output_key;
        TextEmbedding::try_new_from_user_defined(model, Default::default()).unwrap()
    }

    #[test]
    fn embed_gives_the_vectors_of_transform_for_every_batch_size() {
        let mut model = token_ids_model(None, None);
        let texts = [
            "hello",
            "hello hello hello",
            "hello hello",
            "hello",
            "hello hello",
        ];
        for batch_size in [None, Some(1), Some(2), Some(3), Some(usize::MAX)] {
            let embedded = model.embed(texts, batch_size).unwrap();
            let transformed = model
                .transform(texts, batch_size)
                .unwrap()
                .export_with_transformer(output::transformer_with_precedence(
                    output::OUTPUT_TYPE_PRECEDENCE,
                    None,
                ))
                .unwrap();
            assert_eq!(embedded, transformed, "batch size {batch_size:?}");
            assert_eq!(embedded.len(), texts.len());
        }
        // One text alone: its ids 0, 2 and 1, normalized.
        let alone = &model.embed(["hello"], None).unwrap()[0];
        let norm = 5f32.sqrt();
        for (value, id) in alone.iter().zip([0.0, 2.0, 1.0]) {
            assert!((value - id / norm).abs() < 1e-6, "{alone:?}");
        }
    }

    #[test]
    fn an_error_in_a_later_batch_comes_before_a_pooling_error() {
        // Rows of three tokens run. No output has the name asked for, so pooling fails.
        let mut model = token_ids_model(Some(3), Some(OutputKey::ByName("missing")));
        let (fits, too_long) = ("hello", "hello hello hello");

        assert!(matches!(model.embed([fits], Some(1)), Err(Error::Other(_))));
        for texts in [[fits, too_long], [too_long, fits]] {
            assert!(matches!(
                model.embed(texts, Some(1)),
                Err(Error::OrtSession(_))
            ));
        }

        let mut model = token_ids_model(Some(3), None);
        assert_eq!(model.embed([fits, fits], Some(1)).unwrap().len(), 2);
    }

    #[test]
    fn a_dynamically_quantized_model_refuses_to_split_its_texts() {
        let mut model = model(QuantizationMode::Dynamic);
        assert!(matches!(
            model.embed(["hello", "hello hello"], Some(1)),
            Err(Error::InvalidArgument(_))
        ));
        // Two texts in one batch reach the test model, which takes no token ids.
        for batch_size in [None, Some(2), Some(3)] {
            assert!(matches!(
                model.embed(["hello", "hello hello"], batch_size),
                Err(Error::OrtSession(_))
            ));
        }
    }
}
