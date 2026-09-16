//! Initialization options for the text embedding models.
//!

use crate::{
    common::{init_session_builder, Error, Result, TokenizerFiles},
    init::{HasMaxLength, InitOptionsWithLength},
    pooling::Pooling,
    EmbeddingModel, OutputKey, QuantizationMode,
};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{builder::SessionBuilder, Session},
};
use tokenizers::Tokenizer;

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for EmbeddingModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the TextEmbedding model
pub type TextInitOptions = InitOptionsWithLength<EmbeddingModel>;

/// Options for initializing UserDefinedEmbeddingModel
///
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
    /// Refuse session creation when any graph node would fall back to the CPU
    /// execution provider. This is useful for accelerator conformance tests
    /// and for applications where silent partial placement is incorrect.
    pub disable_cpu_fallback: bool,
    /// Override named free dimensions before ORT optimizes and places the
    /// graph. Static-shape execution providers such as QNN can use this with
    /// one user-defined model session per admitted shape.
    pub dimension_overrides: Vec<(String, i64)>,
    /// ONNX Runtime session configuration entries, applied with
    /// `SessionBuilder::with_config_entry`. Use this for settings that have
    /// no dedicated builder method, such as `mlas.disable_kleidiai`.
    pub session_config: Vec<(String, String)>,
}

impl InitOptionsUserDefined {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the number of intra-op threads ONNX Runtime uses. By default
    /// (`None`) all available CPU cores are used. Capping this limits CPU
    /// usage at the cost of per-inference throughput.
    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }

    pub fn with_disable_cpu_fallback(mut self, disable: bool) -> Self {
        self.disable_cpu_fallback = disable;
        self
    }

    pub fn with_dimension_override(mut self, name: impl Into<String>, size: i64) -> Self {
        self.dimension_overrides.push((name.into(), size));
        self
    }

    /// Add an ONNX Runtime session configuration entry, applied with
    /// `SessionBuilder::with_config_entry`. Call it once per entry.
    /// Example: `.with_session_config("mlas.disable_kleidiai", "1")`.
    pub fn with_session_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.session_config.push((key.into(), value.into()));
        self
    }

    /// A session builder with every option applied, plus the requested tokenizer max length.
    pub(crate) fn into_session_builder(self) -> Result<(SessionBuilder, usize)> {
        let Self {
            execution_providers,
            max_length,
            intra_threads,
            disable_cpu_fallback,
            dimension_overrides,
            session_config,
        } = self;

        let builder_error = |err: ort::Error<SessionBuilder>| Error::OrtBuilder(err.to_string());
        let mut builder = init_session_builder(execution_providers, intra_threads, session_config)?;
        if disable_cpu_fallback {
            builder = builder.with_disable_cpu_fallback().map_err(builder_error)?;
        }
        for (name, size) in dimension_overrides {
            builder = builder
                .with_dimension_override(name, size)
                .map_err(builder_error)?;
        }
        Ok((builder, max_length))
    }
}

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            intra_threads: None,
            disable_cpu_fallback: false,
            dimension_overrides: Vec::new(),
            session_config: Vec::new(),
        }
    }
}

/// Convert InitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<TextInitOptions> for InitOptionsUserDefined {
    fn from(options: TextInitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
            intra_threads: options.intra_threads,
            disable_cpu_fallback: false,
            dimension_overrides: Vec::new(),
            session_config: options.session_config,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub external_initializers: Vec<ExternalInitializerFile>,
    pub tokenizer_files: TokenizerFiles,
    pub pooling: Option<Pooling>,
    pub quantization: QuantizationMode,
    pub output_key: Option<OutputKey>,
}

/// Struct for adding external initializers to "bring your own" embedding models
///
/// The buffer is expecting the data of the external initializer and the file_name
/// must match the one referenced by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalInitializerFile {
    pub file_name: String,
    pub buffer: Vec<u8>,
}

impl UserDefinedEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            external_initializers: Vec::new(),
            tokenizer_files,
            quantization: QuantizationMode::None,
            pooling: None,
            output_key: None,
        }
    }

    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }

    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }

    pub fn with_external_initializer(mut self, file_name: String, buffer: Vec<u8>) -> Self {
        self.external_initializers
            .push(ExternalInitializerFile { file_name, buffer });
        self
    }
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) pooling: Option<Pooling>,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) quantization: QuantizationMode,
    pub(crate) output_key: Option<OutputKey>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_defined_session_controls_are_opt_in_and_composable() {
        let defaults = InitOptionsUserDefined::default();
        assert!(!defaults.disable_cpu_fallback);
        assert!(defaults.dimension_overrides.is_empty());

        let configured = InitOptionsUserDefined::new()
            .with_disable_cpu_fallback(true)
            .with_dimension_override("batch_size", 1)
            .with_dimension_override("sequence_length", 512);
        assert!(configured.disable_cpu_fallback);
        assert_eq!(
            configured.dimension_overrides,
            [
                ("batch_size".to_string(), 1),
                ("sequence_length".to_string(), 512)
            ]
        );
    }

    #[test]
    fn session_config_is_collected_and_carried_by_from() {
        let opts = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_session_config("a", "1")
            .with_session_config("b", "2");
        assert_eq!(
            opts.session_config,
            vec![("a".into(), "1".into()), ("b".into(), "2".into())]
        );

        let user_defined = InitOptionsUserDefined::from(opts);
        assert_eq!(
            user_defined.session_config,
            vec![("a".into(), "1".into()), ("b".into(), "2".into())]
        );
    }
}
