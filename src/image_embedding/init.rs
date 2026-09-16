use super::utils::Compose;
use crate::{init::InitOptions, ImageEmbeddingModel};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};

/// Options for initializing the ImageEmbedding model
pub type ImageInitOptions = InitOptions<ImageEmbeddingModel>;

/// Options for initializing UserDefinedImageEmbeddingModel
///
/// Model files are held by the UserDefinedImageEmbeddingModel struct
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ImageInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
    /// ONNX Runtime session configuration entries, applied with
    /// `SessionBuilder::with_config_entry`. Use this for settings that have
    /// no dedicated builder method, such as `mlas.disable_kleidiai`.
    pub session_config: Vec<(String, String)>,
}

impl ImageInitOptionsUserDefined {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Set the number of intra-op threads ONNX Runtime uses. By default
    /// (`None`) all available CPU cores are used. Capping this limits CPU
    /// usage at the cost of per-inference throughput.
    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }

    /// Add an ONNX Runtime session configuration entry, applied with
    /// `SessionBuilder::with_config_entry`. Call it once per entry.
    /// Example: `.with_session_config("mlas.disable_kleidiai", "1")`.
    pub fn with_session_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.session_config.push((key.into(), value.into()));
        self
    }
}

/// Convert ImageInitOptions to ImageInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<ImageInitOptions> for ImageInitOptionsUserDefined {
    fn from(options: ImageInitOptions) -> Self {
        ImageInitOptionsUserDefined {
            execution_providers: options.execution_providers,
            intra_threads: options.intra_threads,
            session_config: options.session_config,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and preprocessor_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedImageEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub preprocessor_file: Vec<u8>,
}

impl UserDefinedImageEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, preprocessor_file: Vec<u8>) -> Self {
        Self {
            onnx_file,
            preprocessor_file,
        }
    }
}

/// Rust representation of the ImageEmbedding model
pub struct ImageEmbedding {
    pub(crate) preprocessor: Compose,
    pub(crate) session: Session,
}
