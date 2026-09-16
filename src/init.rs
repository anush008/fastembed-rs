use crate::get_cache_dir;
use ort::execution_providers::ExecutionProviderDispatch;
use std::path::PathBuf;

pub trait HasMaxLength {
    const MAX_LENGTH: usize;
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsWithLength<M> {
    pub model_name: M,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    pub max_length: usize,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
    /// ONNX Runtime session configuration entries, applied with
    /// `SessionBuilder::with_config_entry`. Use this for settings that have
    /// no dedicated builder method, such as `mlas.disable_kleidiai`.
    pub session_config: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptions<M> {
    pub model_name: M,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
    /// ONNX Runtime session configuration entries, applied with
    /// `SessionBuilder::with_config_entry`. Use this for settings that have
    /// no dedicated builder method, such as `mlas.disable_kleidiai`.
    pub session_config: Vec<(String, String)>,
}

impl<M: Default + HasMaxLength> Default for InitOptionsWithLength<M> {
    fn default() -> Self {
        Self {
            model_name: M::default(),
            execution_providers: Default::default(),
            cache_dir: get_cache_dir().into(),
            show_download_progress: true,
            max_length: M::MAX_LENGTH,
            intra_threads: None,
            session_config: Vec::new(),
        }
    }
}

impl<M: Default> Default for InitOptions<M> {
    fn default() -> Self {
        Self {
            model_name: M::default(),
            execution_providers: Default::default(),
            cache_dir: get_cache_dir().into(),
            show_download_progress: true,
            intra_threads: None,
            session_config: Vec::new(),
        }
    }
}

impl<M: Default + HasMaxLength> InitOptionsWithLength<M> {
    /// Create a new InitOptionsWithLength with the given model name
    pub fn new(model_name: M) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    /// Set the maximum length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the cache directory for the model file
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set the execution providers for the model
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

    /// Set whether to show download progress
    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}

impl<M: Default> InitOptions<M> {
    /// Create a new InitOptions with the given model name
    pub fn new(model_name: M) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    /// Set the cache directory for the model file
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set the execution providers for the model
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

    /// Set whether to show download progress
    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_threads_defaults_none_and_builder_sets() {
        let o = InitOptions::<crate::ImageEmbeddingModel>::default();
        assert_eq!(o.intra_threads, None);
        let o = o.with_intra_threads(4);
        assert_eq!(o.intra_threads, Some(4));
    }
}
