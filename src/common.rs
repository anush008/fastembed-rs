#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
use hf_hub::api::sync::{ApiBuilder, ApiError, ApiRepo};
#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{
    environment::Environment,
    ep::ExecutionProviderDispatch,
    session::{
        builder::{AutoDevicePolicy, GraphOptimizationLevel, SessionBuilder},
        SessionInputValue,
    },
    value::Value,
};
#[cfg(feature = "onnx")]
use std::borrow::Cow;
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::sync::{Mutex, PoisonError};
#[cfg(feature = "onnx")]
use tokenizers::{AddedToken, EncodeInput, Encoding, PaddingParams, PaddingStrategy};
#[cfg(any(
    feature = "onnx",
    feature = "qwen3-text",
    feature = "nomic-v2-moe",
    feature = "bert-text"
))]
use tokenizers::{Tokenizer, TruncationParams};

const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";

pub fn get_cache_dir() -> String {
    std::env::var("FASTEMBED_CACHE_DIR").unwrap_or(DEFAULT_CACHE_DIR.into())
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseEmbedding {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

/// Type alias for the embedding vector
pub type Embedding = Vec<f32>;

/// Error type returned by fastembed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Failed to retrieve model file '{file}'")]
    ModelRetrieval {
        file: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Invalid tokenizer configuration: {0}")]
    TokenizerConfig(String),

    #[error("Failed to tokenize input: {0}")]
    Tokenization(String),

    #[error("Tokenizer returned empty encodings for the batch")]
    EmptyTokenizations,

    #[cfg(feature = "onnx")]
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Failed to build ONNX session: {0}")]
    OrtBuilder(String),

    #[error("ONNX session error: {0}")]
    OrtSession(String),

    #[error("Output tensor '{key}' not found")]
    OutputKeyMissing { key: String },

    #[error("Failed to extract tensor: {0}")]
    TensorExtraction(String),

    #[error("Failed to decode image: {0}")]
    ImageDecode(String),

    #[error("Invalid preprocessor configuration: {0}")]
    PreprocessorConfig(String),

    #[error("Image transform error: {0}")]
    ImageTransform(String),

    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("an ONNX Runtime environment already exists in this process, pass it with `with_environment`")]
    EnvironmentRequired,

    #[error("{0}")]
    Other(String),
}

/// Type alias for `Result` returning the fastembed [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "hf-hub")]
impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        Error::Other(format!("HuggingFace API error: {e}"))
    }
}

// Tokenizer files for "bring your own" models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_file: Vec<u8>,
    pub config_file: Vec<u8>,
    pub special_tokens_map_file: Vec<u8>,
    pub tokenizer_config_file: Vec<u8>,
}

/// Loading tokenizer files from the Hugging Face Hub is separate from the main
/// `load_tokenizer` function, which expects bytes from any source.
#[cfg(all(feature = "onnx", feature = "hf-hub"))]
pub fn load_tokenizer_hf_hub(model_repo: HubRepo, max_length: usize) -> Result<Tokenizer> {
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json")?)?,
        config_file: std::fs::read(&model_repo.get("config.json")?)?,
        special_tokens_map_file: std::fs::read(&model_repo.get("special_tokens_map.json")?)?,

        tokenizer_config_file: std::fs::read(&model_repo.get("tokenizer_config.json")?)?,
    };

    load_tokenizer(tokenizer_files, max_length)
}

/// tokenizers subtracts the special-token count from `max_length` unchecked, so reject
/// smaller values.
#[cfg(any(
    feature = "onnx",
    feature = "qwen3-text",
    feature = "nomic-v2-moe",
    feature = "bert-text"
))]
pub(crate) fn truncation_to(tokenizer: &Tokenizer, max_length: usize) -> Result<TruncationParams> {
    check_special_tokens(tokenizer, max_length, false)?;
    Ok(TruncationParams {
        max_length,
        ..Default::default()
    })
}

#[cfg(any(
    feature = "onnx",
    feature = "qwen3-text",
    feature = "nomic-v2-moe",
    feature = "bert-text"
))]
fn check_special_tokens(tokenizer: &Tokenizer, max_length: usize, pair: bool) -> Result<()> {
    use tokenizers::PostProcessor;

    let special_tokens = tokenizer
        .get_post_processor()
        .map_or(0, |p| p.added_tokens(pair));
    if max_length < special_tokens {
        let input = if pair { "pair" } else { "text" };
        return Err(Error::InvalidArgument(format!(
            "max_length is {max_length} and must be at least {special_tokens}, the number of special tokens the tokenizer adds to a {input}"
        )));
    }
    Ok(())
}

/// The pair-encoding counterpart of the check in `truncation_to`, for rerankers.
#[cfg(feature = "onnx")]
pub(crate) fn check_pair_truncation(tokenizer: &Tokenizer) -> Result<()> {
    match tokenizer.get_truncation() {
        Some(truncation) => check_special_tokens(tokenizer, truncation.max_length, true),
        None => Ok(()),
    }
}

/// Function can be called directly from the try_new_from_user_defined function (providing file bytes)
///
/// Or indirectly from the try_new function via load_tokenizer_hf_hub (converting HF files to bytes)
#[cfg(feature = "onnx")]
pub fn load_tokenizer(tokenizer_files: TokenizerFiles, max_length: usize) -> Result<Tokenizer> {
    let unreadable = |file: &str, cause: &dyn std::fmt::Display| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Error building TokenizerFiles for UserDefinedEmbeddingModel. \
                 Could not read {file} file: {cause}"
            ),
        )
    };

    // Deserialize each tokenizer file
    let config: serde_json::Value = serde_json::from_slice(&tokenizer_files.config_file)
        .map_err(|e| unreadable("config.json", &e))?;
    let special_tokens_map: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.special_tokens_map_file)
            .map_err(|e| unreadable("special_tokens_map.json", &e))?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.tokenizer_config_file)
            .map_err(|e| unreadable("tokenizer_config.json", &e))?;
    let mut tokenizer: tokenizers::Tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_files.tokenizer_file)
            .map_err(|e| unreadable("tokenizer.json", &e))?;

    // For BGEBaseSmall, model_max_length is set to 1000000000000000019884624838656.
    // This value fits in an f64.
    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .ok_or_else(|| {
            Error::TokenizerConfig(
                "tokenizer_config.json is missing a numeric `model_max_length` field".into(),
            )
        })? as f32;
    let max_length = max_length.min(model_max_length as usize);
    let pad_token_value = &tokenizer_config["pad_token"];
    let pad_token: String = pad_token_value
        .as_str()
        .or_else(|| pad_token_value["content"].as_str())
        .ok_or_else(|| {
            Error::TokenizerConfig(
                "tokenizer_config.json is missing a string `pad_token` field".into(),
            )
        })?
        .into();
    let pad_id = config["pad_token_id"]
        .as_u64()
        .map(|id| id as u32)
        .or_else(|| tokenizer.token_to_id(&pad_token))
        .unwrap_or(0);
    let truncation = truncation_to(&tokenizer, max_length)?;

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            // TODO: the user should be able to choose the padding strategy
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(truncation))
        .map_err(|e| Error::TokenizerConfig(e.to_string()))?
        .clone();
    if let serde_json::Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if value.is_string() {
                if let Some(content) = value.as_str() {
                    // tokenizer.json already has the token, with the flags to match it by.
                    if tokenizer.get_added_vocabulary().is_special_token(content) {
                        continue;
                    }
                    tokenizer
                        .add_special_tokens([AddedToken {
                            content: content.into(),
                            special: true,
                            ..Default::default()
                        }])
                        .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
                }
            } else if value.is_object() {
                if let (
                    Some(content),
                    Some(single_word),
                    Some(lstrip),
                    Some(rstrip),
                    Some(normalized),
                ) = (
                    value["content"].as_str(),
                    value["single_word"].as_bool(),
                    value["lstrip"].as_bool(),
                    value["rstrip"].as_bool(),
                    value["normalized"].as_bool(),
                ) {
                    tokenizer
                        .add_special_tokens([AddedToken {
                            content: content.into(),
                            special: true,
                            single_word,
                            lstrip,
                            rstrip,
                            normalized,
                        }])
                        .map_err(|e| Error::TokenizerConfig(e.to_string()))?;
                }
            }
        }
    }
    Ok(tokenizer.into())
}

/// `reader.decode()`, which goes by the file extension. When that fails, the file's content
/// names the format instead. The first error is the one returned.
#[cfg(any(feature = "image-models", feature = "qwen3-vl"))]
pub(crate) fn decode_image(
    path: &std::path::Path,
    reader: image::ImageReader<std::io::BufReader<std::fs::File>>,
) -> image::ImageResult<image::DynamicImage> {
    let by_extension = match reader.decode() {
        Ok(image) => return Ok(image),
        Err(err) => err,
    };
    // A pipe cannot be read a second time, and opening it again blocks.
    if !path.is_file() {
        return Err(by_extension);
    }
    image::ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .ok()
        .and_then(|reader| reader.decode().ok())
        .filter(|image| image.width() > 0 && image.height() > 0)
        .ok_or(by_extension)
}

#[cfg(feature = "onnx")]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = (v.iter().map(|val| val * val).sum::<f32>()).sqrt();
    let epsilon = 1e-12;

    // We add the super-small epsilon to avoid dividing by zero
    v.iter().map(|&val| val / (norm + epsilon)).collect()
}

/// Pulls a model repo from HuggingFace.
/// HF_HOME decides the location of the cache folder
/// HF_ENDPOINT modifies the URL for the HuggingFace location.
/// HF_TOKEN authenticates the requests. Without it the token written by
/// `huggingface-cli login` is used when present.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
pub fn pull_from_hf(
    model_name: String,
    default_cache_dir: PathBuf,
    show_download_progress: bool,
) -> Result<HubRepo> {
    use std::env;

    let cache_dir = env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or(default_cache_dir);

    let endpoint = hf_endpoint(env::var("HF_ENDPOINT").ok());

    let token = env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .or_else(default_hf_token);

    // `ApiBuilder::new` panics without a home dir.
    let api = ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir))
        .with_endpoint(endpoint)
        .with_token(token)
        .with_progress(show_download_progress)
        .build()
        .map_err(|e| Error::Other(format!("Failed to initialize HuggingFace API: {e}")))?;

    Ok(HubRepo(api.model(model_name)))
}

/// A model repository on the Hugging Face Hub.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
pub struct HubRepo(ApiRepo);

#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
impl HubRepo {
    /// The path of `file` in the cache. The file is downloaded when it is not there.
    ///
    /// hf-hub returns an error for a file that another thread or process has been downloading
    /// for five seconds. This waits while that download writes to disk, and through a minute
    /// in which it writes nothing, and then finds the file in the cache.
    pub fn get(&self, file: &str) -> std::result::Result<PathBuf, ApiError> {
        wait_for_download(
            || self.0.get(file),
            || std::thread::sleep(std::time::Duration::from_secs(1)),
        )
    }
}

/// `fetch`, tried again when it fails on the lock of a download that writes to disk. `pause`
/// is the wait between two looks at the partial file of that download: one second.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn wait_for_download<T>(
    mut fetch: impl FnMut() -> std::result::Result<T, ApiError>,
    mut pause: impl FnMut(),
) -> std::result::Result<T, ApiError> {
    // A download that writes nothing for this many pauses in a row is stalled. A network that
    // stops for a moment is not.
    const STALLED_AFTER_PAUSES: u32 = 60;

    let mut partial_size = None;
    loop {
        match fetch() {
            Err(ApiError::LockAcquisition(lock))
                if partial_file_changed(&lock, &mut partial_size) =>
            {
                // A finished download renames its partial file. A stalled one may have
                // died, and then the next try takes its place.
                let mut idle = 0;
                while partial_size.is_some() && idle < STALLED_AFTER_PAUSES {
                    pause();
                    idle = match partial_file_changed(&lock, &mut partial_size) {
                        true => 0,
                        false => idle + 1,
                    };
                }
            }
            result => return result,
        }
    }
}

/// Whether the partial file of the download that holds `lock` changed since the last call.
/// `size` keeps its length between calls, and starts as `None`.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn partial_file_changed(lock: &std::path::Path, size: &mut Option<u64>) -> bool {
    let current = std::fs::metadata(lock.with_extension("part"))
        .map(|file| file.len())
        .ok();
    std::mem::replace(size, current) != current
}

/// An empty `HF_ENDPOINT` has no host to download from, so it means the default one. An empty
/// `HF_HOME` or `FASTEMBED_CACHE_DIR` keeps meaning the working directory, where caches made
/// with it already are. A trailing slash would give `//` in every file URL.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn hf_endpoint(value: Option<String>) -> String {
    value
        .map(|endpoint| endpoint.trim_end_matches('/').to_string())
        .filter(|endpoint| !endpoint.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// Token written by `huggingface-cli login`: in `HF_HOME` when that is set and holds one, else
/// in the home dir. `Cache::default` panics without a home dir.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn default_hf_token() -> Option<String> {
    let hf_home = std::env::var_os("HF_HOME").filter(|home| !home.is_empty());
    if let Some(token) = hf_home.and_then(|home| hf_home_token(home.as_ref())) {
        return Some(token);
    }
    let has_home = (std::env::var_os("HOME").is_some()
        || std::env::var_os("USERPROFILE").is_some())
        && std::env::home_dir().is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "onnx",
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn hf_home_token(hf_home: &std::path::Path) -> Option<String> {
    hf_hub::Cache::new(hf_home.join("hub")).token()
}

/// Safetensors files of a repo, resolving `model.safetensors.index.json` for sharded checkpoints.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
pub(crate) fn safetensors_weight_files(repo: &HubRepo) -> Result<Vec<PathBuf>> {
    const SINGLE_FILE: &str = "model.safetensors";
    const INDEX_FILE: &str = "model.safetensors.index.json";

    let single_file_error = match repo.get(SINGLE_FILE) {
        Ok(path) => return Ok(vec![path]),
        Err(err) => err,
    };
    // The repo has one of the two files, so the cause can be in either error.
    let index_path = repo
        .get(INDEX_FILE)
        .map_err(|index_error| Error::ModelRetrieval {
            file: format!("{SINGLE_FILE} or {INDEX_FILE}"),
            source: format!("{SINGLE_FILE}: {single_file_error}; {INDEX_FILE}: {index_error}")
                .into(),
        })?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path)?)
        .map_err(|e| Error::Other(format!("Failed to parse {INDEX_FILE}: {e}")))?;

    shard_files(&index, INDEX_FILE)?
        .iter()
        .map(|file| {
            repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })
        })
        .collect()
}

/// `err` as text, with the cause that its `Display` leaves out. The candle loaders carry
/// errors as text.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
pub(crate) fn error_with_cause(err: Error) -> String {
    match std::error::Error::source(&err) {
        Some(cause) => format!("{err}: {cause}"),
        None => err.to_string(),
    }
}

/// Shard file names of a safetensors index, each one a path inside the repo.
#[cfg(all(
    feature = "hf-hub",
    any(
        feature = "qwen3-text",
        feature = "nomic-v2-moe",
        feature = "bert-text"
    )
))]
fn shard_files(index: &serde_json::Value, index_file: &str) -> Result<Vec<String>> {
    use std::path::{Component, Path};

    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Other(format!("{index_file} has no `weight_map` object")))?
        .values()
        .filter_map(|file| file.as_str().map(str::to_string))
        .collect();
    shards.sort_unstable();
    shards.dedup();

    let leaves_repo = |file: &&String| {
        Path::new(file).components().any(|part| {
            matches!(
                part,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        })
    };
    if let Some(file) = shards.iter().find(leaves_repo) {
        return Err(Error::Other(format!(
            "{index_file} names `{file}`, which is outside the repository"
        )));
    }
    Ok(shards)
}

/// One tokenized batch as `[batch, sequence]` tensors.
#[cfg(feature = "onnx")]
pub(crate) struct EncodedBatch {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub token_type_ids: Array2<i64>,
}

#[cfg(feature = "onnx")]
impl EncodedBatch {
    /// ONNX session inputs. `token_type_ids` is moved out and attached only when the graph
    /// declares it, while `input_ids` and `attention_mask` stay available for post-processing.
    pub fn session_inputs(
        &mut self,
        need_token_type_ids: bool,
    ) -> Result<Vec<(Cow<'static, str>, SessionInputValue<'static>)>> {
        let mut inputs = ort::inputs![
            "input_ids" => Value::from_array(self.input_ids.clone())?,
            "attention_mask" => Value::from_array(self.attention_mask.clone())?,
        ];
        if need_token_type_ids {
            let token_type_ids = std::mem::take(&mut self.token_type_ids);
            inputs.push((
                "token_type_ids".into(),
                Value::from_array(token_type_ids)?.into(),
            ));
        }
        Ok(inputs)
    }
}

/// Tokenize a batch into padded `[batch, sequence]` tensors.
#[cfg(feature = "onnx")]
pub(crate) fn encode_batch<'s, E>(tokenizer: &Tokenizer, inputs: Vec<E>) -> Result<EncodedBatch>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| Error::Tokenization(format!("Failed to encode the batch: {e}")))?;
    encodings_to_batch(&encodings)
}

#[cfg(feature = "onnx")]
fn encodings_to_batch(encodings: &[Encoding]) -> Result<EncodedBatch> {
    let encoding_length = encodings.first().ok_or(Error::EmptyTokenizations)?.len();
    let batch_size = encodings.len();
    let max_size = encoding_length * batch_size;

    let mut ids = Vec::with_capacity(max_size);
    let mut mask = Vec::with_capacity(max_size);
    let mut type_ids = Vec::with_capacity(max_size);
    for encoding in encodings {
        ids.extend(encoding.get_ids().iter().map(|&x| x as i64));
        mask.extend(encoding.get_attention_mask().iter().map(|&x| x as i64));
        type_ids.extend(encoding.get_type_ids().iter().map(|&x| x as i64));
    }

    let shape = (batch_size, encoding_length);
    let to_array = |data: Vec<i64>| {
        Array2::from_shape_vec(shape, data).map_err(|e| Error::InvalidShape(e.to_string()))
    };
    Ok(EncodedBatch {
        input_ids: to_array(ids)?,
        attention_mask: to_array(mask)?,
        token_type_ids: to_array(type_ids)?,
    })
}

#[cfg(feature = "onnx")]
fn default_environment() -> Result<Environment> {
    static DEFAULT: Mutex<Option<Environment>> = Mutex::new(None);
    let mut default = DEFAULT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(environment) = default.as_ref() {
        return Ok(environment.clone());
    }
    #[cfg(not(feature = "ort-load-dynamic"))]
    let builder = ort::init();
    // `ort::init` panics when the library does not load. This is the path it looks at.
    #[cfg(feature = "ort-load-dynamic")]
    let builder = {
        use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
        let path = std::env::var("ORT_DYLIB_PATH")
            .ok()
            .filter(|path| !path.is_empty())
            .unwrap_or_else(|| format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}"));
        ort::init_from(path).map_err(|e| Error::Other(format!("ONNX Runtime: {e}")))?
    };
    let environment = builder.build().map_err(|e| {
        if e.message().contains("only one environment") {
            Error::EnvironmentRequired
        } else {
            Error::Ort(e)
        }
    })?;
    *default = Some(environment.clone());
    Ok(environment)
}

/// The requested thread count, else one thread per core. ONNX Runtime picks the count itself
/// when the core count is unknown.
#[cfg(feature = "onnx")]
fn intra_threads_or(
    requested: Option<usize>,
    cores: std::io::Result<std::num::NonZeroUsize>,
) -> Option<usize> {
    requested.or_else(|| cores.ok().map(std::num::NonZeroUsize::get))
}

#[cfg(feature = "onnx")]
pub(crate) fn init_session_builder(
    environment: Option<Environment>,
    execution_providers: Vec<ExecutionProviderDispatch>,
    intra_threads: Option<usize>,
    session_config: Vec<(String, String)>,
    auto_device: Option<AutoDevicePolicy>,
) -> Result<SessionBuilder> {
    let threads = intra_threads_or(intra_threads, std::thread::available_parallelism());

    #[cfg(feature = "directml")]
    let has_directml = execution_providers
        .iter()
        .any(|ep| ep.name() == <ort::ep::DirectML as ort::ep::ExecutionProvider>::NAME);
    #[cfg(not(feature = "directml"))]
    let has_directml = false;

    let builder_error = |err: ort::Error<SessionBuilder>| Error::OrtBuilder(err.to_string());

    let environment = match environment {
        Some(environment) => environment,
        None => default_environment()?,
    };

    let mut builder = ort::session::Session::builder(&environment)?
        .with_execution_providers(execution_providers)
        .map_err(builder_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(builder_error)?;
    if let Some(threads) = threads {
        builder = builder.with_intra_threads(threads).map_err(builder_error)?;
    }

    for (key, value) in session_config {
        builder = builder
            .with_config_entry(&key, &value)
            .map_err(builder_error)?;
    }

    if let Some(policy) = auto_device {
        builder = builder.with_auto_device(policy).map_err(builder_error)?;
    }

    if has_directml {
        builder = builder
            .with_memory_pattern(false)
            .map_err(builder_error)?
            .with_parallel_execution(false)
            .map_err(builder_error)?;
    }

    Ok(builder)
}

#[cfg(all(test, any(feature = "image-models", feature = "qwen3-vl")))]
pub(crate) mod image_tests {
    use super::decode_image;
    use std::path::{Path, PathBuf};

    /// A directory of test files, removed when the test ends.
    pub(crate) struct Files(PathBuf);

    impl Files {
        pub(crate) fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("fastembed-{test}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        pub(crate) fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for Files {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) fn png() -> Vec<u8> {
        let mut bytes = Vec::new();
        image::DynamicImage::new_rgb8(3, 2)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        bytes
    }

    fn decode(path: &Path) -> image::ImageResult<image::DynamicImage> {
        decode_image(path, image::ImageReader::open(path).unwrap())
    }

    fn extension_error(path: &Path) -> String {
        let reader = image::ImageReader::open(path).unwrap();
        reader.decode().unwrap_err().to_string()
    }

    #[test]
    fn decode_image_reads_a_file_with_a_wrong_or_missing_extension() {
        let files = Files::new("decode-ok");
        for name in ["right.png", "wrong.jpg", "none"] {
            let image = decode(&files.write(name, &png())).unwrap();
            assert_eq!((image.width(), image.height()), (3, 2), "{name}");
        }
    }

    #[test]
    fn decode_image_keeps_the_extension_error_when_the_content_fails_too() {
        let files = Files::new("decode-err");
        for (name, bytes) in [
            ("garbage.png", &b"not an image"[..]),
            ("garbage", b"not an image"),
            ("empty.png", b""),
            ("png-magic.jpg", b"\x89PNG\r\n\x1a\nnot a png"),
            ("bmp-magic.png", b"BM but not a bitmap"),
            // Decodes by content, to an image that is five pixels tall and none wide.
            ("no-pixels.png", b"P6 0 5 255\n"),
        ] {
            let path = files.write(name, bytes);
            assert_eq!(
                decode(&path).unwrap_err().to_string(),
                extension_error(&path),
                "{name}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn decode_image_reads_a_pipe_or_directory_once() {
        let files = Files::new("decode-once");

        let dir = files.0.join("dir.png");
        std::fs::create_dir(&dir).unwrap();
        assert_eq!(decode(&dir).unwrap_err().to_string(), extension_error(&dir));

        let pipe = files.0.join("pipe.png");
        let made = std::process::Command::new("mkfifo")
            .arg(&pipe)
            .status()
            .unwrap();
        assert!(made.success());
        let writer = {
            let pipe = pipe.clone();
            std::thread::spawn(move || std::fs::write(pipe, b"not an image").unwrap())
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || sender.send(decode(&pipe).is_err()));
        let failed = receiver.recv_timeout(std::time::Duration::from_secs(10));
        writer.join().unwrap();
        assert_eq!(failed, Ok(true), "the decode did not return its error");
    }
}

#[cfg(all(test, feature = "onnx"))]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;

    fn minimal_tokenizer_bytes() -> Vec<u8> {
        // Minimal valid tokenizer.json (word-level model with a tiny vocab).
        br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "[PAD]": 1, "hello": 2}
            }
        }"#
        .to_vec()
    }

    fn tokenizer_files(tokenizer_config: &str) -> TokenizerFiles {
        TokenizerFiles {
            tokenizer_file: minimal_tokenizer_bytes(),
            config_file: br#"{"pad_token_id": 0}"#.to_vec(),
            special_tokens_map_file: b"{}".to_vec(),
            tokenizer_config_file: tokenizer_config.as_bytes().to_vec(),
        }
    }

    #[cfg(feature = "hf-hub")]
    #[test]
    fn hf_endpoint_defaults_when_unset_or_empty() {
        assert_eq!(hf_endpoint(None), "https://huggingface.co");
        assert_eq!(hf_endpoint(Some(String::new())), "https://huggingface.co");
        assert_eq!(hf_endpoint(Some("/".into())), "https://huggingface.co");
        for endpoint in ["https://hf-mirror.example", "https://hf-mirror.example/"] {
            assert_eq!(
                hf_endpoint(Some(endpoint.into())),
                "https://hf-mirror.example"
            );
        }
    }

    #[cfg(feature = "hf-hub")]
    #[test]
    fn a_download_in_progress_shows_in_its_partial_file() {
        let blobs = std::env::temp_dir().join(format!("fastembed-blobs-{}", std::process::id()));
        std::fs::create_dir_all(&blobs).unwrap();
        let lock = blobs.join("0123abcd.lock");
        let partial = blobs.join("0123abcd.part");

        let mut size = None;
        // No download has a partial file, so the lock error is final.
        let none = partial_file_changed(&lock, &mut size);
        std::fs::write(&partial, [0; 3]).unwrap();
        let started = partial_file_changed(&lock, &mut size);
        let stalled = partial_file_changed(&lock, &mut size);
        std::fs::write(&partial, [0; 5]).unwrap();
        let grew = partial_file_changed(&lock, &mut size);
        // A finished download renames its partial file.
        std::fs::remove_file(&partial).unwrap();
        let finished = partial_file_changed(&lock, &mut size);
        let gone = partial_file_changed(&lock, &mut size);
        std::fs::remove_dir_all(&blobs).unwrap();

        assert_eq!(
            [none, started, stalled, grew, finished, gone],
            [false, true, false, true, true, false]
        );
    }

    #[cfg(feature = "hf-hub")]
    #[test]
    fn a_load_waits_for_a_download_that_writes_to_disk() {
        let blobs = std::env::temp_dir().join(format!("fastembed-wait-{}", std::process::id()));
        std::fs::create_dir_all(&blobs).unwrap();
        let lock = blobs.join("0123abcd.lock");
        let partial = blobs.join("0123abcd.part");
        // The other download holds the lock for as long as its partial file is there.
        let fetch = || match partial.exists() {
            true => Err(ApiError::LockAcquisition(lock.clone())),
            false => Ok("model.onnx"),
        };
        let locked = || Err::<&str, _>(ApiError::LockAcquisition(lock.clone()));

        // No partial file: nothing shows a download, and the lock error is final.
        let (mut tries, mut pauses) = (0, 0);
        let result = wait_for_download(
            || {
                tries += 1;
                locked()
            },
            || pauses += 1,
        );
        assert!(matches!(result, Err(ApiError::LockAcquisition(_))));
        assert_eq!((tries, pauses), (1, 0));

        // A partial file that does not grow for 60 pauses: one more try, then the lock error.
        std::fs::write(&partial, [0; 3]).unwrap();
        let (mut tries, mut pauses) = (0, 0);
        let result = wait_for_download(
            || {
                tries += 1;
                fetch()
            },
            || pauses += 1,
        );
        assert!(matches!(result, Err(ApiError::LockAcquisition(_))));
        assert_eq!((tries, pauses), (2, 60));

        // A download that grows, writes nothing for 59 pauses, grows again and is then
        // renamed: the next try succeeds.
        let (mut tries, mut pauses) = (0, 0);
        let result = wait_for_download(
            || {
                tries += 1;
                fetch()
            },
            || {
                pauses += 1;
                match pauses {
                    1..=3 | 63 => std::fs::write(&partial, vec![0; 3 + pauses]).unwrap(),
                    4..=62 => {}
                    _ => std::fs::remove_file(&partial).unwrap(),
                }
            },
        );
        std::fs::remove_dir_all(&blobs).unwrap();
        assert_eq!(result.unwrap(), "model.onnx");
        assert_eq!((tries, pauses), (2, 64));
    }

    #[cfg(feature = "hf-hub")]
    #[test]
    fn hf_home_token_reads_the_token_file_in_hf_home() {
        let hf_home =
            std::env::temp_dir().join(format!("fastembed-hf-home-{}", std::process::id()));
        std::fs::create_dir_all(&hf_home).unwrap();
        let without_file = hf_home_token(&hf_home);
        std::fs::write(hf_home.join("token"), " hf_token\n").unwrap();
        let with_file = hf_home_token(&hf_home);
        std::fs::remove_dir_all(&hf_home).unwrap();

        assert_eq!(without_file, None);
        assert_eq!(with_file.as_deref(), Some("hf_token"));
    }

    #[cfg(all(
        feature = "hf-hub",
        any(
            feature = "qwen3-text",
            feature = "nomic-v2-moe",
            feature = "bert-text"
        )
    ))]
    #[test]
    fn shard_files_lists_each_shard_once_and_stays_inside_the_repo() {
        let index = |files: &[&str]| {
            let weight_map: serde_json::Map<String, serde_json::Value> = files
                .iter()
                .enumerate()
                .map(|(i, file)| (format!("weight.{i}"), (*file).into()))
                .collect();
            serde_json::json!({ "weight_map": weight_map })
        };

        let shards = shard_files(
            &index(&[
                "model-2.safetensors",
                "model-1.safetensors",
                "model-2.safetensors",
            ]),
            "index",
        );
        assert_eq!(
            shards.unwrap(),
            ["model-1.safetensors", "model-2.safetensors"]
        );
        assert_eq!(
            shard_files(&index(&["shards/model-1.safetensors"]), "index").unwrap(),
            ["shards/model-1.safetensors"]
        );

        for file in ["/etc/hostname", "../model.safetensors", "shards/../../x"] {
            let err = shard_files(&index(&[file]), "index").unwrap_err();
            assert!(err.to_string().contains(file), "error was: {err}");
        }
        assert!(shard_files(&serde_json::json!({}), "index").is_err());
    }

    #[cfg(all(
        feature = "hf-hub",
        any(
            feature = "qwen3-text",
            feature = "nomic-v2-moe",
            feature = "bert-text"
        )
    ))]
    #[test]
    fn error_with_cause_adds_the_source_of_a_failed_download() {
        let failed = Error::ModelRetrieval {
            file: "model.safetensors".into(),
            source: "status code 401".into(),
        };
        assert_eq!(
            error_with_cause(failed),
            "Failed to retrieve model file 'model.safetensors': status code 401"
        );
        assert_eq!(error_with_cause(Error::Other("plain".into())), "plain");
    }

    #[test]
    fn intra_threads_or_prefers_the_request_and_tolerates_an_unknown_core_count() {
        let cores = || Ok(std::num::NonZeroUsize::new(8).unwrap());
        let unknown = || Err(std::io::Error::other("unknown"));
        assert_eq!(intra_threads_or(Some(2), cores()), Some(2));
        assert_eq!(intra_threads_or(None, cores()), Some(8));
        assert_eq!(intra_threads_or(Some(2), unknown()), Some(2));
        assert_eq!(intra_threads_or(None, unknown()), None);
    }

    #[test]
    fn load_tokenizer_ok_with_complete_config() {
        let files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        assert!(load_tokenizer(files, 512).is_ok());
    }

    #[test]
    fn load_tokenizer_errors_on_missing_pad_token() {
        let files = tokenizer_files(r#"{"model_max_length": 512}"#);
        let err = load_tokenizer(files, 512).unwrap_err();
        assert!(
            err.to_string().contains("pad_token"),
            "error message was: {err}"
        );
    }

    #[test]
    fn load_tokenizer_errors_on_missing_model_max_length() {
        let files = tokenizer_files(r#"{"pad_token": "[PAD]"}"#);
        let err = load_tokenizer(files, 512).unwrap_err();
        assert!(
            err.to_string().contains("model_max_length"),
            "error message was: {err}"
        );
    }
    #[test]
    fn load_tokenizer_accepts_added_token_object_as_pad_token() {
        let files = tokenizer_files(
            r#"{"model_max_length": 512, "pad_token": {"__type": "AddedToken", "content": "[PAD]", "lstrip": false}}"#,
        );
        let tokenizer = load_tokenizer(files, 512).unwrap();
        assert_eq!(tokenizer.get_padding().unwrap().pad_token, "[PAD]");
    }

    #[test]
    fn load_tokenizer_resolves_pad_id_from_vocab_when_config_lacks_it() {
        let mut files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        files.config_file = b"{}".to_vec();
        let tokenizer = load_tokenizer(files, 512).unwrap();
        assert_eq!(tokenizer.get_padding().unwrap().pad_id, 1);
    }

    #[test]
    fn load_tokenizer_names_the_cause_of_an_unreadable_file() {
        let mut files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        files.config_file = b"{".to_vec();
        let err = load_tokenizer(files, 512).unwrap_err();
        assert!(matches!(err, Error::Io(_)), "error was: {err}");
        assert!(
            err.to_string()
                .contains("Could not read config.json file: EOF while parsing"),
            "error message was: {err}"
        );
    }

    #[test]
    fn load_tokenizer_keeps_the_flags_of_a_special_token_in_tokenizer_json() {
        let mut files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&files.tokenizer_file).unwrap();
        tokenizer["normalizer"] = serde_json::json!({"type": "Lowercase"});
        tokenizer["model"]["vocab"]["[SEP]"] = 3.into();
        tokenizer["added_tokens"] = serde_json::json!([{
            "id": 3,
            "content": "[SEP]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }]);
        files.tokenizer_file = serde_json::to_vec(&tokenizer).unwrap();
        files.special_tokens_map_file =
            br#"{"sep_token": "[SEP]", "mask_token": "[MASK]"}"#.to_vec();

        let tokenizer = load_tokenizer(files, 512).unwrap();
        let ids = |text: &str| tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        assert_eq!(ids("hello [SEP]"), [2, 3]);
        // Only the exact string is the special token. The lower-case one is three unknown words.
        assert_eq!(ids("hello [sep]"), [2, 0, 0, 0]);
        // A token that tokenizer.json does not have is still added.
        assert_eq!(ids("hello [MASK]"), [2, 4]);
    }

    /// Tokenizer files whose post-processor adds two special tokens to a text, three to a pair.
    pub(crate) fn bert_tokenizer_files() -> TokenizerFiles {
        let mut files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&files.tokenizer_file).unwrap();
        tokenizer["post_processor"] = serde_json::json!({
            "type": "BertProcessing",
            "sep": ["[PAD]", 1],
            "cls": ["[UNK]", 0]
        });
        files.tokenizer_file = serde_json::to_vec(&tokenizer).unwrap();
        files
    }

    #[test]
    fn load_tokenizer_refuses_a_max_length_below_the_special_tokens() {
        let files = bert_tokenizer_files();

        let err = load_tokenizer(files.clone(), 1).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "error was: {err}");

        let tokenizer = load_tokenizer(files.clone(), 2).unwrap();
        let encoding = tokenizer.encode("hello hello hello", true).unwrap();
        assert_eq!(encoding.get_ids(), [0, 1]);
        assert!(matches!(
            check_pair_truncation(&tokenizer),
            Err(Error::InvalidArgument(_))
        ));

        let tokenizer = load_tokenizer(files, 3).unwrap();
        assert!(check_pair_truncation(&tokenizer).is_ok());
        let encoding = tokenizer.encode(("hello hello", "hello"), true).unwrap();
        assert_eq!(encoding.get_ids(), [0, 1, 1]);
    }

    #[test]
    fn encode_batch_pads_to_longest_and_masks_padding() {
        let files = tokenizer_files(r#"{"model_max_length": 512, "pad_token": "[PAD]"}"#);
        let tokenizer = load_tokenizer(files, 512).unwrap();
        let batch = encode_batch(&tokenizer, vec!["hello", "hello hello"]).unwrap();
        assert_eq!(batch.input_ids.dim(), (2, 2));
        assert_eq!(batch.attention_mask.row(0).to_vec(), vec![1, 0]);
        assert_eq!(batch.attention_mask.row(1).to_vec(), vec![1, 1]);
        assert_eq!(batch.input_ids[[0, 1]], 0);
        assert_eq!(batch.input_ids.row(1).to_vec(), vec![2, 2]);
        assert!(encode_batch::<&str>(&tokenizer, vec![]).is_err());
    }

    #[test]
    fn init_session_builder_applies_config_entry() {
        let builder = init_session_builder(
            None,
            vec![],
            Some(1),
            vec![("session.disable_prepacking".into(), "1".into())],
            None,
        );
        assert!(builder.is_ok());

        // ONNX Runtime refuses an empty key, so an error shows that the entry reached it.
        let refused = init_session_builder(
            None,
            vec![],
            Some(1),
            vec![(String::new(), "1".into())],
            None,
        );
        assert!(matches!(refused, Err(Error::OrtBuilder(_))));
    }

    /// Encodes a protobuf field tag followed by a length-delimited payload.
    fn protobuf_bytes(field: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![(field << 3) | 2];
        let mut len = payload.len();
        while len >= 0x80 {
            out.push((len as u8 & 0x7f) | 0x80);
            len >>= 7;
        }
        out.push(len as u8);
        out.extend_from_slice(payload);
        out
    }

    const FLOAT: u8 = 1;
    const INT64: u8 = 7;

    /// A graph input or output: a tensor of `elem_type`, each dimension a size or `None` for
    /// any size.
    fn value_info(name: &str, elem_type: u8, dims: &[Option<u8>]) -> Vec<u8> {
        let dims: Vec<u8> = dims
            .iter()
            .flat_map(|dim| match dim {
                Some(size) => protobuf_bytes(1, &[0x08, *size]),
                None => protobuf_bytes(1, &protobuf_bytes(2, b"any")),
            })
            .collect();
        let tensor_type = protobuf_bytes(
            1,
            &[vec![0x08, elem_type], protobuf_bytes(2, &dims)].concat(),
        );
        [
            protobuf_bytes(1, name.as_bytes()),
            protobuf_bytes(2, &tensor_type),
        ]
        .concat()
    }

    /// An ONNX model of `graph`, with opset 17.
    fn model_bytes(graph: &[u8]) -> Vec<u8> {
        [
            vec![0x08, 0x08],
            protobuf_bytes(7, graph),
            protobuf_bytes(8, &[0x10, 0x11]),
        ]
        .concat()
    }

    /// A one-node ONNX model, `y = Relu(x)` over a float tensor of shape [1, 4].
    pub(crate) fn relu_model_bytes() -> Vec<u8> {
        let node = [
            protobuf_bytes(1, b"x"),
            protobuf_bytes(2, b"y"),
            protobuf_bytes(4, b"Relu"),
        ]
        .concat();
        let shape = [Some(1), Some(4)];
        model_bytes(
            &[
                protobuf_bytes(1, &node),
                protobuf_bytes(2, b"relu"),
                protobuf_bytes(11, &value_info("x", FLOAT, &shape)),
                protobuf_bytes(12, &value_info("y", FLOAT, &shape)),
            ]
            .concat(),
        )
    }

    /// A one-node ONNX text model, `y = Cast(input_ids)` to float: each text embeds to its
    /// own token ids. With `tokens`, the model takes only rows of that many tokens.
    pub(crate) fn token_ids_model_bytes(tokens: Option<u8>) -> Vec<u8> {
        // The attribute `to = FLOAT`.
        let to_float = [
            protobuf_bytes(1, b"to"),
            vec![0x18, FLOAT, 0xa0, 0x01, 0x02],
        ]
        .concat();
        let node = [
            protobuf_bytes(1, b"input_ids"),
            protobuf_bytes(2, b"y"),
            protobuf_bytes(4, b"Cast"),
            protobuf_bytes(5, &to_float),
        ]
        .concat();
        let shape = [None, tokens];
        model_bytes(
            &[
                protobuf_bytes(1, &node),
                protobuf_bytes(2, b"token-ids"),
                protobuf_bytes(11, &value_info("input_ids", INT64, &shape)),
                protobuf_bytes(11, &value_info("attention_mask", INT64, &shape)),
                protobuf_bytes(12, &value_info("y", FLOAT, &shape)),
            ]
            .concat(),
        )
    }

    #[test]
    fn init_session_builder_applies_auto_device_policy() {
        let messages = Arc::new(Mutex::new(Vec::<String>::new()));
        let captured = Arc::clone(&messages);
        let logger: ort::logging::LoggerFunction = Arc::new(move |_, _, _, _, message: &str| {
            if message.contains("placed on") {
                captured
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(message.to_owned());
            }
        });
        init_session_builder(
            None,
            vec![],
            Some(1),
            vec![],
            Some(AutoDevicePolicy::PreferCPU),
        )
        .unwrap()
        .with_logger(logger)
        .unwrap()
        .with_log_level(ort::logging::LogLevel::Verbose)
        .unwrap()
        .commit_from_memory(&relu_model_bytes())
        .unwrap();

        // Without the policy, ort's default `MaxEfficiency` places the node on CoreML on macOS.
        let messages = messages.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(!messages.is_empty(), "no node placement was logged");
        assert!(
            messages
                .iter()
                .all(|message| message.contains("[CPUExecutionProvider]")),
            "nodes were placed off the CPU: {messages:?}"
        );
    }

    #[test]
    fn init_session_builder_shares_one_default_environment() {
        let first = init_session_builder(None, vec![], Some(1), vec![], None);
        let second = init_session_builder(None, vec![], Some(1), vec![], None);
        assert!(first.is_ok() && second.is_ok());
    }
}
