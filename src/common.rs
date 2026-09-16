#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use ndarray::Array2;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        SessionInputValue,
    },
    value::Value,
};
use std::borrow::Cow;
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::{
    AddedToken, EncodeInput, Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
};

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

/// The procedure for loading tokenizer files from the hugging face hub is separated
/// from the main load_tokenizer function (which is expecting bytes, from any source).
#[cfg(feature = "hf-hub")]
pub fn load_tokenizer_hf_hub(model_repo: ApiRepo, max_length: usize) -> Result<Tokenizer> {
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json")?)?,
        config_file: std::fs::read(&model_repo.get("config.json")?)?,
        special_tokens_map_file: std::fs::read(&model_repo.get("special_tokens_map.json")?)?,

        tokenizer_config_file: std::fs::read(&model_repo.get("tokenizer_config.json")?)?,
    };

    load_tokenizer(tokenizer_files, max_length)
}

/// Function can be called directly from the try_new_from_user_defined function (providing file bytes)
///
/// Or indirectly from the try_new function via load_tokenizer_hf_hub (converting HF files to bytes)
pub fn load_tokenizer(tokenizer_files: TokenizerFiles, max_length: usize) -> Result<Tokenizer> {
    let base_error_message =
        "Error building TokenizerFiles for UserDefinedEmbeddingModel. Could not read {} file.";

    // Deserialize each tokenizer file
    let config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "config.json"),
            )
        })?;
    let special_tokens_map: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.special_tokens_map_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "special_tokens_map.json"),
            )
        })?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.tokenizer_config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer_config.json"),
            )
        })?;
    let mut tokenizer: tokenizers::Tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_files.tokenizer_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer.json"),
            )
        })?;

    //For BGEBaseSmall, the model_max_length value is set to 1000000000000000019884624838656. Which fits in a f64
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

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            // TODO: the user should be able to choose the padding strategy
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| Error::TokenizerConfig(e.to_string()))?
        .clone();
    if let serde_json::Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if value.is_string() {
                if let Some(content) = value.as_str() {
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
#[cfg(feature = "hf-hub")]
pub fn pull_from_hf(
    model_name: String,
    default_cache_dir: PathBuf,
    show_download_progress: bool,
) -> Result<ApiRepo> {
    use std::env;

    let cache_dir = env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or(default_cache_dir);

    let endpoint = env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());

    let token = env::var("HF_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .or_else(default_hf_token);

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .with_endpoint(endpoint)
        .with_token(token)
        .with_progress(show_download_progress)
        .build()
        .map_err(|e| Error::Other(format!("Failed to initialize HuggingFace API: {e}")))?;

    let repo = api.model(model_name);
    Ok(repo)
}

/// Token written by `huggingface-cli login`. `Cache::default` panics without a home dir.
#[cfg(feature = "hf-hub")]
fn default_hf_token() -> Option<String> {
    let has_home = std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some();
    has_home.then(|| hf_hub::Cache::default().token()).flatten()
}

/// Safetensors files of a repo, resolving `model.safetensors.index.json` for sharded checkpoints.
#[cfg(any(feature = "qwen3", feature = "nomic-v2-moe"))]
pub(crate) fn safetensors_weight_files(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    const SINGLE_FILE: &str = "model.safetensors";
    const INDEX_FILE: &str = "model.safetensors.index.json";

    if let Ok(path) = repo.get(SINGLE_FILE) {
        return Ok(vec![path]);
    }

    let index_path = repo.get(INDEX_FILE).map_err(|e| Error::ModelRetrieval {
        file: format!("{SINGLE_FILE} or {INDEX_FILE}"),
        source: Box::new(e),
    })?;
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_path)?)
        .map_err(|e| Error::Other(format!("Failed to parse {INDEX_FILE}: {e}")))?;

    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Other(format!("{INDEX_FILE} has no `weight_map` object")))?
        .values()
        .filter_map(|file| file.as_str().map(str::to_string))
        .collect();
    shards.sort_unstable();
    shards.dedup();

    shards
        .iter()
        .map(|file| {
            repo.get(file).map_err(|e| Error::ModelRetrieval {
                file: file.clone(),
                source: Box::new(e),
            })
        })
        .collect()
}

/// One tokenized batch as `[batch, sequence]` tensors.
pub(crate) struct EncodedBatch {
    pub input_ids: Array2<i64>,
    pub attention_mask: Array2<i64>,
    pub token_type_ids: Array2<i64>,
}

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
pub(crate) fn encode_batch<'s, E>(tokenizer: &Tokenizer, inputs: Vec<E>) -> Result<EncodedBatch>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| Error::Tokenization(format!("Failed to encode the batch: {e}")))?;
    encodings_to_batch(&encodings)
}

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

pub(crate) fn init_session_builder(
    execution_providers: Vec<ExecutionProviderDispatch>,
    intra_threads: Option<usize>,
    session_config: Vec<(String, String)>,
) -> Result<SessionBuilder> {
    let threads = match intra_threads {
        Some(n) => n,
        None => std::thread::available_parallelism()?.get(),
    };

    #[cfg(feature = "directml")]
    let has_directml = execution_providers
        .iter()
        .any(|ep| ep.downcast_ref::<ort::ep::DirectML>().is_some());
    #[cfg(not(feature = "directml"))]
    let has_directml = false;

    let builder_error = |err: ort::Error<SessionBuilder>| Error::OrtBuilder(err.to_string());

    let mut builder = ort::session::Session::builder()?
        .with_execution_providers(execution_providers)
        .map_err(builder_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(builder_error)?
        .with_intra_threads(threads)
        .map_err(builder_error)?;

    for (key, value) in session_config {
        builder = builder
            .with_config_entry(&key, &value)
            .map_err(builder_error)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
            vec![],
            Some(1),
            vec![("session.disable_prepacking".into(), "1".into())],
        );
        assert!(builder.is_ok());
    }
}
