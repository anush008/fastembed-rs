use std::{collections::HashMap, convert::TryFrom, fmt::Display, str::FromStr, sync::OnceLock};

use super::{model_info::ModelInfo, ModelTrait};

/// Lazy static list of all available models.
static MODEL_MAP: OnceLock<HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>>> = OnceLock::new();

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    /// sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2,
    /// Quantized sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2Q,
    /// sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2,
    /// Quantized sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2Q,
    /// sentence-transformers/all-mpnet-base-v2
    AllMpnetBaseV2,
    /// BAAI/bge-base-en-v1.5
    BGEBaseENV15,
    /// Quantized BAAI/bge-base-en-v1.5
    BGEBaseENV15Q,
    /// BAAI/bge-large-en-v1.5
    BGELargeENV15,
    /// Quantized BAAI/bge-large-en-v1.5
    BGELargeENV15Q,
    /// BAAI/bge-small-en-v1.5 - Default
    #[default]
    BGESmallENV15,
    /// Quantized BAAI/bge-small-en-v1.5
    BGESmallENV15Q,
    /// nomic-ai/nomic-embed-text-v1
    NomicEmbedTextV1,
    /// nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15,
    /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15Q,
    /// sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    ParaphraseMLMiniLML12V2,
    /// Quantized sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    ParaphraseMLMiniLML12V2Q,
    /// sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    ParaphraseMLMpnetBaseV2,
    /// BAAI/bge-small-zh-v1.5
    BGESmallZHV15,
    /// BAAI/bge-large-zh-v1.5
    BGELargeZHV15,
    /// BAAI/bge-m3
    BGEM3,
    /// lightonai/modernbert-embed-large
    ModernBertEmbedLarge,
    /// intfloat/multilingual-e5-small
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large
    MultilingualE5Large,
    /// mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1,
    /// Quantized mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1Q,
    /// Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15,
    /// Quantized Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15Q,
    /// Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15,
    /// Quantized Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15Q,
    /// Qdrant/clip-ViT-B-32-text
    ClipVitB32,
    /// jinaai/jina-embeddings-v2-base-code
    JinaEmbeddingsV2BaseCode,
    /// jinaai/jina-embeddings-v2-base-en
    JinaEmbeddingsV2BaseEN,
    /// onnx-community/embeddinggemma-300m-ONNX
    EmbeddingGemma300M,
    /// Quantized (4-bit) onnx-community/embeddinggemma-300m-ONNX
    EmbeddingGemma300MQ4,
    /// Quantized onnx-community/embeddinggemma-300m-ONNX
    EmbeddingGemma300MQ,
    /// snowflake/snowflake-arctic-embed-xs
    SnowflakeArcticEmbedXS,
    /// Quantized snowflake/snowflake-arctic-embed-xs
    SnowflakeArcticEmbedXSQ,
    /// snowflake/snowflake-arctic-embed-s
    SnowflakeArcticEmbedS,
    /// Quantized snowflake/snowflake-arctic-embed-s
    SnowflakeArcticEmbedSQ,
    /// snowflake/snowflake-arctic-embed-m
    SnowflakeArcticEmbedM,
    /// Quantized snowflake/snowflake-arctic-embed-m
    SnowflakeArcticEmbedMQ,
    /// snowflake/snowflake-arctic-embed-m-long
    SnowflakeArcticEmbedMLong,
    /// Quantized snowflake/snowflake-arctic-embed-m-long
    SnowflakeArcticEmbedMLongQ,
    /// snowflake/snowflake-arctic-embed-l
    SnowflakeArcticEmbedL,
    /// Quantized snowflake/snowflake-arctic-embed-l
    SnowflakeArcticEmbedLQ,
}

/// Centralized function to initialize the models map.
fn init_models_map() -> HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>> {
    let models_list = vec![
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
            model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L6-v2"),
            model_code: String::from("Xenova/all-MiniLM-L6-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML12V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::AllMpnetBaseV2,
            dim: 768,
            description: String::from("Sentence Transformer model, mpnet-base-v2"),
            model_code: String::from("Xenova/all-mpnet-base-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15,
            dim: 768,
            description: String::from("v1.5 release of the base English model"),
            model_code: String::from("Xenova/bge-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized v1.5 release of the base English model"),
            model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15,
            dim: 1024,
            description: String::from("v1.5 release of the large English model"),
            model_code: String::from("Xenova/bge-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized v1.5 release of the large English model"),
            model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15,
            dim: 384,
            description: String::from("v1.5 release of the fast and default English model"),
            model_code: String::from("Xenova/bge-small-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15Q,
            dim: 384,
            description: String::from(
                "Quantized v1.5 release of the fast and default English model",
            ),
            model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV1,
            dim: 768,
            description: String::from("8192 context length english model"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15,
            dim: 768,
            description: String::from("v1.5 release of the 8192 context length english model"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15Q,
            dim: 768,
            description: String::from(
                "Quantized v1.5 release of the 8192 context length english model",
            ),
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Multi-lingual model"),
            model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2,
            dim: 384,
            description: String::from("Multi-lingual model"),
            model_code: String::from("Xenova/paraphrase-multilingual-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMpnetBaseV2,
            dim: 768,
            description: String::from(
                "Sentence-transformers model for tasks like clustering or semantic search",
            ),
            model_code: String::from("Xenova/paraphrase-multilingual-mpnet-base-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallZHV15,
            dim: 512,
            description: String::from("v1.5 release of the small Chinese model"),
            model_code: String::from("Xenova/bge-small-zh-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeZHV15,
            dim: 1024,
            description: String::from("v1.5 release of the large Chinese model"),
            model_code: String::from("Xenova/bge-large-zh-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::BGEM3,
            dim: 1024,
            description: String::from(
                "Multilingual M3 model with 8192 context length, supports 100+ languages",
            ),
            model_code: String::from("BAAI/bge-m3"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![
                "onnx/model.onnx_data".to_string(),
                "onnx/Constant_7_attr__value".to_string(),
            ],
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::ModernBertEmbedLarge,
            dim: 1024,
            description: String::from("Large model of ModernBert Text Embeddings"),
            model_code: String::from("lightonai/modernbert-embed-large"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Small,
            dim: 384,
            description: String::from("Small model of multilingual E5 Text Embeddings"),
            model_code: String::from("intfloat/multilingual-e5-small"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Base,
            dim: 768,
            description: String::from("Base model of multilingual E5 Text Embeddings"),
            model_code: String::from("intfloat/multilingual-e5-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Large,
            dim: 1024,
            description: String::from("Large model of multilingual E5 Text Embeddings"),
            model_code: String::from("Qdrant/multilingual-e5-large-onnx"),
            model_file: String::from("model.onnx"),
            additional_files: vec!["model.onnx_data".to_string()],
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1,
            dim: 1024,
            description: String::from("Large English embedding model from MixedBreed.ai"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1Q,
            dim: 1024,
            description: String::from("Quantized Large English embedding model from MixedBreed.ai"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::GTEBaseENV15,
            dim: 768,
            description: String::from("Base English embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::GTEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized base English embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::GTELargeENV15,
            dim: 1024,
            description: String::from("Large English embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::GTELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized large English embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::ClipVitB32,
            dim: 512,
            description: String::from("CLIP text encoder based on ViT-B/32"),
            model_code: String::from("Qdrant/clip-ViT-B-32-text"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
            dim: 768,
            description: String::from("Jina embeddings v2 base code"),
            model_code: String::from("jinaai/jina-embeddings-v2-base-code"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::JinaEmbeddingsV2BaseEN,
            dim: 768,
            description: String::from("Jina embeddings v2 base English"),
            model_code: String::from("jinaai/jina-embeddings-v2-base-en"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::EmbeddingGemma300M,
            dim: 768,
            description: String::from("EmbeddingGemma is a 300M parameter from Google"),
            model_code: String::from("onnx-community/embeddinggemma-300m-ONNX"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec!["onnx/model.onnx_data".to_string()],
            output_key: Some(crate::OutputKey::ByName("sentence_embedding")),
        },
        ModelInfo {
            model: EmbeddingModel::EmbeddingGemma300MQ4,
            dim: 768,
            description: String::from(
                "Quantized (4-bit) EmbeddingGemma is a 300M parameter from Google",
            ),
            model_code: String::from("onnx-community/embeddinggemma-300m-ONNX"),
            model_file: String::from("onnx/model_q4.onnx"),
            additional_files: vec!["onnx/model_q4.onnx_data".to_string()],
            output_key: Some(crate::OutputKey::ByName("sentence_embedding")),
        },
        ModelInfo {
            model: EmbeddingModel::EmbeddingGemma300MQ,
            dim: 768,
            description: String::from("Quantized EmbeddingGemma is a 300M parameter from Google"),
            model_code: String::from("onnx-community/embeddinggemma-300m-ONNX"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec!["onnx/model_quantized.onnx_data".to_string()],
            output_key: Some(crate::OutputKey::ByName("sentence_embedding")),
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedXS,
            dim: 384,
            description: String::from("Snowflake Arctic embed model, xs"),
            model_code: String::from("snowflake/snowflake-arctic-embed-xs"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedXSQ,
            dim: 384,
            description: String::from("Quantized Snowflake Arctic embed model, xs"),
            model_code: String::from("snowflake/snowflake-arctic-embed-xs"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedS,
            dim: 384,
            description: String::from("Snowflake Arctic embed model, small"),
            model_code: String::from("snowflake/snowflake-arctic-embed-s"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedSQ,
            dim: 384,
            description: String::from("Quantized Snowflake Arctic embed model, small"),
            model_code: String::from("snowflake/snowflake-arctic-embed-s"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedM,
            dim: 768,
            description: String::from("Snowflake Arctic embed model, medium"),
            model_code: String::from("Snowflake/snowflake-arctic-embed-m"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedMQ,
            dim: 768,
            description: String::from("Quantized Snowflake Arctic embed model, medium"),
            model_code: String::from("Snowflake/snowflake-arctic-embed-m"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedMLong,
            dim: 768,
            description: String::from("Snowflake Arctic embed model, medium with 2048 context"),
            model_code: String::from("snowflake/snowflake-arctic-embed-m-long"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedMLongQ,
            dim: 768,
            description: String::from(
                "Quantized Snowflake Arctic embed model, medium with 2048 context",
            ),
            model_code: String::from("snowflake/snowflake-arctic-embed-m-long"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedL,
            dim: 1024,
            description: String::from("Snowflake Arctic embed model, large"),
            model_code: String::from("snowflake/snowflake-arctic-embed-l"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: EmbeddingModel::SnowflakeArcticEmbedLQ,
            dim: 1024,
            description: String::from("Quantized Snowflake Arctic embed model, large"),
            model_code: String::from("snowflake/snowflake-arctic-embed-l"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
    ];

    models_list
        .into_iter()
        .fold(HashMap::new(), |mut map, model| {
            // Insert the model into the map
            map.insert(model.model.clone(), model);
            map
        })
}

/// Get a map of all available models.
pub fn models_map() -> &'static HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>> {
    MODEL_MAP.get_or_init(init_models_map)
}

/// Get a list of all available models.
///
/// This will assign new memory to the models list. Where possible, use
/// [`models_map`] instead.
pub fn models_list() -> Vec<ModelInfo<EmbeddingModel>> {
    models_map().values().cloned().collect()
}

impl ModelTrait for EmbeddingModel {
    type Model = Self;

    /// Get model information by model code.
    fn get_model_info(model: &EmbeddingModel) -> Option<&ModelInfo<EmbeddingModel>> {
        models_map().get(model)
    }
}

impl Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for EmbeddingModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_map()
            .keys()
            .find(|m| format!("{:?}", m).eq_ignore_ascii_case(s))
            .cloned()
            .ok_or_else(|| format!("Unknown embedding model: {s}"))
    }
}

impl TryFrom<String> for EmbeddingModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
pub(crate) fn all_variants() -> Vec<EmbeddingModel> {
    fn _exhaustive_guard(m: &EmbeddingModel) {
        match m {
            EmbeddingModel::AllMiniLML6V2 => (),
            EmbeddingModel::AllMiniLML6V2Q => (),
            EmbeddingModel::AllMiniLML12V2 => (),
            EmbeddingModel::AllMiniLML12V2Q => (),
            EmbeddingModel::AllMpnetBaseV2 => (),
            EmbeddingModel::BGEBaseENV15 => (),
            EmbeddingModel::BGEBaseENV15Q => (),
            EmbeddingModel::BGELargeENV15 => (),
            EmbeddingModel::BGELargeENV15Q => (),
            EmbeddingModel::BGESmallENV15 => (),
            EmbeddingModel::BGESmallENV15Q => (),
            EmbeddingModel::NomicEmbedTextV1 => (),
            EmbeddingModel::NomicEmbedTextV15 => (),
            EmbeddingModel::NomicEmbedTextV15Q => (),
            EmbeddingModel::ParaphraseMLMiniLML12V2 => (),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => (),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => (),
            EmbeddingModel::BGESmallZHV15 => (),
            EmbeddingModel::BGELargeZHV15 => (),
            EmbeddingModel::BGEM3 => (),
            EmbeddingModel::ModernBertEmbedLarge => (),
            EmbeddingModel::MultilingualE5Small => (),
            EmbeddingModel::MultilingualE5Base => (),
            EmbeddingModel::MultilingualE5Large => (),
            EmbeddingModel::MxbaiEmbedLargeV1 => (),
            EmbeddingModel::MxbaiEmbedLargeV1Q => (),
            EmbeddingModel::GTEBaseENV15 => (),
            EmbeddingModel::GTEBaseENV15Q => (),
            EmbeddingModel::GTELargeENV15 => (),
            EmbeddingModel::GTELargeENV15Q => (),
            EmbeddingModel::ClipVitB32 => (),
            EmbeddingModel::JinaEmbeddingsV2BaseCode => (),
            EmbeddingModel::JinaEmbeddingsV2BaseEN => (),
            EmbeddingModel::EmbeddingGemma300M => (),
            EmbeddingModel::EmbeddingGemma300MQ4 => (),
            EmbeddingModel::EmbeddingGemma300MQ => (),
            EmbeddingModel::SnowflakeArcticEmbedXS => (),
            EmbeddingModel::SnowflakeArcticEmbedXSQ => (),
            EmbeddingModel::SnowflakeArcticEmbedS => (),
            EmbeddingModel::SnowflakeArcticEmbedSQ => (),
            EmbeddingModel::SnowflakeArcticEmbedM => (),
            EmbeddingModel::SnowflakeArcticEmbedMQ => (),
            EmbeddingModel::SnowflakeArcticEmbedMLong => (),
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => (),
            EmbeddingModel::SnowflakeArcticEmbedL => (),
            EmbeddingModel::SnowflakeArcticEmbedLQ => (),
        }
    }
    vec![
        EmbeddingModel::AllMiniLML6V2,
        EmbeddingModel::AllMiniLML6V2Q,
        EmbeddingModel::AllMiniLML12V2,
        EmbeddingModel::AllMiniLML12V2Q,
        EmbeddingModel::AllMpnetBaseV2,
        EmbeddingModel::BGEBaseENV15,
        EmbeddingModel::BGEBaseENV15Q,
        EmbeddingModel::BGELargeENV15,
        EmbeddingModel::BGELargeENV15Q,
        EmbeddingModel::BGESmallENV15,
        EmbeddingModel::BGESmallENV15Q,
        EmbeddingModel::NomicEmbedTextV1,
        EmbeddingModel::NomicEmbedTextV15,
        EmbeddingModel::NomicEmbedTextV15Q,
        EmbeddingModel::ParaphraseMLMiniLML12V2,
        EmbeddingModel::ParaphraseMLMiniLML12V2Q,
        EmbeddingModel::ParaphraseMLMpnetBaseV2,
        EmbeddingModel::BGESmallZHV15,
        EmbeddingModel::BGELargeZHV15,
        EmbeddingModel::BGEM3,
        EmbeddingModel::ModernBertEmbedLarge,
        EmbeddingModel::MultilingualE5Small,
        EmbeddingModel::MultilingualE5Base,
        EmbeddingModel::MultilingualE5Large,
        EmbeddingModel::MxbaiEmbedLargeV1,
        EmbeddingModel::MxbaiEmbedLargeV1Q,
        EmbeddingModel::GTEBaseENV15,
        EmbeddingModel::GTEBaseENV15Q,
        EmbeddingModel::GTELargeENV15,
        EmbeddingModel::GTELargeENV15Q,
        EmbeddingModel::ClipVitB32,
        EmbeddingModel::JinaEmbeddingsV2BaseCode,
        EmbeddingModel::JinaEmbeddingsV2BaseEN,
        EmbeddingModel::EmbeddingGemma300M,
        EmbeddingModel::EmbeddingGemma300MQ4,
        EmbeddingModel::EmbeddingGemma300MQ,
        EmbeddingModel::SnowflakeArcticEmbedXS,
        EmbeddingModel::SnowflakeArcticEmbedXSQ,
        EmbeddingModel::SnowflakeArcticEmbedS,
        EmbeddingModel::SnowflakeArcticEmbedSQ,
        EmbeddingModel::SnowflakeArcticEmbedM,
        EmbeddingModel::SnowflakeArcticEmbedMQ,
        EmbeddingModel::SnowflakeArcticEmbedMLong,
        EmbeddingModel::SnowflakeArcticEmbedMLongQ,
        EmbeddingModel::SnowflakeArcticEmbedL,
        EmbeddingModel::SnowflakeArcticEmbedLQ,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_model_info() {
        let listed: Vec<_> = models_list().into_iter().map(|i| i.model).collect();
        for variant in all_variants() {
            assert!(
                listed.contains(&variant),
                "{variant:?} is missing from models_list(); get_model_info would panic"
            );
        }
    }
}
