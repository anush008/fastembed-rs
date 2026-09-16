#![cfg_attr(docsrs, feature(doc_cfg))]
//! [FastEmbed](https://github.com/Anush008/fastembed-rs) - Fast, light, accurate library built for retrieval embedding generation.
//!
//! Local ONNX inference, synchronous, no Tokio. Models download once and run offline thereafter.
//!
//! # What's here
//!
//! - [`TextEmbedding`] - dense text embeddings (default: BGE small en v1.5)
//! - [`SparseTextEmbedding`] - sparse (SPLADE) embeddings for lexical search
//! - [`Bgem3Embedding`] - dense + sparse + ColBERT in a single pass (BGE-M3)
//! - [`ImageEmbedding`] - image embeddings (CLIP, ResNet, ...)
//! - [`TextRerank`] - cross-encoder reranking of candidates
//!
//! Qwen3 and Nomic v2 MoE embeddings are available behind the `qwen3` and
//! `nomic-v2-moe` feature flags (candle backend).
//!
//! # Model cache
//!
//! Models download to `./.fastembed_cache` on first use, then load from there.
//! Override the location with the `FASTEMBED_CACHE_DIR` env var or
//! [`TextInitOptions`]`::with_cache_dir`. `HF_HOME` takes precedence over both.
//! Set `HF_ENDPOINT` to pull from a mirror.
//!
#![cfg_attr(
    feature = "hf-hub",
    doc = r#"
 ### Instantiating [TextEmbedding](crate::TextEmbedding)
 ```
 use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};

# fn model_demo() -> fastembed::Result<()> {
 // With default TextInitOptions
 let model = TextEmbedding::try_new(Default::default())?;

 // List all supported models
 dbg!(TextEmbedding::list_supported_models());

 // With custom TextInitOptions
 let model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
 )?;
 # Ok(())
 # }
 ```
"#
)]
//! Find more info about the available options in the [TextInitOptions](crate::TextInitOptions) documentation.
//!
#![cfg_attr(
    feature = "hf-hub",
    doc = r#"
 ### Embeddings generation
```
# use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};
# fn embedding_demo() -> fastembed::Result<()> {
# let mut model: TextEmbedding = TextEmbedding::try_new(Default::default())?;
 let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-rs is licensed under Apache 2.0"
    ];

 // Generate embeddings with the default batch size, 256
 let embeddings = model.embed(documents, None)?;

 println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
 # Ok(())
 # }
 ```
"#
)]

mod common;

mod bgem3_embedding;
#[cfg(feature = "image-models")]
mod image_embedding;
mod init;
mod models;
pub mod output;
mod pooling;
mod reranking;
pub mod similarity;
mod sparse_text_embedding;
mod text_embedding;

pub use ort::execution_providers::ExecutionProviderDispatch;

pub use crate::common::{get_cache_dir, Embedding, Error, Result, SparseEmbedding, TokenizerFiles};
pub use crate::models::{
    model_info::ModelInfo, model_info::RerankerModelInfo, quantization::QuantizationMode,
};
pub use crate::output::{EmbeddingOutput, OutputKey, OutputPrecedence, SingleBatchOutput};
pub use crate::pooling::Pooling;

// For all Embedding
pub use crate::init::{InitOptions as BaseInitOptions, InitOptionsWithLength};
pub use crate::models::ModelTrait;

// For Text Embedding
pub use crate::models::text_embedding::EmbeddingModel;
pub use crate::text_embedding::{
    InitOptionsUserDefined, TextEmbedding, TextInitOptions, UserDefinedEmbeddingModel,
};
#[deprecated(note = "use `TextInitOptions` instead")]
pub type InitOptions = TextInitOptions;

// For Sparse Text Embedding
pub use crate::models::sparse::SparseModel;
pub use crate::sparse_text_embedding::{
    SparseInitOptions, SparseTextEmbedding, UserDefinedSparseModel,
};

// For BGEM3 Joint Embedding
pub use crate::bgem3_embedding::{
    Bgem3Embedding, Bgem3EmbeddingOutput, Bgem3InitOptions, UserDefinedBgem3Model,
};
pub use crate::models::bgem3::Bgem3Model;

// For Image Embedding
#[cfg(feature = "image-models")]
pub use crate::image_embedding::{
    ImageEmbedding, ImageInitOptions, ImageInitOptionsUserDefined, UserDefinedImageEmbeddingModel,
};
pub use crate::models::image_embedding::ImageEmbeddingModel;

// For Reranking
pub use crate::models::reranking::RerankerModel;
pub use crate::reranking::{
    OnnxSource, RerankInitOptions, RerankInitOptionsUserDefined, RerankResult, TextRerank,
    UserDefinedRerankingModel,
};

// For Qwen3 (candle backend)
#[cfg(feature = "qwen3")]
pub use crate::models::qwen3::{
    Config as Qwen3Config, Qwen3Model, Qwen3TextEmbedding, Qwen3VLEmbedding,
};

// For Nomic Embed Text v2 MoE (candle backend)
#[cfg(feature = "nomic-v2-moe")]
pub use crate::models::nomic_v2_moe::{NomicBertModel, NomicConfig, NomicV2MoeTextEmbedding};
