#![cfg_attr(docsrs, feature(doc_cfg))]
//! [FastEmbed](https://github.com/Anush008/fastembed-rs) - Library for generating vector embeddings and reranking locally.
//!
//! # What's here
//!
//! - [`TextEmbedding`] - dense text embeddings (default: BGE small en v1.5)
//! - [`SparseTextEmbedding`] - sparse (SPLADE) embeddings for lexical search
//! - [`Bgem3Embedding`] - dense + sparse + ColBERT in a single pass (BGE-M3)
//! - [`ImageEmbedding`] - image embeddings (CLIP, ResNet, ...)
//! - [`TextRerank`] - cross-encoder reranking of candidates
//!
//! These run on ONNX Runtime and need the `onnx` feature, which is on by default
//! and which every `ort-*` feature turns on.
//!
//! Qwen3 and Nomic v2 MoE embeddings are available behind the `qwen3` and
//! `nomic-v2-moe` feature flags (candle backend). A build with
//! `default-features = false` and only those features compiles no ONNX Runtime.
//! `qwen3` is `qwen3-text` (the text embedding model and the reranker, loaded from
//! local files or GGUF) plus `qwen3-vl` (the Qwen3-VL image and text model) plus
//! `hf-hub` (the `from_hf` loaders); `qwen3-text` alone pulls neither hf-hub nor
//! the image crates.
//!
//! # Model cache
//!
//! Models download to `./.fastembed_cache` on first use, then load from there.
//! Override the location with the `FASTEMBED_CACHE_DIR` env var or
//! [`TextInitOptions`]`::with_cache_dir`. `HF_HOME` takes precedence over both.
//! Set `HF_ENDPOINT` to pull from a mirror.
//!
// Without `onnx` the ONNX types above are not in the crate, so their links point at docs.rs.
#![cfg_attr(
    not(feature = "onnx"),
    doc = r#"
[`TextEmbedding`]: https://docs.rs/fastembed/latest/fastembed/?search=TextEmbedding
[`SparseTextEmbedding`]: https://docs.rs/fastembed/latest/fastembed/?search=SparseTextEmbedding
[`Bgem3Embedding`]: https://docs.rs/fastembed/latest/fastembed/?search=Bgem3Embedding
[`ImageEmbedding`]: https://docs.rs/fastembed/latest/fastembed/?search=ImageEmbedding
[`TextRerank`]: https://docs.rs/fastembed/latest/fastembed/?search=TextRerank
[`TextInitOptions`]: https://docs.rs/fastembed/latest/fastembed/?search=TextInitOptions
"#
)]
#![cfg_attr(
    all(feature = "onnx", feature = "hf-hub"),
    doc = r#"
 ### Instantiating [TextEmbedding]
 ```
 use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};

# fn model_demo() -> fastembed::Result<()> {
 // With default TextInitOptions
 let model = TextEmbedding::try_new(Default::default())?;

 // List all supported models
 dbg!(TextEmbedding::supported_models());

 // With custom TextInitOptions
 let model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
 )?;
 # Ok(())
 # }
 ```
"#
)]
#![cfg_attr(
    feature = "onnx",
    doc = "Find more info about the available options in the [TextInitOptions] documentation.\n"
)]
//!
#![cfg_attr(
    all(feature = "onnx", feature = "hf-hub"),
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

#[cfg(feature = "onnx")]
mod bgem3_embedding;
#[cfg(feature = "image-models")]
mod image_embedding;
#[cfg(feature = "onnx")]
mod init;
mod models;
#[cfg(feature = "onnx")]
pub mod output;
#[cfg(feature = "onnx")]
mod pooling;
#[cfg(feature = "onnx")]
mod reranking;
pub mod similarity;
#[cfg(feature = "onnx")]
mod sparse_text_embedding;
#[cfg(feature = "onnx")]
mod text_embedding;

#[cfg(feature = "onnx")]
pub use ort::ep::ExecutionProviderDispatch;

pub use crate::common::{get_cache_dir, Embedding, Error, Result, SparseEmbedding, TokenizerFiles};
#[cfg(feature = "onnx")]
pub use crate::models::{
    model_info::ModelInfo, model_info::RerankerModelInfo, quantization::QuantizationMode,
};
#[cfg(feature = "onnx")]
pub use crate::output::{EmbeddingOutput, OutputKey, OutputPrecedence, SingleBatchOutput};
#[cfg(feature = "onnx")]
pub use crate::pooling::Pooling;

// For all Embedding
#[cfg(feature = "onnx")]
pub use crate::init::{InitOptions as BaseInitOptions, InitOptionsWithLength};
#[cfg(feature = "onnx")]
pub use crate::models::ModelTrait;

// For Text Embedding
#[cfg(feature = "onnx")]
pub use crate::models::text_embedding::EmbeddingModel;
#[cfg(feature = "onnx")]
pub use crate::text_embedding::{
    InitOptionsUserDefined, TextEmbedding, TextInitOptions, UserDefinedEmbeddingModel,
};
#[cfg(feature = "onnx")]
#[deprecated(note = "use `TextInitOptions` instead")]
pub type InitOptions = TextInitOptions;

// For Sparse Text Embedding
#[cfg(feature = "onnx")]
pub use crate::models::sparse::SparseModel;
#[cfg(feature = "onnx")]
pub use crate::sparse_text_embedding::{
    SparseInitOptions, SparseTextEmbedding, UserDefinedSparseModel,
};

// For BGEM3 Joint Embedding
#[cfg(feature = "onnx")]
pub use crate::bgem3_embedding::{
    Bgem3Embedding, Bgem3EmbeddingOutput, Bgem3InitOptions, UserDefinedBgem3Model,
};
#[cfg(feature = "onnx")]
pub use crate::models::bgem3::Bgem3Model;

// For Image Embedding
#[cfg(feature = "image-models")]
pub use crate::image_embedding::{
    ImageEmbedding, ImageInitOptions, ImageInitOptionsUserDefined, ImagePreprocessor,
    UserDefinedImageEmbeddingModel,
};
#[cfg(feature = "onnx")]
pub use crate::models::image_embedding::ImageEmbeddingModel;

// For Reranking
#[cfg(feature = "onnx")]
pub use crate::models::reranking::RerankerModel;
#[cfg(feature = "onnx")]
pub use crate::reranking::{
    OnnxSource, RerankInitOptions, RerankInitOptionsUserDefined, RerankResult, TextRerank,
    UserDefinedRerankingModel,
};

// For Qwen3 (candle backend)
#[cfg(feature = "qwen3-vl")]
pub use crate::models::qwen3::Qwen3VLEmbedding;
#[cfg(feature = "qwen3-text")]
pub use crate::models::qwen3::{
    Config as Qwen3Config, Qwen3Model, Qwen3Reranker, Qwen3TextEmbedding,
    QWEN3_RERANKER_DEFAULT_INSTRUCTION,
};

// For Nomic Embed Text v2 MoE (candle backend)
#[cfg(feature = "nomic-v2-moe")]
pub use crate::models::nomic_v2_moe::{NomicBertModel, NomicConfig, NomicV2MoeTextEmbedding};
