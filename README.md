<div align="center">
  <h1><a href="https://crates.io/crates/fastembed">FastEmbed-rs 🦀</a></h1>
 <h3>Rust library for generating vector embeddings, reranking locally!</h3>
  <a href="https://www.atlascloud.ai/?ref=FCWMKG"><img src="https://www.atlascloud.ai/oss-program/powered-by-atlas-cloud.svg" alt="Powered by Atlas Cloud"/></a>
</div>

## Features

- Supports synchronous usage. No dependency on Tokio.
- Uses [@pykeio/ort](https://github.com/pykeio/ort) for performant ONNX inference.
- Uses [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) for fast encodings.

## Not looking for Rust?

- Python: [fastembed](https://github.com/qdrant/fastembed)
- Go: [fastembed-go](https://github.com/Anush008/fastembed-go)
- JavaScript: [fastembed-js](https://github.com/Anush008/fastembed-js)

## Supported Models

### Text Embedding

<details><summary>Click to list models</summary>

- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-large-en-v1.5**](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [**BAAI/bge-small-zh-v1.5**](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
- [**sentence-transformers/all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [**sentence-transformers/all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [**sentence-transformers/paraphrase-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-multilingual-mpnet-base-v2**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [**nomic-ai/nomic-embed-text-v1**](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [**nomic-ai/nomic-embed-text-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - pairs with `nomic-embed-vision-v1.5` for image-to-text search
- [**intfloat/multilingual-e5-small**](https://huggingface.co/intfloat/multilingual-e5-small)
- [**intfloat/multilingual-e5-base**](https://huggingface.co/intfloat/multilingual-e5-base)
- [**intfloat/multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large)
- [**mixedbread-ai/mxbai-embed-large-v1**](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [**Alibaba-NLP/gte-base-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)
- [**Alibaba-NLP/gte-large-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- [**lightonai/ModernBERT-embed-large**](https://huggingface.co/lightonai/modernbert-embed-large)
- [**Qdrant/clip-ViT-B-32-text**](https://huggingface.co/Qdrant/clip-ViT-B-32-text) - pairs with `clip-ViT-B-32-vision` for image-to-text search
- [**jinaai/jina-embeddings-v2-base-code**](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
- [**jinaai/jina-embeddings-v2-base-en**](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
- [**google/embeddinggemma-300m**](https://huggingface.co/google/embeddinggemma-300m)
- [**nomic-ai/nomic-embed-text-v2-moe**](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) - requires `nomic-v2-moe` feature (candle backend)
- [**Qwen/Qwen3-Embedding-0.6B**](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-Embedding-4B**](https://huggingface.co/Qwen/Qwen3-Embedding-4B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-Embedding-8B**](https://huggingface.co/Qwen/Qwen3-Embedding-8B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-VL-Embedding-2B**](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) - requires `qwen3` feature (candle backend, multimodal via `Qwen3VLEmbedding`)
- [**snowflake/snowflake-arctic-embed-xs**](https://huggingface.co/snowflake/snowflake-arctic-embed-xs)
- [**snowflake/snowflake-arctic-embed-s**](https://huggingface.co/snowflake/snowflake-arctic-embed-s)
- [**snowflake/snowflake-arctic-embed-m**](https://huggingface.co/snowflake/snowflake-arctic-embed-m)
- [**snowflake/snowflake-arctic-embed-m-long**](https://huggingface.co/snowflake/snowflake-arctic-embed-m-long)
- [**snowflake/snowflake-arctic-embed-l**](https://huggingface.co/snowflake/snowflake-arctic-embed-l)

Quantized versions are also available for several models above (append `Q` to the model enum variant, e.g., `EmbeddingModel::BGESmallENV15Q`). EmbeddingGemma additionally ships a 4-bit build as `EmbeddingModel::EmbeddingGemma300MQ4`.

</details>

### Sparse Text Embedding

<details><summary>Click to list models</summary>
  
- [**prithivida/Splade_PP_en_v1**](https://huggingface.co/prithivida/Splade_PP_en_v1) - Default
- [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
- [**opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte**](https://huggingface.co/Qdrant/opensearch-neural-sparse-encoding-doc-v3-gte) - [Inference-free](#inference-free-sparse-embeddings)

</details>

### Image Embedding

<details><summary>Click to list models</summary>
  
- [**Qdrant/clip-ViT-B-32-vision**](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) - Default
- [**Qdrant/resnet50-onnx**](https://huggingface.co/Qdrant/resnet50-onnx)
- [**Qdrant/Unicom-ViT-B-16**](https://huggingface.co/Qdrant/Unicom-ViT-B-16)
- [**Qdrant/Unicom-ViT-B-32**](https://huggingface.co/Qdrant/Unicom-ViT-B-32)
- [**nomic-ai/nomic-embed-vision-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)

</details>

### Reranking

<details><summary>Click to list models</summary>
  
- [**BAAI/bge-reranker-base**](https://huggingface.co/BAAI/bge-reranker-base) - Default
- [**BAAI/bge-reranker-v2-m3**](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [**jinaai/jina-reranker-v1-turbo-en**](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)
- [**jinaai/jina-reranker-v2-base-multiligual**](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

</details>

## ✊ Support

To support the library, please donate to our primary upstream dependency, [`ort`](https://github.com/pykeio/ort?tab=readme-ov-file#-sponsor-ort) - The Rust wrapper for the ONNX runtime.

## Installation

Run the following in your project directory:

```bash
cargo add fastembed
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
fastembed = "7"
```

### Text Embeddings

```rust
use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};

// With default options
let mut model = TextEmbedding::try_new(Default::default())?;

// With custom options
let mut model = TextEmbedding::try_new(
    TextInitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true).with_intra_threads(4),
)?;

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
 println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384
```

### Sparse Text Embeddings

```rust
use fastembed::{SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding};

// With default options
let mut model = SparseTextEmbedding::try_new(Default::default())?;

// With custom options
let mut model = SparseTextEmbedding::try_new(
    SparseInitOptions::new(SparseModel::SPLADEPPV1).with_show_download_progress(true),
)?;

let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    "fastembed-rs is licensed under Apache 2.0"
];

// Generate embeddings with the default batch size, 256
let embeddings: Vec<SparseEmbedding> = model.embed(documents, None)?;
```

### Image Embeddings

```rust
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel};

// With default options
let mut model = ImageEmbedding::try_new(Default::default())?;

// With custom options
let mut model = ImageEmbedding::try_new(
    ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
)?;

let images = vec!["assets/image_0.png", "assets/image_1.png"];

// Generate embeddings with the default batch size, 256
let embeddings = model.embed(images, None)?;

println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 2
println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 512
```

### Candidates Reranking

```rust
use fastembed::{TextRerank, RerankInitOptions, RerankerModel};

// With default options
let mut model = TextRerank::try_new(Default::default())?;

// With custom options
let mut model = TextRerank::try_new(
    RerankInitOptions::new(RerankerModel::BGERerankerBase).with_show_download_progress(true),
)?;

let documents = vec![
    "hi",
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear, is a bear species endemic to China.",
    "panda is animal",
    "i dont know",
    "kind of mammal",
];

// Rerank with the default batch size, 256 and return document contents
let results = model.rerank("what is panda?", documents, true, None)?;
println!("Rerank result: {:?}", results);
```

### Locally Available Models

Alternatively, local model files can be used for inference via the `try_new_from_user_defined(...)` methods of the respective structs.

### Similarity Search

Helpers in the [`similarity`](https://docs.rs/fastembed/latest/fastembed/similarity/) module score and rank the vectors `embed` returns, so a quick in-memory search needs no extra crate:

```rust
use fastembed::similarity::{cosine_similarity, top_k};

// `embeddings` is the Vec<Embedding> from model.embed(...)
let query = &embeddings[0];

// Score two vectors directly ([-1.0, 1.0], higher = closer)
let score = cosine_similarity(query, &embeddings[1]);

// Or rank the corpus: (index, score) pairs, best first
let hits = top_k(query, &embeddings, 5);
println!("Closest: {:?}", hits);
```

For larger corpora or persistence, push the vectors to a vector search engine (e.g. [Qdrant](https://qdrant.tech/)) and query there.

### Inference-free Sparse Embeddings

```rust
use fastembed::{SparseInitOptions, SparseModel, SparseTextEmbedding};

let mut model = SparseTextEmbedding::try_new(
    SparseInitOptions::new(SparseModel::OpenSearchNeuralSparseDocV3Gte).with_max_length(8192),
)?;

//  This model emits one score per vocabulary entry per token.
// So keep the batch size small.
let documents = model.embed(vec!["Hello World"], Some(4))?;

// Queries are from the tokenizer and a precomputed IDF table alone, without any inference.
let queries = model.query_embed(vec!["Hello World"])?;
```

### Qwen3 Embeddings

Qwen3 embedding models are available behind the `qwen3` feature flag (candle backend).

```toml
[dependencies]
fastembed = { version = "5", features = ["qwen3"] }
```

```rust
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;

let device = Device::Cpu;
let model = Qwen3TextEmbedding::from_hf(
    "Qwen/Qwen3-Embedding-0.6B",
    &device,
    DType::F32,
    512,
)?;

// Text-only usage with the Qwen3-VL embedding checkpoint is also supported:
// let model = Qwen3TextEmbedding::from_hf("Qwen/Qwen3-VL-Embedding-2B", &device, DType::F32, 512)?;

let embeddings = model.embed(&["query: ...", "passage: ..."])?;
println!("Embeddings length: {}", embeddings.len());
```

For multimodal text/image usage with `Qwen/Qwen3-VL-Embedding-2B`:

```rust
use candle_core::{DType, Device};
use fastembed::Qwen3VLEmbedding;

let device = Device::Cpu;
let model = Qwen3VLEmbedding::from_hf(
    "Qwen/Qwen3-VL-Embedding-2B",
    &device,
    DType::F32,
    2048,
)?;

let image_embeddings = model.embed_images(&["tests/assets/image_0.png", "tests/assets/image_1.png"])?;
let text_embeddings = model.embed_texts(&["query: blue cat", "query: red cat"])?;

println!("Image embeddings: {}", image_embeddings.len());
println!("Text embeddings: {}", text_embeddings.len());
```

### Nomic Embed Text v2 MoE

The [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) model is available behind the `nomic-v2-moe` feature flag (candle backend). First general-purpose MoE embedding model with 100+ language support.

```toml
[dependencies]
fastembed = { version = "5", features = ["nomic-v2-moe"] }
```

```rust
use candle_core::{DType, Device};
use fastembed::NomicV2MoeTextEmbedding;

let device = Device::Cpu;
let model = NomicV2MoeTextEmbedding::from_hf(
    "nomic-ai/nomic-embed-text-v2-moe",
    &device,
    DType::F32,
    512,
)?;

let embeddings = model.embed(&["search_query: ...", "search_document: ..."])?;
println!("Embeddings length: {}", embeddings.len());
```

### BGE-M3 Joint Embeddings

The BGE-M3 model produces dense, sparse, and ColBERT embeddings simultaneously in a single forward pass.

```rust
use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model};

// With default options
let mut model = Bgem3Embedding::try_new(Default::default())?;

// With custom options (supporting custom max length up to 8192 tokens)
let mut model = Bgem3Embedding::try_new(
    Bgem3InitOptions::new(Bgem3Model::BGEM3Q)
        .with_max_length(1024)
        .with_show_download_progress(true),
)?;

let documents = vec![
    "Hello, World!",
    "This is an example passage.",
    "fastembed-rs is licensed under Apache 2.0",
    "i dont know"
];

// Generate all three representations in a single forward pass
let output = model.embed(documents, None)?;

println!("Dense dimension: {}", output.dense[0].len()); // -> Dense dimension: 1024

let sparse_emb = &output.sparse[0];
println!("Sparse non-zero tokens: {}", sparse_emb.indices.len());

println!("ColBERT token count: {}", output.colbert[0].len());
```

## Model cache

Models download on first use and load from cache afterwards (no network needed at runtime once cached).

- `FASTEMBED_CACHE_DIR` — cache location (default: `.fastembed_cache`). Equivalent to `TextInitOptions::with_cache_dir`.
- `HF_HOME` — if set, takes precedence over the above.
- `HF_ENDPOINT` — Hugging Face mirror base URL, for restricted networks.
- `HF_TOKEN` — Hugging Face access token, for gated repositories and higher rate limits. Without it, the token saved by `huggingface-cli login` is used when present.

### DirectML (Windows)

To run models on a GPU via DirectML on Windows, enable the `directml` feature:

```toml
[dependencies]
fastembed = { version = "5", features = ["directml"] }
```

Then pass a DirectML execution provider when initializing a model:

```rust
use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};
use ort::ep::DirectML;

let model = TextEmbedding::try_new(
    TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_execution_providers(vec![DirectML::default().into()]),
)?;
```

When DirectML is detected, fastembed automatically disables memory pattern optimization and parallel execution on the ONNX Runtime session, as required by the DirectML execution provider.

## Error handling

Fastembed returns a typed [`fastembed::Error`](https://docs.rs/fastembed/latest/fastembed/enum.Error.html). The type is re-exported from the crate root. The enum is `#[non_exhaustive]`. New variants can be added in minor releases without breaking `match` arms.

```rust
use fastembed::{Error, Result, TextEmbedding};

fn load() -> Result<TextEmbedding> {
    let model = TextEmbedding::try_new(Default::default())?;
    Ok(model)
    // ...
}
```

To handle an error, match on the variant that applies:

```rust
use fastembed::{Error, TextEmbedding, TextInitOptions, EmbeddingModel};

match TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)) {
    Ok(model) => { /* ... */ }
    Err(Error::ModelRetrieval { file, source }) => {
        eprintln!("could not fetch {file}: {source}");
    }
    /*
    ...
    */
    Err(Error::Ort(err)) => {
        eprintln!("ONNX runtime error: {err}");
    }
    Err(e) => eprintln!("{e}"),
}
```

## LICENSE

[Apache 2.0](https://github.com/Anush008/fastembed-rs/blob/main/LICENSE)
