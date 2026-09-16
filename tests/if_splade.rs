#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use fastembed::{SparseInitOptions, SparseModel, SparseTextEmbedding};

const EPS: f32 = 1e-3;

/// The MLM head emits a `vocab_size` wide tensor per token, so batches have to stay small:
/// the default batch size of 256 would allocate `256 * 512 * 30522 * 4` bytes at once.
const BATCH_SIZE: usize = 4;

#[test]
fn test_if_splade_embeddings_match_python() {
    let mut model = SparseTextEmbedding::try_new(SparseInitOptions::new(
        SparseModel::OpenSearchNeuralSparseDocV3Gte,
    ))
    .expect("Failed to initialize the inference-free SPLADE model");

    let documents = vec!["Hello World"];

    // Expected values from Python
    // from fastembed import SparseTextEmbedding
    // model = SparseTextEmbedding("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte")
    // The document embedding has many more non-zero dimensions, these are the leading ones.
    let expected_document_indices = [
        999, 1010, 1011, 1024, 1028, 1029, 1045, 1074, 1993, 2017, 2033, 2054, 2073, 2080, 2088,
    ];
    let expected_document_values = [
        0.16544909, 0.00529129, 0.0392109, 0.12337475, 0.09640586, 0.05325737, 0.09611791,
        0.03159865, 0.01349991, 0.09392473, 0.01928805, 0.05238346, 0.05515401, 0.03156782,
        0.98263124,
    ];
    let expected_query_indices = [2088, 7592];
    let expected_query_values = [3.42086864, 6.93775654];

    let embeddings = model
        .embed(documents.clone(), Some(BATCH_SIZE))
        .expect("Embedding failed");

    assert_eq!(embeddings.len(), documents.len());
    let document = &embeddings[0];
    assert_eq!(document.indices.len(), document.values.len());
    assert!(document.indices.len() > expected_document_indices.len());
    assert_eq!(
        document.indices[..expected_document_indices.len()],
        expected_document_indices
    );
    for (i, expected) in expected_document_values.iter().enumerate() {
        assert!(
            (document.values[i] - expected).abs() < EPS,
            "dimension {} is {}, expected {expected}",
            document.indices[i],
            document.values[i],
        );
    }

    // Queries are embedded from the tokenizer and the IDF table alone. `query_embed` takes
    // `&self` while running the ONNX session needs `&mut self`, so a shared borrow of the model
    // is enough to prove that no inference happens on the query side.
    let model: &SparseTextEmbedding = &model;
    let query_embeddings = model
        .query_embed(documents)
        .expect("Query embedding failed");

    assert_eq!(query_embeddings.len(), 1);
    let query = &query_embeddings[0];
    assert_eq!(query.indices, expected_query_indices);
    assert_eq!(query.values.len(), expected_query_values.len());
    for (i, expected) in expected_query_values.iter().enumerate() {
        assert!(
            (query.values[i] - expected).abs() < EPS,
            "dimension {} is {}, expected {expected}",
            query.indices[i],
            query.values[i],
        );
    }
}
