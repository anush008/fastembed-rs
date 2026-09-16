#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use fastembed::{SparseInitOptions, SparseModel, SparseTextEmbedding};
use std::collections::HashMap;

#[test]
fn test_bgem3_sparse_embeddings_match_python() {
    let mut model = SparseTextEmbedding::try_new(SparseInitOptions::new(SparseModel::BGEM3))
        .expect("Failed to initialize BGEM3 model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let embeddings = model.embed(sentences, None).expect("Embedding failed");

    // Expected values from Python
    // from FlagEmbedding import BGEM3FlagModel
    // model = BGEM3FlagModel('BAAI/bge-m3')
    let expected_0: HashMap<usize, f32> = [
        (4271, 0.17456965),
        (195, 0.16260204),
        (13482, 0.28582922),
        (9, 0.04153823),
        (4295, 0.24856839),
        (83, 0.07778944),
        (86872, 0.17708361),
        (71, 0.08359783),
        (1379, 0.10846229),
        (9795, 0.1580239),
        (1430, 0.15291117),
        (16655, 0.223301),
    ]
    .into_iter()
    .collect();

    let expected_1: HashMap<usize, f32> = [
        (183497, 0.32012847),
        (83, 0.19853045),
        (70, 0.16743071),
        (2965, 0.24451455),
        (1601, 0.17550871),
        (90865, 0.25476876),
        (111, 0.11962792),
        (756, 0.13541803),
        (1733, 0.18622744),
    ]
    .into_iter()
    .collect();

    let expected = [expected_0, expected_1];

    assert_eq!(embeddings.len(), expected.len());

    for (i, embedding) in embeddings.iter().enumerate() {
        let expected_map = &expected[i];

        assert_eq!(embedding.indices.len(), expected_map.len());

        for (idx, val) in embedding.indices.iter().zip(embedding.values.iter()) {
            let expected_val = expected_map.get(idx).expect("Unexpected index");
            assert!((val - expected_val).abs() < 1e-4);
        }
    }
}
