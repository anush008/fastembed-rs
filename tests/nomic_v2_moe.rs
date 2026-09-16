#![cfg(feature = "hf-hub")]
#![cfg(feature = "nomic-v2-moe")]
#![allow(clippy::excessive_precision)]

use candle_core::{DType, Device};
use fastembed::NomicV2MoeTextEmbedding;

const REPO: &str = "nomic-ai/nomic-embed-text-v2-moe";
const MAX_LENGTH: usize = 512;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
fn nomic_v2_moe_embed() {
    let device = Device::Cpu;
    let model = NomicV2MoeTextEmbedding::from_hf(REPO, &device, DType::F32, MAX_LENGTH)
        .expect("load model");

    let queries = [
        "search_query: What is the capital of China?",
        "search_query: Explain gravity",
    ];
    let documents = [
        "search_document: Beijing is the capital of China.",
        "search_document: Gravity is a force that attracts objects toward each other.",
    ];

    let all_texts: Vec<&str> = queries.iter().chain(documents.iter()).copied().collect();
    let embeddings = model.embed(&all_texts).expect("embed");

    assert_eq!(embeddings.len(), all_texts.len());
    for emb in &embeddings {
        assert_eq!(emb.len(), model.config().hidden_size);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected L2-normalized, got {norm}"
        );
    }

    let q0_d0 = cosine_sim(&embeddings[0], &embeddings[2]);
    let q0_d1 = cosine_sim(&embeddings[0], &embeddings[3]);
    let q1_d0 = cosine_sim(&embeddings[1], &embeddings[2]);
    let q1_d1 = cosine_sim(&embeddings[1], &embeddings[3]);

    println!("q0-d0: {q0_d0:.4}, q0-d1: {q0_d1:.4}");
    println!("q1-d0: {q1_d0:.4}, q1-d1: {q1_d1:.4}");

    assert!(
        q0_d0 > q0_d1,
        "query0 should match doc0 better: {q0_d0} vs {q0_d1}"
    );
    assert!(
        q1_d1 > q1_d0,
        "query1 should match doc1 better: {q1_d1} vs {q1_d0}"
    );
}
