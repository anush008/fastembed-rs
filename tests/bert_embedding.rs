#![cfg(all(feature = "bert-text", feature = "hf-hub"))]
#![allow(clippy::excessive_precision)]

use candle_core::{DType, Device};
use fastembed::BertTextEmbedding;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

const TEXTS: [&str; 3] = [
    "fn main() { println!(\"hi\"); }",
    "A horse walks into a bar.",
    "The cache stores chunks by content hash.",
];

/// F32 everywhere: candle's f16 BERT LayerNorm overflows to NaN on this checkpoint
/// (the f16 path is deliberately not the default here until that is fixed).
fn load(device: &Device, expect: &str) -> BertTextEmbedding {
    BertTextEmbedding::from_hf(
        "mixedbread-ai/mxbai-embed-large-v1",
        device,
        DType::F32,
        512,
    )
    .expect(expect)
    .with_batch_size(4)
}

/// Runs on every `bert-text` build, including the CPU-only and CUDA ones: the Metal parity
/// test below needs a Metal device, so the device-independent checks stay here.
#[test]
fn mxbai_cpu_normalizes_embeddings() {
    let cpu = load(&Device::Cpu, "cpu load");
    let out = cpu.embed(&TEXTS).expect("cpu embed");
    assert_eq!(out.len(), TEXTS.len());
    for vector in &out {
        assert_eq!(vector.len(), 1024);
        let n = norm(vector);
        assert!((n - 1.0).abs() < 1e-3, "cpu vector norm {n}");
    }
    // Length-sorted batching still returns the vectors in the input order.
    for (text, batched) in TEXTS.iter().zip(&out) {
        let alone = &cpu.embed(&[text]).expect("cpu embed one text")[0];
        let c = cos(batched, alone);
        assert!(c > 0.999_9, "{text:?} in a batch against alone: cosine {c}");
    }
}

/// RoBERTa counts positions from the padding id, and BERT from 0.
#[test]
fn distilroberta_matches_transformers() {
    let model = BertTextEmbedding::from_hf(
        "sentence-transformers/all-distilroberta-v1",
        &Device::Cpu,
        DType::F32,
        4096,
    )
    .expect("cpu load");
    // The position table has 514 rows, and the first two belong to no token.
    assert_eq!(model.max_length(), 512);

    // The first 8 values of the normalized `last_hidden_state[:, 0]` that transformers 5.23
    // returns for `TEXTS`, padded to one batch.
    let baseline: [[f32; 8]; 3] = [
        [
            2.858_899_0e-2,
            -3.270_944_2e-2,
            -1.929_039_0e-2,
            2.974_913_1e-2,
            2.228_091_3e-2,
            6.800_662_7e-2,
            6.506_870_5e-3,
            2.517_060_4e-2,
        ],
        [
            -1.284_241_5e-2,
            -2.229_934_4e-2,
            -3.423_807_4e-2,
            -4.778_776_7e-2,
            -1.637_971_8e-2,
            -4.964_015_0e-3,
            -3.411_704_7e-2,
            4.602_278_0e-2,
        ],
        [
            3.986_878_3e-2,
            -9.248_185_2e-2,
            -2.973_057_0e-2,
            -3.489_137_8e-2,
            -1.661_183_1e-3,
            6.729_853_9e-2,
            -6.651_513_3e-2,
            -3.024_310_6e-2,
        ],
    ];
    let out = model.embed(&TEXTS).expect("cpu embed");
    for (vector, expected) in out.iter().zip(&baseline) {
        assert_eq!(vector.len(), 768);
        for (actual, expected) in vector.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} is not {expected}"
            );
        }
    }
}

#[cfg(feature = "metal")]
#[test]
fn mxbai_metal_matches_cpu_and_normalizes() {
    let cpu = load(&Device::Cpu, "cpu load");
    let metal = load(&Device::new_metal(0).expect("metal"), "metal load");

    let cpu_out = cpu.embed(&TEXTS).expect("cpu embed");
    let metal_out = metal.embed(&TEXTS).expect("metal embed");
    assert_eq!(cpu_out.len(), TEXTS.len());
    assert_eq!(cpu_out[0].len(), 1024);
    for (reference, candidate) in cpu_out.iter().zip(&metal_out) {
        let c = cos(reference, candidate);
        assert!(c > 0.99, "metal/f32 cosine {c}");
        let n = norm(candidate);
        assert!((n - 1.0).abs() < 1e-3, "metal vector norm {n}");
    }
}
