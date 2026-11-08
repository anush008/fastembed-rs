#![cfg(all(feature = "bert-text", feature = "hf-hub"))]

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
    let again = cpu.embed(&TEXTS).expect("cpu embed again");
    for (first, second) in out.iter().zip(&again) {
        assert!(
            cos(first, second) > 0.999_999,
            "cpu embedding changed between runs"
        );
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
