#![cfg(all(feature = "bert-text", feature = "hf-hub"))]

use candle_core::{DType, Device};
use fastembed::BertTextEmbedding;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn mxbai_metal_matches_cpu_and_normalizes() {
    let texts = [
        "fn main() { println!(\"hi\"); }",
        "A horse walks into a bar.",
        "The cache stores chunks by content hash.",
    ];
    let cpu = BertTextEmbedding::from_hf(
        "mixedbread-ai/mxbai-embed-large-v1",
        &Device::Cpu,
        DType::F32,
        512,
    )
    .expect("cpu load")
    .with_batch_size(4);
    // F32 on Metal: candle's f16 BERT LayerNorm overflows to NaN on this checkpoint
    // (the f16 path is deliberately not the default here until that is fixed).
    let metal = BertTextEmbedding::from_hf(
        "mixedbread-ai/mxbai-embed-large-v1",
        &Device::new_metal(0).expect("metal"),
        DType::F32,
        512,
    )
    .expect("metal load")
    .with_batch_size(4);

    let cpu_out = cpu.embed(&texts).expect("cpu embed");
    let metal_out = metal.embed(&texts).expect("metal embed");
    assert_eq!(cpu_out.len(), texts.len());
    assert_eq!(cpu_out[0].len(), 1024);
    for (reference, candidate) in cpu_out.iter().zip(&metal_out) {
        let c = cos(reference, candidate);
        assert!(c > 0.99, "metal/f32 cosine {c}");
        let n = norm(candidate);
        assert!((n - 1.0).abs() < 1e-3, "metal vector norm {n}");
    }
}
