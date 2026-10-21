#![cfg(feature = "hf-hub")]
#![cfg(feature = "qwen3")]
#![allow(clippy::excessive_precision)]

use candle_core::{DType, Device};
use fastembed::{Qwen3TextEmbedding, Qwen3VLEmbedding};

const REPO_06B: &str = "Qwen/Qwen3-Embedding-0.6B";
const REPO_4B: &str = "Qwen/Qwen3-Embedding-4B";
const REPO_8B: &str = "Qwen/Qwen3-Embedding-8B";
const REPO_VL_2B: &str = "Qwen/Qwen3-VL-Embedding-2B";
const MAX_LENGTH: usize = 512;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn run_embed_test(repo_id: &str) {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(repo_id, &device, DType::F32, MAX_LENGTH).expect("load model");

    let queries = ["What is the capital of China?", "Explain gravity"];
    let documents = [
        "Beijing is the capital of China.",
        "Gravity is a force that attracts objects toward each other.",
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

    assert!(
        q0_d0 > q0_d1,
        "query0 should match doc0 better: {q0_d0} vs {q0_d1}"
    );
    assert!(
        q1_d1 > q1_d0,
        "query1 should match doc1 better: {q1_d1} vs {q1_d0}"
    );
}

#[test]
fn qwen3_06b_embed() {
    run_embed_test(REPO_06B);
}

/// Reference scores from official Qwen3-Embedding model card: [[0.7646, 0.1414], [0.1355, 0.6000]]
#[test]
fn qwen3_06b_reference_scores() {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(REPO_06B, &device, DType::F32, 8192).expect("load model");

    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let queries = [
        format!("Instruct: {task}\nQuery:What is the capital of China?"),
        format!("Instruct: {task}\nQuery:Explain gravity"),
    ];
    let documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. \
         It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ];

    let input_texts: Vec<&str> = queries
        .iter()
        .map(|s| s.as_str())
        .chain(documents)
        .collect();
    let embeddings = model.embed(&input_texts).expect("embed");

    let scores = [
        [
            cosine_sim(&embeddings[0], &embeddings[2]),
            cosine_sim(&embeddings[0], &embeddings[3]),
        ],
        [
            cosine_sim(&embeddings[1], &embeddings[2]),
            cosine_sim(&embeddings[1], &embeddings[3]),
        ],
    ];
    let expected = [[0.7646f32, 0.1414f32], [0.1355f32, 0.6000f32]];

    for i in 0..2 {
        for j in 0..2 {
            let diff = (scores[i][j] - expected[i][j]).abs();
            assert!(
                diff < 0.05,
                "score[{i}][{j}]: got {:.4}, expected {:.4}",
                scores[i][j],
                expected[i][j]
            );
        }
    }
}

/// The official Q8_0 GGUF of Qwen3-Embedding-0.6B on the CPU against Python. candle's CPU
/// quantized matmul rounds each activation row to 8-bit blocks before the dot products (as
/// llama.cpp's CPU path does), which moves single values by up to about 7e-3 and a vector's sum
/// by up to about 4e-2.
#[test]
fn qwen3_06b_gguf_matches_python() {
    check_06b_gguf_against_python(&Device::Cpu, 1e-2, 6e-2);
}

/// The same check on Metal, whose quantized matmul keeps the activations in f32: values agree with
/// Python to about 1e-4.
#[cfg(feature = "metal")]
#[test]
fn qwen3_06b_gguf_matches_python_on_metal() {
    check_06b_gguf_against_python(&Device::new_metal(0).expect("metal"), 5e-4, 5e-3);
}

/// The official Q8_0 GGUF of Qwen3-Embedding-0.6B, each text embedded alone, against reference
/// values from Python: transformers loading the same GGUF file (it dequantizes the weights to
/// f32), the original model's tokenizer, last-token pooling and L2 normalization, f32 on the CPU.
fn check_06b_gguf_against_python(device: &Device, dim_tolerance: f32, sum_tolerance: f32) {
    use fastembed::GgufMatmul;

    let model = Qwen3TextEmbedding::from_hf_gguf(
        "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "Qwen3-Embedding-0.6B-Q8_0.gguf",
        REPO_06B,
        device,
        8192,
        GgufMatmul::default(),
    )
    .expect("load model");

    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let texts = [
        format!("Instruct: {task}\nQuery:What is the capital of China?"),
        format!("Instruct: {task}\nQuery:Explain gravity"),
        "The capital of China is Beijing.".to_string(),
        "Gravity is a force that attracts two bodies towards each other. \
         It gives weight to physical objects and is responsible for the movement of planets around the sun."
            .to_string(),
    ];
    let expected_first8_from_python = [
        [
            -0.05209115_f32,
            -0.03108022,
            -0.00009076,
            -0.02695240,
            -0.03562966,
            -0.05211519,
            0.04888062,
            0.02313953,
        ],
        [
            -0.01199439,
            -0.03724955,
            -0.00091878,
            0.04037875,
            0.02750070,
            0.05807723,
            0.00280154,
            -0.02796479,
        ],
        [
            -0.04752747,
            -0.02216994,
            0.00374661,
            -0.03286595,
            -0.05911085,
            -0.00602503,
            0.01416930,
            0.05129674,
        ],
        [
            -0.05318979,
            -0.01622364,
            -0.00109899,
            0.00721747,
            0.03609022,
            0.03295489,
            0.01974075,
            -0.00920590,
        ],
    ];
    let expected_sum_from_python = [-0.4117671_f32, 0.8972959, -0.3112229, 1.5693551];

    let mut embeddings = Vec::new();
    for text in &texts {
        embeddings.extend(model.embed(std::slice::from_ref(text)).expect("embed"));
    }
    for (idx, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), model.config().hidden_size);
        let sum = emb.iter().sum::<f32>();
        assert!(
            (sum - expected_sum_from_python[idx]).abs() < sum_tolerance,
            "text {idx} sum mismatch: got {sum}, expected {}",
            expected_sum_from_python[idx]
        );
        for dim in 0..8 {
            let got = emb[dim];
            let expected = expected_first8_from_python[idx][dim];
            assert!(
                (got - expected).abs() < dim_tolerance,
                "text {idx} dim {dim} mismatch: got {got}, expected {expected}"
            );
        }
    }

    // The model card's scores for these texts, as `qwen3_06b_reference_scores` checks them.
    let expected = [[0.7646f32, 0.1414f32], [0.1355f32, 0.6000f32]];
    for i in 0..2 {
        for j in 0..2 {
            let score = cosine_sim(&embeddings[i], &embeddings[2 + j]);
            assert!(
                (score - expected[i][j]).abs() < 0.05,
                "score[{i}][{j}]: got {score:.4}, expected {:.4}",
                expected[i][j]
            );
        }
    }
}

#[test]
fn qwen3_4b_embed() {
    if std::env::var("RUN_QWEN3_4B").is_err() {
        return;
    }
    run_embed_test(REPO_4B);
}

#[test]
fn qwen3_8b_embed() {
    if std::env::var("RUN_QWEN3_8B").is_err() {
        return;
    }
    run_embed_test(REPO_8B);
}

#[test]
fn qwen3_vl_2b_text_embed() {
    if std::env::var("RUN_QWEN3_VL_2B").is_err() {
        return;
    }
    run_embed_test(REPO_VL_2B);
}

#[test]
fn qwen3_vl_2b_image_embed() {
    if std::env::var("RUN_QWEN3_VL_2B_IMAGE").is_err() {
        return;
    }

    let device = Device::Cpu;
    let model = Qwen3VLEmbedding::from_hf(REPO_VL_2B, &device, DType::F32, 2048).expect("load");

    let images = ["tests/assets/image_0.png", "tests/assets/image_1.png"];
    let embeddings = model.embed_images(&images).expect("embed images");
    let expected_sum_from_python = [0.58487093_f32, 0.98103923_f32];
    let expected_first8_from_python = [
        [
            0.06573638_f32,
            -0.0034184097_f32,
            0.019454964_f32,
            -0.009746583_f32,
            0.029501466_f32,
            0.03822506_f32,
            -0.04341881_f32,
            0.038255773_f32,
        ],
        [
            0.055722896_f32,
            0.03549703_f32,
            0.013118794_f32,
            -0.015905563_f32,
            0.0067542493_f32,
            0.029612768_f32,
            0.005418665_f32,
            0.028382933_f32,
        ],
    ];
    let expected_dot_from_python = 0.5559935_f32;

    assert_eq!(embeddings.len(), images.len());
    for (idx, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), model.config().hidden_size);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected L2-normalized, got {norm}"
        );

        // Reference values generated by official Python implementation:
        // https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_embedding.py
        let sum = emb.iter().sum::<f32>();
        assert!(
            (sum - expected_sum_from_python[idx]).abs() < 1e-2,
            "image {idx} sum mismatch: got {sum}, expected {}",
            expected_sum_from_python[idx]
        );

        for dim in 0..8 {
            let got = emb[dim];
            let expected = expected_first8_from_python[idx][dim];
            assert!(
                (got - expected).abs() < 1.5e-2,
                "image {idx} dim {dim} mismatch: got {got}, expected {expected}"
            );
        }
    }

    let image_dot = cosine_sim(&embeddings[0], &embeddings[1]);
    assert!(
        (image_dot - expected_dot_from_python).abs() < 1e-2,
        "image cosine mismatch: got {image_dot}, expected {expected_dot_from_python}"
    );
}
