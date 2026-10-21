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

/// Held around every model download in this file. The tests run in parallel and several fetch
/// the same repo; hf-hub 0.5 waits only about 5 seconds for another download's file lock and
/// then fails with `LockAcquisition`, so on a cold cache all but the first would fail.
static HF_DOWNLOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `f` with the download lock held. A test that panicked while holding it leaves nothing half
/// done that the next load could see, so a poisoned lock is taken as it is.
fn one_download_at_a_time<T>(f: impl FnOnce() -> T) -> T {
    let _guard = HF_DOWNLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn run_embed_test(repo_id: &str) {
    let device = Device::Cpu;
    let model = one_download_at_a_time(|| {
        Qwen3TextEmbedding::from_hf(repo_id, &device, DType::F32, MAX_LENGTH)
    })
    .expect("load model");

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
        one_download_at_a_time(|| Qwen3TextEmbedding::from_hf(REPO_06B, &device, DType::F32, 8192))
            .expect("load model");

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

/// candle's CPU quantized matmul rounds the activations to 8-bit blocks, as llama.cpp does.
#[test]
fn qwen3_06b_gguf_matches_python() {
    check_06b_gguf_against_python(&Device::Cpu, 1e-2, 6e-2);
}

#[cfg(feature = "metal")]
#[test]
fn qwen3_06b_gguf_matches_python_on_metal() {
    check_06b_gguf_against_python(&Device::new_metal(0).expect("metal"), 5e-4, 5e-3);
}

/// The Q8_0 GGUF of Qwen3-Embedding-0.6B, each text embedded alone. The Python values come from
/// transformers loading the same file in f32, with last-token pooling and L2 normalization. Each
/// full vector is also compared with the f32 safetensors model.
fn check_06b_gguf_against_python(device: &Device, dim_tolerance: f32, sum_tolerance: f32) {
    let model = one_download_at_a_time(|| {
        Qwen3TextEmbedding::from_hf_gguf(
            "Qwen/Qwen3-Embedding-0.6B-GGUF",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
            REPO_06B,
            device,
            8192,
        )
    })
    .expect("load model");
    let reference = one_download_at_a_time(|| {
        Qwen3TextEmbedding::from_hf(REPO_06B, &Device::Cpu, DType::F32, 8192)
    })
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
        let emb = model.embed(std::slice::from_ref(text)).expect("embed");
        let want = reference.embed(std::slice::from_ref(text)).expect("embed");
        let cosine = cosine_sim(&emb[0], &want[0]);
        // Q8_0 weights and 8-bit activations give 0.9988 to 0.9994 on the CPU.
        assert!(cosine > 0.998, "cosine to safetensors: {cosine}");
        embeddings.extend(emb);
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

    // The model card's scores.
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

/// Qwen3-Reranker-0.6B on the model card's queries and documents, f32 on the CPU, against the
/// model card's own Python code (transformers, f32 on the CPU, each pair alone). Each query's
/// documents run here as one left-padded batch.
#[test]
fn qwen3_reranker_06b_matches_python() {
    use fastembed::Qwen3Reranker;

    let model = Qwen3Reranker::from_hf("Qwen/Qwen3-Reranker-0.6B", &Device::Cpu, DType::F32, 8192)
        .expect("load model");
    let documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical \
         objects and is responsible for the movement of planets around the sun.",
    ];
    let expected_from_python = [
        (
            "What is the capital of China?",
            [0.99949825_f32, 0.00000507],
        ),
        ("Explain gravity", [0.00004957, 0.99936193]),
    ];
    for (query, expected) in expected_from_python {
        let scores = model.score(None, query, &documents).expect("score");
        for (got, want) in scores.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-4,
                "{query:?}: got {got}, expected {want}"
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
    let model =
        one_download_at_a_time(|| Qwen3VLEmbedding::from_hf(REPO_VL_2B, &device, DType::F32, 2048))
            .expect("load");

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
