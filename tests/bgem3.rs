#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use fastembed::{
    get_cache_dir, Bgem3Embedding, Bgem3InitOptions, Bgem3Model, InitOptionsUserDefined,
    TokenizerFiles, UserDefinedBgem3Model,
};
use std::collections::HashMap;
use std::sync::Mutex;

static MODEL_LOCK: Mutex<()> = Mutex::new(());
const EPS: f32 = 2e-2;

#[test]
fn test_bgem3_joint_embeddings_match_python() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let mut model = Bgem3Embedding::try_new(Bgem3InitOptions::new(Bgem3Model::BGEM3Q))
        .expect("Failed to initialize BGEM3Q model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // 1. Verify Dense Embeddings
    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.dense[0].len(), 1024);
    assert_eq!(output.dense[1].len(), 1024);

    let expected_dense_0 = [
        -0.018156249076128006,
        -0.017044715583324432,
        0.000982290250249207,
        0.0583689846098423,
        -0.01868816465139389,
    ];
    let expected_dense_1 = [
        -0.011247089132666588,
        0.031949788331985474,
        -0.02800164744257927,
        -0.009801163338124752,
        -0.014338407665491104,
    ];

    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_dense_1.iter().enumerate() {
        assert!((output.dense[1][i] - val).abs() < EPS);
    }

    // 2. Verify Sparse Embeddings
    assert_eq!(output.sparse.len(), 2);

    let expected_sparse_0: HashMap<usize, f32> = [
        (9, 0.04261402785778046),
        (71, 0.09023943543434143),
        (83, 0.08396764099597931),
        (195, 0.16971012949943542),
        (1379, 0.10828342288732529),
        (1430, 0.13637235760688782),
        (4271, 0.16798287630081177),
        (4295, 0.2422717958688736),
        (9795, 0.1467694491147995),
        (13482, 0.277856707572937),
        (16655, 0.22456319630146027),
        (86872, 0.18163326382637024),
    ]
    .into_iter()
    .collect();

    let expected_sparse_1: HashMap<usize, f32> = [
        (70, 0.15444988012313843),
        (83, 0.1826561987400055),
        (111, 0.10440966486930847),
        (756, 0.1211288720369339),
        (1601, 0.16578607261180878),
        (1733, 0.1628011018037796),
        (2965, 0.24118179082870483),
        (90865, 0.23467521369457245),
        (183497, 0.30678409337997437),
    ]
    .into_iter()
    .collect();

    assert_eq!(output.sparse[0].indices.len(), expected_sparse_0.len());
    for (idx, val) in output.sparse[0]
        .indices
        .iter()
        .zip(output.sparse[0].values.iter())
    {
        let expected_val = expected_sparse_0
            .get(idx)
            .expect("Unexpected index in sparse 0");
        assert!(
            (val - expected_val).abs() < EPS,
            "Sparse 0 index {}: expected {}, got {}",
            idx,
            expected_val,
            val
        );
    }

    assert_eq!(output.sparse[1].indices.len(), expected_sparse_1.len());
    for (idx, val) in output.sparse[1]
        .indices
        .iter()
        .zip(output.sparse[1].values.iter())
    {
        let expected_val = expected_sparse_1
            .get(idx)
            .expect("Unexpected index in sparse 1");
        assert!(
            (val - expected_val).abs() < EPS,
            "Sparse 1 index {}: expected {}, got {}",
            idx,
            expected_val,
            val
        );
    }

    // 3. Verify ColBERT Embeddings
    assert_eq!(output.colbert.len(), 2);
    assert_eq!(output.colbert[0].len(), 13);
    assert_eq!(output.colbert[1].len(), 10);

    let expected_colbert_0_tok1 = [
        -0.02416383847594261,
        -0.0405534990131855,
        -0.0560004822909832,
    ];
    let expected_colbert_0_tok2 = [
        -0.01845022290945053,
        -0.042646653950214386,
        -0.033078353852033615,
    ];
    let expected_colbert_1_tok1 = [
        -0.013565482571721077,
        -0.04746083542704582,
        -0.027890587225556374,
    ];
    let expected_colbert_1_tok2 = [
        0.004542498383671045,
        -0.05220562964677811,
        -0.045384544879198074,
    ];

    for (i, val) in expected_colbert_0_tok1.iter().enumerate() {
        assert!((output.colbert[0][0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_0_tok2.iter().enumerate() {
        assert!((output.colbert[0][1][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_1_tok1.iter().enumerate() {
        assert!((output.colbert[1][0][i] - val).abs() < EPS);
    }
    for (i, val) in expected_colbert_1_tok2.iter().enumerate() {
        assert!((output.colbert[1][1][i] - val).abs() < EPS);
    }
}

#[test]
fn test_bgem3_user_defined_model() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    // We will verify the user-defined loader by pulling the files from HF and feeding them manually to simulate a local deployment

    // Reuse fastembed's cache, the model is already downloaded by test_bgem3_joint_embeddings_match_python
    let cache = hf_hub::Cache::new(std::path::PathBuf::from(get_cache_dir()));
    let model_repo = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(false)
        .build()
        .expect("Failed to build API client")
        .model(Bgem3Model::BGEM3Q.to_string());

    let onnx_file = std::fs::read(
        model_repo
            .get("model_quantized.onnx")
            .expect("Failed to get model file"),
    )
    .expect("Failed to read model file");

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: std::fs::read(model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: std::fs::read(model_repo.get("special_tokens_map.json").unwrap())
            .unwrap(),
        tokenizer_config_file: std::fs::read(model_repo.get("tokenizer_config.json").unwrap())
            .unwrap(),
    };

    let user_model = UserDefinedBgem3Model::new(onnx_file, tokenizer_files);
    let mut model =
        Bgem3Embedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::default())
            .expect("Failed to build user defined BGEM3 model");

    let sentences = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "Superman is the best superhero of all time",
    ];

    let output = model.embed(sentences, None).expect("Embedding failed");

    assert_eq!(output.dense.len(), 2);
    assert_eq!(output.sparse.len(), 2);
    assert_eq!(output.colbert.len(), 2);

    // Verify a simple weight from the first sentence to ensure it produces identical numbers
    let expected_dense_0 = [
        -0.018156249076128006,
        -0.017044715583324432,
        0.000982290250249207,
        0.0583689846098423,
        -0.01868816465139389,
    ];
    for (i, val) in expected_dense_0.iter().enumerate() {
        assert!((output.dense[0][i] - val).abs() < EPS);
    }
}

#[test]
fn test_bgem3_custom_max_length() {
    let _guard = MODEL_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    // Verify that the user can override the max length (e.g. to 5 tokens) and it successfully truncates
    let mut model =
        Bgem3Embedding::try_new(Bgem3InitOptions::new(Bgem3Model::BGEM3Q).with_max_length(5))
            .expect("Failed to initialize BGEM3Q model with custom max length");

    let sentences = vec!["fastembed-rs is licensed under Apache 2.0 and is very cool"];

    let output = model.embed(sentences, None).expect("Embedding failed");

    // The sentence "fastembed-rs is licensed under Apache 2.0 and is very cool" has more than 5 tokens.
    // If truncated to 5 tokens, the ColBERT output token count (excluding CLS) should be exactly 4 tokens
    // since the total tokens with CLS/EOS is limited by max_length = 5.
    assert_eq!(output.colbert[0].len(), 4);
}
