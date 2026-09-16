#![cfg(feature = "hf-hub")]
#![cfg(feature = "optimum-cli")]
#![allow(clippy::excessive_precision)]
//! Test the use of the ``optimum-cli`` to pull models from the Hugging Face Hub,
//! and generate embeddings successfully with the pulled model.
//!
//! Generated models from optimum can have different output types - `last_hidden_state`
//! may not be the default output. This test is to ensure that the correct output key
//! is used when generating embeddings.

use std::{io, path::PathBuf, process};

use fastembed::{
    get_cache_dir, Error, Pooling, QuantizationMode, Result, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};

const EPS: f32 = 1e-4;

/// Check if the ``optimum-cli`` is available.
fn has_optimum_cli() -> bool {
    process::Command::new("optimum-cli")
        .arg("--help")
        .output()
        .is_ok()
}

/// Pull a model from the Hugging Face Hub using ``optimum-cli``.
///
/// This function assumes you have already checked if the ``optimum-cli`` is available.
/// The return error will not distinguish between a missing ``optimum-cli`` and a failed download.
fn pull_model(
    model_name: &str,
    output: &PathBuf,
    pooling: Option<Pooling>,
) -> Result<TextEmbedding> {
    eprintln!("Pulling {model_name} from the Hugging Face Hub...");
    process::Command::new("optimum-cli")
        .args(&[
            "export",
            "onnx",
            "--model",
            model_name,
            output
                .as_os_str()
                .to_str()
                .expect("Failed to convert path to string"),
        ])
        .output()
        .map_err(|e| Error::Other(format!("Failed to pull model: {e}")))?;

    load_model(output, pooling)
}

/// Load bytes from a file, with a nicer error message.
fn load_bytes_from_file(path: &PathBuf) -> io::Result<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to read file at {path:?}: {e}")))
}

/// Load a model from a local directory.
fn load_model(output: &PathBuf, pooling: Option<Pooling>) -> Result<TextEmbedding> {
    let model = UserDefinedEmbeddingModel {
        onnx_file: load_bytes_from_file(&output.join("model.onnx"))?,
        external_initializers: Vec::new(),
        tokenizer_files: TokenizerFiles {
            tokenizer_file: load_bytes_from_file(&output.join("tokenizer.json"))?,
            config_file: load_bytes_from_file(&output.join("config.json"))?,
            special_tokens_map_file: load_bytes_from_file(&output.join("special_tokens_map.json"))?,
            tokenizer_config_file: load_bytes_from_file(&output.join("tokenizer_config.json"))?,
        },
        pooling,
        quantization: QuantizationMode::None,
        output_key: None,
    };

    TextEmbedding::try_new_from_user_defined(model, Default::default())
}

macro_rules! create_test {
    (
        repo_name: $repo_name:literal,
        repo_owner: $repo_owner:literal,
        name: $name:ident,
        pooling: $pooling:expr,
        expected_embedding_dim: $expected_embedding_dim:literal,
        expected: $expected:expr
    ) => {
        #[test]
        fn $name() {
            let repo_name = $repo_name;
            let repo_owner = $repo_owner;
            let model_name = format!("{}/{}", repo_owner, repo_name);
            let cache_dir = get_cache_dir();
            let output_path = format!("{cache_dir}/exported--{repo_owner}--{repo_name}-onnx");
            let output = PathBuf::from(output_path);

            assert!(
                has_optimum_cli(),
                "optimum-cli is not available. Please install it with `pip install optimum-cli`"
            );

            let model = load_model(&output, $pooling).unwrap_or_else(|_| {
                pull_model(&model_name, &output, $pooling).expect("Failed to pull model")
            });
            let mut model = model;

            let documents = vec![
                "Hello, World!",
                "This is an example passage.",
                "fastembed-rs is licensed under Apache-2.0",
                "Some other short text here blah blah blah",
            ];
            let expected_length = documents.len();

            // Generate embeddings with the default batch size, 256
            let embeddings = model
                .embed(documents.clone(), Some(3))
                .expect("Failed to generate embeddings");

            assert_eq!(embeddings.len(), expected_length);
            assert_eq!(embeddings[0].len(), $expected_embedding_dim);

            embeddings
                .into_iter()
                .map(|embedding| embedding.iter().sum::<f32>())
                .zip($expected.iter())
                .enumerate()
                .for_each(|(index, (embedding, expected))| {
                    assert!(
                        (embedding - expected).abs() < EPS,
                        "Mismatched embeddings sum for '{}': Expected: {}, Got: {}",
                        documents[index],
                        expected,
                        embedding
                    );
                });
        }
    };
}

create_test! {
    repo_name: "all-MiniLM-L6-v2",
    repo_owner: "sentence-transformers",
    name: optimum_cli_export_all_minilm_l6_v2_mean,
    pooling: Some(Pooling::Mean), // Mean does not matter here because the output is 2D
    expected_embedding_dim: 384,
    // These are generated by Python. There could be accumulated variations
    // when summed.
    expected: [ 0.5960538 ,  0.36542776, -0.16450086, -0.40904027]
}
create_test! {
    repo_name: "all-MiniLM-L6-v2",
    repo_owner: "sentence-transformers",
    name: optimum_cli_export_all_minilm_l6_v2_cls,
    pooling: Some(Pooling::Cls),
    expected_embedding_dim: 384,
    // These are generated by Python. There could be accumulated variations
    // when summed.
    expected: [ 0.5960538 ,  0.36542776, -0.16450086, -0.40904027]
}
create_test! {
    repo_name: "all-mpnet-base-v2",
    repo_owner: "sentence-transformers",
    name: optimum_cli_export_all_mpnet_base_v2_mean,
    pooling: Some(Pooling::Mean),
    expected_embedding_dim: 768,
    // These are generated by Python. There could be accumulated variations
    // when summed.
    expected: [-0.21253565, -0.05080119,  0.14072478, -0.29081905]
}
create_test! {
    repo_name: "all-mpnet-base-v2",
    repo_owner: "sentence-transformers",
    name: optimum_cli_export_all_mpnet_base_v2_cls,
    pooling: Some(Pooling::Cls),
    expected_embedding_dim: 768,
    // These are generated by Python. There could be accumulated variations
    // when summed.
    expected: [-0.21253565, -0.05080119,  0.14072478, -0.29081905]
}
