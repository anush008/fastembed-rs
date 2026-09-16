#![cfg(feature = "hf-hub")]
#![allow(clippy::excessive_precision)]

use std::fs;
use std::path::Path;

use hf_hub::Repo;

use fastembed::{
    get_cache_dir, Embedding, EmbeddingModel, InitOptionsUserDefined, OnnxSource, Pooling,
    QuantizationMode, RerankInitOptions, RerankInitOptionsUserDefined, RerankerModel,
    RerankerModelInfo, SparseInitOptions, SparseTextEmbedding, TextEmbedding, TextInitOptions,
    TextRerank, TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};

/// A small epsilon value for floating point comparisons.
const EPS: f32 = 2e-2;

/// Precalculated embeddings for the supported models using #99
/// (4f09b6842ce1fcfaf6362678afcad9a176e05304).
///
/// These are the sum of all embedding values for each document. While not
/// perfect, they should be good enough to verify that the embeddings are being
/// generated correctly.
///
/// If you have just inserted a new `EmbeddingModel` variant, please update the
/// expected embeddings.
///
/// # Returns
///
/// If the embeddings are correct, this function returns `Ok(())`. If there are
/// any mismatches, it returns `Err(Vec<usize>)` with the indices of the
/// mismatched embeddings.
#[allow(unreachable_patterns)]
fn verify_embeddings(model: &EmbeddingModel, embeddings: &[Embedding]) -> Result<(), Vec<usize>> {
    let expected = match model {
        EmbeddingModel::AllMiniLML12V2 => [-0.12147753, 0.30144796, -0.06882502, -0.6303331],
        EmbeddingModel::AllMiniLML12V2Q => [-0.07808663, 0.27919534, -0.0770612, -0.75660324],
        EmbeddingModel::AllMiniLML6V2 => [0.59605527, 0.36542925, -0.16450031, -0.40903988],
        EmbeddingModel::AllMiniLML6V2Q => [0.5677276, 0.40180072, -0.15454668, -0.4672576],
        EmbeddingModel::AllMpnetBaseV2=> [-0.21253541, -0.050802127, 0.14072442, -0.2908188],
        EmbeddingModel::BGEBaseENV15 => [-0.51290065, -0.4844747, -0.53036124, -0.5337459],
        EmbeddingModel::BGEBaseENV15Q => [-0.5130697, -0.48461288, -0.53067875, -0.5337806],
        EmbeddingModel::BGELargeENV15 => [-0.19347441, -0.28394595, -0.1549195, -0.22201893],
        EmbeddingModel::BGELargeENV15Q => [-0.19366685, -0.2842059, -0.15471499, -0.22216901],
        EmbeddingModel::BGESmallENV15 => [0.09881669, 0.15151203, 0.12057499, 0.13641948],
        EmbeddingModel::BGESmallENV15Q => [0.09881936, 0.15154803, 0.12057378, 0.13639033],
        EmbeddingModel::BGESmallZHV15 => [-1.1194772, -1.0928253, -1.0325904, -1.0050416],
        EmbeddingModel::BGELargeZHV15 => [-0.62066114, -0.76666945, -0.7013123, -0.86202735],
        EmbeddingModel::BGEM3 => [-0.7138151, -0.69116485, -0.7932898, -0.6727733],
        EmbeddingModel::GTEBaseENV15 => [-1.6900877, -1.7148916, -1.7333382, -1.5121834],
        EmbeddingModel::GTEBaseENV15Q => [-1.7032102, -1.7076654, -1.729326, -1.5317788],
        EmbeddingModel::GTELargeENV15 => [-1.6457459, -1.6582386, -1.6809471, -1.6070237],
        EmbeddingModel::GTELargeENV15Q => [-1.6044945, -1.6469251, -1.6828246, -1.6265479],
        EmbeddingModel::ModernBertEmbedLarge => [ 0.24799639, 0.32174295, 0.17255782, 0.32919246],
        EmbeddingModel::MultilingualE5Base => [-0.057211064, -0.14287914, -0.071678676, -0.17549144],
        EmbeddingModel::MultilingualE5Large => [-0.7473163, -0.76040405, -0.7537941, -0.72920954],
        EmbeddingModel::MultilingualE5Small => [-0.2640718, -0.13929011, -0.08091972, -0.12388548],
        EmbeddingModel::MxbaiEmbedLargeV1 => [-0.2032495, -0.29803938, -0.15803768, -0.23155808],
        EmbeddingModel::MxbaiEmbedLargeV1Q => [-0.1811538, -0.2884392, -0.1636593, -0.21548103],
        EmbeddingModel::NomicEmbedTextV1 => [0.13788113, 0.10750078, 0.050809078, 0.09284662],
        EmbeddingModel::NomicEmbedTextV15 => [0.1932303, 0.13795732, 0.14700879, 0.14940643],
        EmbeddingModel::NomicEmbedTextV15Q => [0.20999804, 0.17161125, 0.15987156, 0.19436662],
        EmbeddingModel::ParaphraseMLMiniLML12V2 => [-0.07795018, -0.059113946, -0.043668486, -0.1880083],
        EmbeddingModel::ParaphraseMLMiniLML12V2Q => [-0.07749095, -0.058981877, -0.043487836, -0.18775631],
        EmbeddingModel::ParaphraseMLMpnetBaseV2 => [0.39132136, 0.49490625, 0.65497226, 0.34237382],
        EmbeddingModel::ClipVitB32 => [0.7057363, 1.3549932, 0.46823958, 0.52351093],
        EmbeddingModel::JinaEmbeddingsV2BaseCode => [-0.31383067, -0.3758629, -0.24878195, -0.35373706],
        EmbeddingModel::JinaEmbeddingsV2BaseEN => [-0.055866606, -0.033922599, 0.012131551, -0.0132129812],
        EmbeddingModel::EmbeddingGemma300M => [0.22703816, 0.6947083, 0.07579082, 1.6958784],
        EmbeddingModel::EmbeddingGemma300MQ4 => [0.3110208, 0.6683019, 0.38347214, 1.787025],
        EmbeddingModel::EmbeddingGemma300MQ => [0.11791767, 0.34993136, -0.018153993, 1.4971508],
        EmbeddingModel::SnowflakeArcticEmbedXS => [0.4418098, 0.46424747, 0.37932625, 0.44663674],
        EmbeddingModel::SnowflakeArcticEmbedXSQ => [0.45034444, 0.46853474, 0.38483432, 0.44833523],
        EmbeddingModel::SnowflakeArcticEmbedS => [-0.64302516, -0.63146704, -0.57860875, -0.5829098],
        EmbeddingModel::SnowflakeArcticEmbedSQ => [-0.63687235, -0.6296427, -0.6070188, -0.57358015],
        EmbeddingModel::SnowflakeArcticEmbedM => [-0.16999032, -0.109130904, -0.016444799, -0.108033374],
        EmbeddingModel::SnowflakeArcticEmbedMQ => [-0.15008105, -0.11513549, 0.00008662231, -0.08609233],
        EmbeddingModel::SnowflakeArcticEmbedMLong => [0.20396729, 0.18245143, 0.13489585, 0.15486401],
        EmbeddingModel::SnowflakeArcticEmbedMLongQ => [0.20531628, 0.18564843, 0.14221531, 0.16035447],
        EmbeddingModel::SnowflakeArcticEmbedL => [0.4049112, 0.42825335, 0.46401042, 0.4064963],
        EmbeddingModel::SnowflakeArcticEmbedLQ => [0.40164998, 0.4278314, 0.4612437, 0.40060186],
        _ => panic!("Model {model} not found. If you have just inserted this `EmbeddingModel` variant, please update the expected embeddings."),
    };

    let mismatched_indices = embeddings
        .iter()
        .map(|embedding| embedding.iter().sum::<f32>())
        .zip(expected.iter())
        .enumerate()
        .filter_map(|(i, (sum, &expected))| {
            if (sum - expected).abs() > EPS {
                eprintln!(
                    "Mismatched embeddings for model {model:?} at index {i}: {sum} != {expected} (expected)",
                    model = model,
                    i = i,
                    sum = sum,
                    expected = expected
                );
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if mismatched_indices.is_empty() {
        Ok(())
    } else {
        Err(mismatched_indices)
    }
}

macro_rules! create_embeddings_test {
    (
        name: $name:ident,
        batch_size: $batch_size:expr,
    ) => {
        #[test]
        fn $name() {
            TextEmbedding::list_supported_models()
                .iter()
                .for_each(|supported_model| {
                    let mut model: TextEmbedding = TextEmbedding::try_new(TextInitOptions::new(supported_model.model.clone()))
                    .unwrap();

                    let documents = vec![
                        "Hello, World!",
                        "This is an example passage.",
                        "fastembed-rs is licensed under Apache-2.0",
                        "Some other short text here blah blah blah",
                    ];

                    // Generate embeddings with the default batch size, 256
                    let batch_size = $batch_size;
                    let embeddings = model.embed(documents.clone(), batch_size);

                    if matches!(
                        (batch_size, TextEmbedding::get_quantization_mode(&supported_model.model)),
                        (Some(n), QuantizationMode::Dynamic) if n < documents.len()
                    ) {
                        // For Dynamic quantization, the batch size must be greater than or equal to the number of documents
                        // Otherwise, an error is expected
                        assert!(embeddings.is_err(), "Expected error for batch size < document count for {model} using dynamic quantization.", model=supported_model.model);
                    } else {
                        let embeddings = embeddings.unwrap_or_else(
                            |exc| panic!("Expected embeddings for {model} to be generated successfully: {exc}", model=supported_model.model, exc=exc),
                        );
                        assert_eq!(embeddings.len(), documents.len());

                        for embedding in &embeddings {
                            assert_eq!(embedding.len(), supported_model.dim);
                        }

                        match verify_embeddings(&supported_model.model, &embeddings) {
                            Ok(_) => {}
                            Err(mismatched_indices) => {
                                panic!(
                                    "Mismatched embeddings for model {model}: {sentences:?}",
                                    model = supported_model.model,
                                    sentences = &mismatched_indices
                                        .iter()
                                        .map(|&i| documents[i])
                                        .collect::<Vec<_>>()
                                );
                            }
                        }
                    }
                });
        }

    };
}

create_embeddings_test!(
    name: test_batch_size_default,
    batch_size: None,
);

#[test]
fn test_sparse_embeddings() {
    SparseTextEmbedding::list_supported_models()
        .iter()
        // The inference-free models produce far denser documents and have a query path of their
        // own, they are covered by the `if_splade` suite instead.
        .filter(|supported_model| {
            supported_model
                .additional_files
                .iter()
                .all(|f| f != "idf.json")
        })
        .for_each(|supported_model| {
            let mut model: SparseTextEmbedding =
                SparseTextEmbedding::try_new(SparseInitOptions::new(supported_model.model.clone()))
                    .unwrap();

            let documents = vec![
                "Hello, World!",
                "This is an example passage.",
                "fastembed-rs is licensed under Apache-2.0",
                "Some other short text here blah blah blah",
            ];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(documents.clone(), None).unwrap();

            assert_eq!(embeddings.len(), documents.len());
            embeddings.into_iter().for_each(|embedding| {
                assert!(embedding.values.iter().all(|&v| v > 0.0));
                assert!(embedding.indices.len() < 100);
                assert_eq!(embedding.indices.len(), embedding.values.len());
            });

            // Symmetric models have no separate query representation
            assert!(model.query_embed(documents).is_err());

            // Clear the model cache to avoid running out of space on GitHub Actions.
            if std::env::var("CI").is_ok() {
                clean_cache(supported_model.model_code.clone())
            }
        });
}

#[test]
fn test_user_defined_embedding_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info = TextEmbedding::get_model_info(&EmbeddingModel::AllMiniLML6V2).unwrap();

    TextEmbedding::try_new(TextInitOptions::new(test_model_info.model.clone())).unwrap();

    // Get the directory of the model
    let model_name = test_model_info.model_code.replace('/', "--");
    let model_dir = Path::new(&get_cache_dir()).join(format!("models--{}", model_name));

    // Find the "snapshots" sub-directory
    let snapshots_dir = model_dir.join("snapshots");

    // Get the first sub-directory in snapshots
    let model_files_dir = snapshots_dir
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    // Find the onnx file - it will be any file ending with .onnx
    let onnx_file = std::fs::read(
        model_files_dir
            .read_dir()
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    == "onnx"
            })
            .unwrap()
            .unwrap()
            .path(),
    )
    .expect("Could not read onnx file");

    // Load the tokenizer files
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_files_dir.join("tokenizer.json"))
            .expect("Could not read tokenizer.json"),
        config_file: std::fs::read(model_files_dir.join("config.json"))
            .expect("Could not read config.json"),
        special_tokens_map_file: std::fs::read(model_files_dir.join("special_tokens_map.json"))
            .expect("Could not read special_tokens_map.json"),
        tokenizer_config_file: std::fs::read(model_files_dir.join("tokenizer_config.json"))
            .expect("Could not read tokenizer_config.json"),
    };
    // Create a UserDefinedEmbeddingModel
    let user_defined_model =
        UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

    // Try creating a TextEmbedding instance from the user-defined model
    let mut user_defined_text_embedding = TextEmbedding::try_new_from_user_defined(
        user_defined_model,
        InitOptionsUserDefined::default(),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let embeddings = user_defined_text_embedding
        .embed(documents.clone(), None)
        .unwrap();
    assert_eq!(embeddings.len(), documents.len());
    for embedding in embeddings {
        assert_eq!(embedding.len(), test_model_info.dim);
    }
}

#[test]
fn test_rerank() {
    let test_one_model = |supported_model: &RerankerModelInfo| {
        println!("supported_model: {:?}", supported_model);

        let mut result =
            TextRerank::try_new(RerankInitOptions::new(supported_model.model.clone())).unwrap();

        let documents = vec![
            "hi",
            "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "panda is an animal",
            "i dont know",
            "kind of mammal",
        ];

        let results = result
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap();

        assert_eq!(
            results.len(),
            documents.len(),
            "rerank model {:?} failed",
            supported_model
        );

        let option_a = "panda is an animal";
        let option_b = "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.";

        assert!(
            results[0].document.as_ref().unwrap() == option_a
                || results[0].document.as_ref().unwrap() == option_b
        );
        assert!(
            results[1].document.as_ref().unwrap() == option_a
                || results[1].document.as_ref().unwrap() == option_b
        );
        assert_ne!(
            results[0].document, results[1].document,
            "The top two results should be different"
        );

        // Clear the model cache to avoid running out of space on GitHub Actions.
        clean_cache(supported_model.model_code.clone())
    };
    TextRerank::list_supported_models()
        .iter()
        .for_each(test_one_model);
}

#[ignore]
#[test]
fn test_user_defined_reranking_large_model() {
    // Setup model to download from Hugging Face
    let cache = hf_hub::Cache::new(std::path::PathBuf::from(&fastembed::get_cache_dir()));
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to build API from cache");
    let model_repo = api.model("rozgo/bge-reranker-v2-m3".to_string());

    // Download the onnx model file
    let onnx_file = model_repo.download("model.onnx").unwrap();
    // Onnx model exceeds the limit of 2GB for a file, so we need to download the data file separately
    let _onnx_data_file = model_repo.get("model.onnx.data").unwrap();

    // OnnxSource::File is used to load the onnx file using onnx session builder commit_from_file
    let onnx_source = OnnxSource::File(onnx_file);

    // Load the tokenizer files
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: std::fs::read(model_repo.get("tokenizer.json").unwrap()).unwrap(),
        config_file: std::fs::read(model_repo.get("config.json").unwrap()).unwrap(),
        special_tokens_map_file: std::fs::read(model_repo.get("special_tokens_map.json").unwrap())
            .unwrap(),

        tokenizer_config_file: std::fs::read(model_repo.get("tokenizer_config.json").unwrap())
            .unwrap(),
    };

    let model = UserDefinedRerankingModel::new(onnx_source, tokenizer_files);

    let mut user_defined_reranker =
        TextRerank::try_new_from_user_defined(model, Default::default()).unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

#[test]
fn test_user_defined_reranking_model() {
    // Constitute the model in order to ensure it's downloaded and cached
    let test_model_info: fastembed::RerankerModelInfo =
        TextRerank::get_model_info(&RerankerModel::JINARerankerV1TurboEn);

    TextRerank::try_new(RerankInitOptions::new(test_model_info.model)).unwrap();

    // Get the directory of the model
    let model_name = test_model_info.model_code.replace('/', "--");
    let model_dir = Path::new(&get_cache_dir()).join(format!("models--{}", model_name));

    // Find the "snapshots" sub-directory
    let snapshots_dir = model_dir.join("snapshots");

    // Get the first sub-directory in snapshots
    let model_files_dir = snapshots_dir
        .read_dir()
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();

    // Find the onnx file - it will be any file in ./onnx ending with .onnx
    let onnx_file = std::fs::read(
        model_files_dir
            .join("onnx")
            .read_dir()
            .unwrap()
            .find(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    == "onnx"
            })
            .unwrap()
            .unwrap()
            .path(),
    )
    .expect("Could not read onnx file");

    // Load the tokenizer files
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_files_dir.join("tokenizer.json"))
            .expect("Could not read tokenizer.json"),
        config_file: std::fs::read(model_files_dir.join("config.json"))
            .expect("Could not read config.json"),
        special_tokens_map_file: std::fs::read(model_files_dir.join("special_tokens_map.json"))
            .expect("Could not read special_tokens_map.json"),
        tokenizer_config_file: std::fs::read(model_files_dir.join("tokenizer_config.json"))
            .expect("Could not read tokenizer_config.json"),
    };
    // Create a UserDefinedEmbeddingModel
    let user_defined_model = UserDefinedRerankingModel::new(onnx_file, tokenizer_files);

    // Try creating a TextEmbedding instance from the user-defined model
    let mut user_defined_reranker = TextRerank::try_new_from_user_defined(
        user_defined_model,
        RerankInitOptionsUserDefined::default(),
    )
    .unwrap();

    let documents = vec![
        "Hello, World!",
        "This is an example passage.",
        "fastembed-rs is licensed under Apache-2.0",
        "Some other short text here blah blah blah",
    ];

    // Generate embeddings over documents
    let results = user_defined_reranker
        .rerank("Ciao, Earth!", documents.clone(), false, None)
        .unwrap();

    assert_eq!(results.len(), documents.len());
    assert_eq!(results.first().unwrap().index, 0);
}

fn clean_cache(model_code: String) {
    let repo = Repo::model(model_code);
    let cache_dir = format!("{}/{}", get_cache_dir(), repo.folder_name());
    fs::remove_dir_all(cache_dir).ok();
}

// This is item "test-environment-aeghhgwpe-pro02a" of the [Aguana corpus](http://argumentation.bplaced.net/arguana/data)
fn get_sample_text() -> String {
    let t = include_str!("assets/sample_text.txt");
    t.to_string()
}

#[test]
fn test_batch_size_does_not_change_output() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2).with_max_length(384),
    )
    .expect("Create model successfully");

    let sentences = vec![
        "Books are no more threatened by Kindle than stairs by elevators.",
        "You are who you are when nobody's watching.",
        "An original idea. That can't be too hard. The library must be full of them.",
        "Gaia visited her daughter Mnemosyne, who was busy being unpronounceable.",
        "You can never be overdressed or overeducated.",
        "I don't want to go to heaven. None of my friends are there.",
        "I never travel without my diary. One should always have something sensational to read in the train.",
        "I can resist anything except temptation.",
        "It is absurd to divide people into good and bad. People are either charming or tedious."
    ];

    let single_batch = model
        .embed(sentences.clone(), None)
        .expect("create successfully");
    let small_batch = model
        .embed(sentences, Some(3))
        .expect("create successfully");

    assert_eq!(single_batch.len(), small_batch.len());
    for (a, b) in single_batch.into_iter().zip(small_batch) {
        assert!(a == b, "Expect each sentence embedding are equal.");
    }
}

#[test]
fn test_bgesmallen1point5_match_python_counterpart() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_max_length(384),
    )
    .expect("Create model successfully");

    let text = get_sample_text();

    // baseline is generated in python using Xenova/bge-small-en-v1.5.onnx
    // Tokenize with python SentenceTransformer("BAAI/bge-small-en-v1.5") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("BAAI/bge-small-en-v1.5") default pooling settings.
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        4.208_193_7e-2,
        -2.748_133_2e-2,
        6.742_810_5e-2,
        2.282_790_5e-2,
        4.257_192e-2,
        -4.163_983_5e-2,
        6.814_807_4e-6,
        -9.643_933e-3,
        -3.475_583e-3,
        6.606_272e-2,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-3;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline)
    {
        assert!((expected - actual).abs() < tolerance);
    }
}

#[test]
fn test_allminilml6v2_match_python_counterpart() {
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::AllMiniLML6V2).with_max_length(384),
    )
    .expect("Create model successfully");

    let text = get_sample_text();

    // baseline is generated in python using qdrant/all-mini-lm-l6-v2.onnx
    // Tokenizer with python SentenceTransformer("all-mini-lm-l6-v2") default tokenizer
    // with (text, padding="max_length",max_length=384,truncation=True, return_tensors="np").
    // Normalized and pooled with SentenceTransformer("all-mini-lm-l6-v2") default pooling settings.
    // we only take a 10 items to keep the test file polite
    let baseline: Vec<f32> = vec![
        3.510_517_6e-2,
        1.046_043e-2,
        3.767_998_5e-2,
        7.073_633_4e-2,
        9.097_775e-2,
        -2.507_714_7e-2,
        -2.214_382e-2,
        -1.016_435_9e-2,
        4.660_127_3e-2,
        7.431_366e-2,
    ];

    let embeddings = model.embed(vec![text], None).expect("create successfully");
    let tolerance: f32 = 1e-6;
    for (expected, actual) in embeddings[0]
        .clone()
        .into_iter()
        .take(baseline.len())
        .zip(baseline)
    {
        assert!((expected - actual).abs() < tolerance);
    }
}

// Ref: https://github.com/Anush008/fastembed-rs/issues/171#issue-3209484009
#[test]
fn clip_vit_b32_deterministic_across_calls() {
    let q = "red car";
    let mut fe = TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::ClipVitB32)).unwrap();
    let first = fe.embed(vec![q], None).unwrap()[0].clone();
    for i in 1..100 {
        let vecs = fe.embed(vec![q], None).unwrap();
        assert_eq!(vecs[0], first, "Embedding changed after {i} iterations");
    }
}
