#![cfg(all(feature = "image-models", feature = "hf-hub"))]

use std::sync::OnceLock;

use ort::environment::Environment;

use fastembed::{
    ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, ModelInfo, TextEmbedding,
    TextInitOptions,
};

fn environment() -> Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| ort::init().build().unwrap())
        .clone()
}

#[test]
fn test_image_embedding_model() {
    let test_one_model = |supported_model: &ModelInfo<ImageEmbeddingModel>| {
        let mut model: ImageEmbedding = ImageEmbedding::try_new(
            ImageInitOptions::new(supported_model.model.clone()).with_environment(environment()),
        )
        .unwrap();

        let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];

        // Generate embeddings with the default batch size, 256
        let embeddings = model.embed(images.clone(), None).unwrap();

        assert_eq!(embeddings.len(), images.len());
        for embedding in &embeddings {
            assert_eq!(embedding.len(), supported_model.dim);
        }
        if supported_model.model == ImageEmbeddingModel::ClipVitB32 {
            assert_clip_matches_python_counterpart(&embeddings);
        }
    };
    ImageEmbedding::list_supported_models()
        .iter()
        .for_each(test_one_model);
}

/// Compares the ClipVitB32 embeddings of the two test images with the vectors that fastembed
/// 0.9.1 (Python) returns. `image_1.png` has the size of the model input. `image_0.png` is
/// resized first, and PIL rounds the resized pixels differently from the image crate: the
/// distance is 0.020 with the bicubic filter of the model, and 0.068 with the bilinear one.
fn assert_clip_matches_python_counterpart(embeddings: &[Vec<f32>]) {
    let baseline: Vec<Vec<f32>> =
        serde_json::from_str(include_str!("assets/clip_vit_b32_images.json")).unwrap();
    assert_eq!(embeddings.len(), baseline.len());
    for ((embedding, expected), tolerance) in embeddings.iter().zip(&baseline).zip([3e-2, 1e-4]) {
        assert_eq!(embedding.len(), expected.len());
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        let distance = embedding
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(distance < tolerance, "distance {distance}");
    }
}

#[test]
#[ignore]
fn test_nomic_embed_vision_v1_5() {
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b)
    }

    fn cosine_similarity_matrix(
        embeddings_a: &[Vec<f32>],
        embeddings_b: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        embeddings_a
            .iter()
            .map(|a| {
                embeddings_b
                    .iter()
                    .map(|b| cosine_similarity(a, b))
                    .collect()
            })
            .collect()
    }

    // Test the NomicEmbedVisionV15 model specifically because it outputs a 3D tensor with a different
    // output key ('last_hidden_state') compared to other models. This test ensures our tensor extraction
    // logic can handle both standard output keys and this model's specific naming convention.
    let mut image_model = ImageEmbedding::try_new(
        ImageInitOptions::new(fastembed::ImageEmbeddingModel::NomicEmbedVisionV15)
            .with_environment(environment()),
    )
    .unwrap();

    // tests/assets/image_0.png is a blue cat
    // tests/assets/image_1.png is a red cat
    let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];
    let image_embeddings = image_model.embed(images.clone(), None).unwrap();
    assert_eq!(image_embeddings.len(), images.len());

    let mut text_model = TextEmbedding::try_new(
        TextInitOptions::new(fastembed::EmbeddingModel::NomicEmbedTextV15)
            .with_environment(environment()),
    )
    .unwrap();
    let texts = vec!["green cat", "blue cat", "red cat", "yellow cat", "dog"];
    let text_embeddings = text_model.embed(texts.clone(), None).unwrap();

    // Generate similarity matrix
    let similarity_matrix = cosine_similarity_matrix(&text_embeddings, &image_embeddings);
    // Print the similarity matrix with text labels
    for (i, row) in similarity_matrix.iter().enumerate() {
        println!("{}: {:?}", texts[i], row);
    }

    assert_eq!(text_embeddings.len(), texts.len());
    assert_eq!(text_embeddings[0].len(), 768);
}
