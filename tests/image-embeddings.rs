#![cfg(all(feature = "image-models", feature = "hf-hub"))]

use fastembed::{
    ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, ModelInfo, TextEmbedding,
    TextInitOptions,
};

#[test]
fn test_image_embedding_model() {
    let test_one_model = |supported_model: &ModelInfo<ImageEmbeddingModel>| {
        let mut model: ImageEmbedding =
            ImageEmbedding::try_new(ImageInitOptions::new(supported_model.model.clone())).unwrap();

        let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];

        // Generate embeddings with the default batch size, 256
        let embeddings = model.embed(images.clone(), None).unwrap();

        assert_eq!(embeddings.len(), images.len());
    };
    ImageEmbedding::list_supported_models()
        .iter()
        .for_each(test_one_model);
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
    let mut image_model = ImageEmbedding::try_new(ImageInitOptions::new(
        fastembed::ImageEmbeddingModel::NomicEmbedVisionV15,
    ))
    .unwrap();

    // tests/assets/image_0.png is a blue cat
    // tests/assets/image_1.png is a red cat
    let images = vec!["tests/assets/image_0.png", "tests/assets/image_1.png"];
    let image_embeddings = image_model.embed(images.clone(), None).unwrap();
    assert_eq!(image_embeddings.len(), images.len());

    let mut text_model = TextEmbedding::try_new(TextInitOptions::new(
        fastembed::EmbeddingModel::NomicEmbedTextV15,
    ))
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
