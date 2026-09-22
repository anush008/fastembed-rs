#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use image::DynamicImage;
use ndarray::{Array3, ArrayView3};
use ort::{session::Session, value::Value};
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use std::{io::Cursor, path::Path};

use crate::{
    common::{init_session_builder, normalize, Error, Result},
    models::image_embedding::models_list,
    Embedding, ImageEmbeddingModel, ModelInfo,
};

#[cfg(feature = "hf-hub")]
use super::ImageInitOptions;
use super::{
    init::{ImageInitOptionsUserDefined, UserDefinedImageEmbeddingModel},
    utils::Compose,
    ImageEmbedding, ImagePreprocessor, DEFAULT_BATCH_SIZE,
};

impl ImageEmbedding {
    /// Try to generate a new ImageEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: ImageInitOptions) -> Result<Self> {
        let ImageInitOptions {
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
            intra_threads,
            session_config,
        } = options;

        let model_repo = ImageEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let preprocessor_file =
            model_repo
                .get("preprocessor_config.json")
                .map_err(|e| Error::ModelRetrieval {
                    file: "preprocessor_config.json".into(),
                    source: Box::new(e),
                })?;
        let preprocessor = ImagePreprocessor::new(Compose::from_file(preprocessor_file)?);

        let model_file_name = ImageEmbedding::get_model_info(&model_name).model_file;
        let model_file_reference =
            model_repo
                .get(&model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        let session = init_session_builder(execution_providers, intra_threads, session_config)?
            .commit_from_file(model_file_reference)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Create a ImageEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedImageEmbeddingModel,
        options: ImageInitOptionsUserDefined,
    ) -> Result<Self> {
        let ImageInitOptionsUserDefined {
            execution_providers,
            intra_threads,
            session_config,
        } = options;

        let preprocessor = ImagePreprocessor::new(Compose::from_bytes(model.preprocessor_file)?);

        let session = init_session_builder(execution_providers, intra_threads, session_config)?
            .commit_from_memory(&model.onnx_file)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Private method to return an instance
    fn new(preprocessor: ImagePreprocessor, session: Session) -> Self {
        Self {
            preprocessor,
            session,
        }
    }

    /// Return the ImageEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: ImageEmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<ImageEmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from ImageEmbeddingModel
    pub fn get_model_info(model: &ImageEmbeddingModel) -> ModelInfo<ImageEmbeddingModel> {
        ImageEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    /// Return a cloneable CPU preprocessor configured for this image model.
    ///
    /// Use it to preprocess images on worker threads, then pass the resulting arrays to
    /// [`Self::embed_preprocessed`]. It deliberately does not include the mutable ONNX session.
    pub fn preprocessor(&self) -> ImagePreprocessor {
        self.preprocessor.clone()
    }

    /// Method to generate image embeddings for a Vec of image bytes
    pub fn embed_bytes(
        &mut self,
        images: &[&[u8]],
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let output = images
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch
                    .iter()
                    .map(|img| {
                        image::ImageReader::new(Cursor::new(img))
                            .with_guessed_format()?
                            .decode()
                            .map_err(|err| Error::ImageDecode(err.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;

                self.embed_images(inputs)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }

    /// Method to generate image embeddings for a collection of image paths.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<Path>`], such as `Vec<String>`, `Vec<PathBuf>`, `&[&str]`, or `&[&Path]`.
    pub fn embed<S: AsRef<Path> + Send + Sync>(
        &mut self,
        images: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let images = images.as_ref();
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let output = images
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch
                    .iter()
                    .map(|img| {
                        image::ImageReader::open(img)?
                            .decode()
                            .map_err(|err| Error::ImageDecode(err.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;

                self.embed_images(inputs)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }

    /// Embed DynamicImages
    pub fn embed_images(&mut self, imgs: Vec<DynamicImage>) -> Result<Vec<Embedding>> {
        let inputs = imgs
            .into_iter()
            .map(|img| self.preprocessor.preprocess(img))
            .collect::<Result<Vec<Array3<f32>>>>()?;
        self.embed_preprocessed(inputs)
    }

    /// Embed images already prepared by this model's [`ImagePreprocessor`].
    ///
    /// This separates CPU image preprocessing from mutable ONNX Runtime inference, allowing
    /// callers to prepare later batches while the accelerator processes the current batch.
    pub fn embed_preprocessed(&mut self, inputs: Vec<Array3<f32>>) -> Result<Vec<Embedding>> {
        // Extract the batch size
        let inputs_view: Vec<ArrayView3<f32>> = inputs.iter().map(|img| img.view()).collect();
        let pixel_values_array = ndarray::stack(ndarray::Axis(0), &inputs_view)
            .map_err(|e| Error::InvalidShape(e.to_string()))?;

        let input_name = self.session.inputs()[0].name().to_string();
        let session_inputs = ort::inputs![
            input_name => Value::from_array(pixel_values_array)
                .map_err(|e| Error::OrtSession(e.to_string()))?,
        ];

        let outputs = self
            .session
            .run(session_inputs)
            .map_err(|e| Error::OrtSession(e.to_string()))?;

        // Try to get the only output key
        // If multiple, then default to few known keys `image_embeds` and `last_hidden_state`
        let last_hidden_state_key = match outputs.len() {
            1 => vec![outputs
                .keys()
                .next()
                .ok_or_else(|| Error::OutputKeyMissing {
                    key: "<only output>".into(),
                })?],
            _ => vec!["image_embeds", "last_hidden_state"],
        };

        // Extract tensor and handle different dimensionalities
        let (shape, data) = last_hidden_state_key
            .iter()
            .find_map(|&key| {
                outputs
                    .get(key)
                    .and_then(|v| v.try_extract_tensor::<f32>().ok())
            })
            .ok_or_else(|| {
                Error::TensorExtraction("Could not extract tensor from any known output key".into())
            })?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let output_array = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)
            .map_err(|e| Error::InvalidShape(e.to_string()))?;

        let embeddings = match output_array.ndim() {
            3 => {
                // For 3D output [batch_size, sequence_length, hidden_size]
                // Take only the first token, sequence_length[0] (CLS token), embedding
                // and return [batch_size, hidden_size]
                (0..output_array.shape()[0])
                    .map(|batch_idx| {
                        let cls_embedding = output_array
                            .slice(ndarray::s![batch_idx, 0, ..])
                            .to_owned()
                            .to_vec();
                        normalize(&cls_embedding)
                    })
                    .collect()
            }
            2 => {
                // For 2D output [batch_size, hidden_size]
                output_array
                    .outer_iter()
                    .map(|row| {
                        row.as_slice()
                            .ok_or_else(|| {
                                Error::Other("Failed to convert array row to slice".into())
                            })
                            .map(normalize)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            _ => {
                return Err(Error::InvalidShape(format!(
                    "Unexpected output tensor shape: {:?}",
                    output_array.shape()
                )))
            }
        };

        Ok(embeddings)
    }
}
