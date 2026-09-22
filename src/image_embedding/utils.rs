use crate::common::{Error, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array, Array3};
use std::ops::{Div, Sub};
#[cfg(feature = "hf-hub")]
use std::{fs::read_to_string, path::Path};

pub enum TransformData {
    Image(DynamicImage),
    NdArray(Array3<f32>),
}

impl TransformData {
    pub fn image(self) -> Result<DynamicImage> {
        match self {
            TransformData::Image(img) => Ok(img),
            _ => Err(Error::ImageTransform("TransformData convert error".into())),
        }
    }

    pub fn array(self) -> Result<Array3<f32>> {
        match self {
            TransformData::NdArray(array) => Ok(array),
            _ => Err(Error::ImageTransform("TransformData convert error".into())),
        }
    }
}

pub(crate) type ResizeFn =
    dyn Fn(DynamicImage, u32, u32, FilterType) -> Result<DynamicImage> + Send + Sync;

fn default_resize(
    image: DynamicImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> Result<DynamicImage> {
    Ok(image.resize_exact(width, height, filter))
}

pub trait Transform: Send + Sync {
    fn transform(&self, data: TransformData) -> Result<TransformData> {
        self.transform_with_resize(data, &default_resize)
    }

    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData>;
}

struct ConvertToRGB;

impl Transform for ConvertToRGB {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let image = data.image()?;
        let image = image.into_rgb8().into();
        Ok(TransformData::Image(image))
    }
}

/// The `size` entry of a `preprocessor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeSize {
    ShortestEdge(u32),
    Exact { width: u32, height: u32 },
}

pub struct Resize {
    pub size: ResizeSize,
    pub resample: FilterType,
}

impl Resize {
    fn target_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        match self.size {
            ResizeSize::Exact { width, height } => (width, height),
            ResizeSize::ShortestEdge(edge) => {
                let (short, long) = if width <= height {
                    (width, height)
                } else {
                    (height, width)
                };
                if short == edge {
                    return (width, height);
                }
                let scaled_long = ((edge as f64 * long as f64) / short as f64) as u32;
                if width <= height {
                    (edge, scaled_long.max(1))
                } else {
                    (scaled_long.max(1), edge)
                }
            }
        }
    }
}

impl Transform for Resize {
    fn transform_with_resize(
        &self,
        data: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        let image = data.image()?;
        let (width, height) = image.dimensions();
        let (new_width, new_height) = self.target_dimensions(width, height);
        if (new_width, new_height) == (width, height) {
            return Ok(TransformData::Image(image));
        }
        let image = resize(image, new_width, new_height, self.resample)?;
        Ok(TransformData::Image(image))
    }
}

pub struct CenterCrop {
    pub size: (u32, u32),
}

impl Transform for CenterCrop {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let mut image = data.image()?;
        let (mut origin_width, mut origin_height) = image.dimensions();
        let (crop_width, crop_height) = self.size;
        if origin_width >= crop_width && origin_height >= crop_height {
            // cropped area is within image boundaries
            let x = (origin_width - crop_width) / 2;
            let y = (origin_height - crop_height) / 2;
            let image = image.crop_imm(x, y, crop_width, crop_height);
            Ok(TransformData::Image(image))
        } else {
            if origin_width > crop_width || origin_height > crop_height {
                let (new_width, new_height) =
                    (origin_width.min(crop_width), origin_height.min(crop_height));
                let (x, y) = if origin_width > crop_width {
                    ((origin_width - crop_width) / 2, 0)
                } else {
                    (0, (origin_height - crop_height) / 2)
                };
                image = image.crop_imm(x, y, new_width, new_height);
                (origin_width, origin_height) = image.dimensions();
            }
            let mut pixels_array =
                Array3::zeros((3usize, crop_height as usize, crop_width as usize));
            let offset_x = (crop_width - origin_width) / 2;
            let offset_y = (crop_height - origin_height) / 2;
            // whc -> chw
            for (x, y, pixel) in image.to_rgb8().enumerate_pixels() {
                pixels_array[[0, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[0] as f32;
                pixels_array[[1, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[1] as f32;
                pixels_array[[2, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[2] as f32;
            }
            Ok(TransformData::NdArray(pixels_array))
        }
    }
}

struct PILToNDarray;

impl Transform for PILToNDarray {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        match data {
            TransformData::Image(image) => {
                let image = image.to_rgb8();
                let (width, height) = image.dimensions();
                // whc -> chw
                let mut pixels_array = Array3::zeros((3usize, height as usize, width as usize));
                for (x, y, pixel) in image.enumerate_pixels() {
                    pixels_array[[0, y as usize, x as usize]] = pixel[0] as f32;
                    pixels_array[[1, y as usize, x as usize]] = pixel[1] as f32;
                    pixels_array[[2, y as usize, x as usize]] = pixel[2] as f32;
                }
                Ok(TransformData::NdArray(pixels_array))
            }
            ndarray => Ok(ndarray),
        }
    }
}

pub struct Rescale {
    pub scale: f32,
}

impl Transform for Rescale {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let array = data.array()?;
        let array = array * self.scale;
        Ok(TransformData::NdArray(array))
    }
}

pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Transform for Normalize {
    fn transform_with_resize(&self, data: TransformData, _: &ResizeFn) -> Result<TransformData> {
        let array = data.array()?;
        let mean = Array::from_vec(self.mean.clone())
            .into_shape_with_order((3, 1, 1))
            .map_err(|e| Error::InvalidShape(format!("Failed to reshape mean array: {e}")))?;
        let std = Array::from_vec(self.std.clone())
            .into_shape_with_order((3, 1, 1))
            .map_err(|e| Error::InvalidShape(format!("Failed to reshape std array: {e}")))?;

        let shape = array.shape().to_vec();
        match shape.as_slice() {
            [c, h, w] => {
                let mean_broadcast = mean.broadcast((*c, *h, *w)).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Failed to broadcast mean array to shape {:?}",
                        (*c, *h, *w)
                    ))
                })?;
                let std_broadcast = std.broadcast((*c, *h, *w)).ok_or_else(|| {
                    Error::InvalidShape(format!(
                        "Failed to broadcast std array to shape {:?}",
                        (*c, *h, *w)
                    ))
                })?;
                let array_normalized = array.sub(mean_broadcast).div(std_broadcast);
                Ok(TransformData::NdArray(array_normalized))
            }
            _ => Err(Error::ImageTransform(
                "Transformer convert error. Normalize operator got error shape.".into(),
            )),
        }
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    #[cfg(feature = "hf-hub")]
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let config = serde_json::from_str(&content)
            .map_err(|e| Error::PreprocessorConfig(format!("Invalid preprocessor JSON: {e}")))?;
        load_preprocessor(config)
    }

    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Compose> {
        let config = serde_json::from_slice(bytes.as_ref())
            .map_err(|e| Error::PreprocessorConfig(format!("Invalid preprocessor JSON: {e}")))?;
        load_preprocessor(config)
    }

    pub(crate) fn preprocess_image(&self, image: DynamicImage) -> Result<Array3<f32>> {
        self.transform(TransformData::Image(image))?.array()
    }

    pub(crate) fn preprocess_image_with_resize(
        &self,
        image: DynamicImage,
        resize: &ResizeFn,
    ) -> Result<Array3<f32>> {
        self.transform_with_resize(TransformData::Image(image), resize)?
            .array()
    }
}

impl Transform for Compose {
    fn transform_with_resize(
        &self,
        mut image: TransformData,
        resize: &ResizeFn,
    ) -> Result<TransformData> {
        for transform in &self.transforms {
            image = transform.transform_with_resize(image, resize)?;
        }
        Ok(image)
    }
}

fn load_preprocessor(config: serde_json::Value) -> Result<Compose> {
    let mut transformers: Vec<Box<dyn Transform>> = vec![];
    transformers.push(Box::new(ConvertToRGB));

    let mode = config["image_processor_type"]
        .as_str()
        .unwrap_or("CLIPImageProcessor");
    match mode {
        "CLIPImageProcessor" | "BitImageProcessor" => {
            if config["do_resize"].as_bool().unwrap_or(false) {
                transformers.push(Box::new(Resize {
                    size: parse_resize_size(&config["size"])?,
                    resample: FilterType::CatmullRom,
                }));
            }

            if config["do_center_crop"].as_bool().unwrap_or(false) {
                transformers.push(Box::new(CenterCrop {
                    size: parse_crop_size(&config["crop_size"])?,
                }));
            }
        }
        "ConvNextFeatureExtractor" => {
            let shortest_edge = config["size"]["shortest_edge"].as_u64().ok_or_else(|| {
                Error::PreprocessorConfig(
                    "Size dictionary must contain 'shortest_edge' key.".into(),
                )
            })? as u32;
            let crop_pct = config["crop_pct"].as_f64().unwrap_or(0.875);
            if shortest_edge < 384 {
                let resize_shortest_edge = shortest_edge as f64 / crop_pct;
                transformers.push(Box::new(Resize {
                    size: ResizeSize::ShortestEdge(resize_shortest_edge as u32),
                    resample: FilterType::CatmullRom,
                }));
                transformers.push(Box::new(CenterCrop {
                    size: (shortest_edge, shortest_edge),
                }))
            } else {
                transformers.push(Box::new(Resize {
                    size: ResizeSize::Exact {
                        width: shortest_edge,
                        height: shortest_edge,
                    },
                    resample: FilterType::CatmullRom,
                }));
            }
        }
        mode => {
            return Err(Error::PreprocessorConfig(format!(
                "Preprocessor {mode} is not supported"
            )));
        }
    }

    transformers.push(Box::new(PILToNDarray));

    if config["do_rescale"].as_bool().unwrap_or(true) {
        let rescale_factor = config["rescale_factor"].as_f64().unwrap_or(1.0f64 / 255.0);
        transformers.push(Box::new(Rescale {
            scale: rescale_factor as f32,
        }));
    }

    if config["do_normalize"].as_bool().unwrap_or(false) {
        let mean = config["image_mean"]
            .as_array()
            .ok_or_else(|| Error::PreprocessorConfig("image_mean must be contained".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or_else(|| Error::PreprocessorConfig("image_mean must be float".into()))
            })
            .collect::<Result<Vec<f32>>>()?;
        let std = config["image_std"]
            .as_array()
            .ok_or_else(|| Error::PreprocessorConfig("image_std must be contained".into()))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or_else(|| Error::PreprocessorConfig("image_std must be float".into()))
            })
            .collect::<Result<Vec<f32>>>()?;
        transformers.push(Box::new(Normalize { mean, std }));
    }

    Ok(Compose::new(transformers))
}

fn parse_resize_size(size: &serde_json::Value) -> Result<ResizeSize> {
    if let Some(shortest_edge) = size["shortest_edge"].as_u64() {
        return Ok(ResizeSize::ShortestEdge(shortest_edge as u32));
    }
    match (size["width"].as_u64(), size["height"].as_u64()) {
        (Some(width), Some(height)) => Ok(ResizeSize::Exact {
            width: width as u32,
            height: height as u32,
        }),
        _ => Err(Error::PreprocessorConfig(
            "Size must contain either 'shortest_edge' or 'height' and 'width'.".into(),
        )),
    }
}

/// `crop_size` as `(width, height)`.
fn parse_crop_size(crop_size: &serde_json::Value) -> Result<(u32, u32)> {
    if let Some(size) = crop_size.as_u64() {
        return Ok((size as u32, size as u32));
    }
    if crop_size.is_object() {
        let height = crop_size["height"].as_u64().ok_or_else(|| {
            Error::PreprocessorConfig("crop_size height must be contained".into())
        })?;
        let width = crop_size["width"]
            .as_u64()
            .ok_or_else(|| Error::PreprocessorConfig("crop_size width must be contained".into()))?;
        return Ok((width as u32, height as u32));
    }
    Err(Error::PreprocessorConfig(format!(
        "Invalid crop size: {crop_size:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn image(width: u32, height: u32) -> TransformData {
        TransformData::Image(DynamicImage::ImageRgb8(RgbImage::from_pixel(
            width,
            height,
            image::Rgb([255, 0, 0]),
        )))
    }

    #[test]
    fn shortest_edge_resize_preserves_aspect_ratio() {
        let resize = Resize {
            size: ResizeSize::ShortestEdge(224),
            resample: FilterType::Nearest,
        };
        let out = resize.transform(image(400, 200)).unwrap().image().unwrap();
        assert_eq!(out.dimensions(), (448, 224));
        let out = resize.transform(image(200, 400)).unwrap().image().unwrap();
        assert_eq!(out.dimensions(), (224, 448));
    }

    #[test]
    fn exact_resize_keeps_width_and_height_apart() {
        let resize = Resize {
            size: ResizeSize::Exact {
                width: 320,
                height: 240,
            },
            resample: FilterType::Nearest,
        };
        let out = resize.transform(image(100, 100)).unwrap().image().unwrap();
        assert_eq!(out.dimensions(), (320, 240));
    }

    #[test]
    fn center_crop_pads_smaller_non_square_images_as_chw() {
        let crop = CenterCrop { size: (224, 100) };
        let out = crop.transform(image(50, 40)).unwrap().array().unwrap();
        assert_eq!(out.dim(), (3, 100, 224));
        assert_eq!(out[[0, 50, 112]], 255.0);
        assert_eq!(out[[0, 0, 0]], 0.0);
    }

    #[test]
    fn clip_config_builds_shortest_edge_pipeline() {
        let config = serde_json::json!({
            "image_processor_type": "CLIPImageProcessor",
            "do_resize": true,
            "size": {"shortest_edge": 224},
            "do_center_crop": true,
            "crop_size": 224,
            "do_rescale": true,
            "do_normalize": true,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        });
        let compose = load_preprocessor(config).unwrap();
        let out = compose.transform(image(640, 480)).unwrap().array().unwrap();
        assert_eq!(out.dim(), (3, 224, 224));
        assert!((out[[0, 100, 100]] - 1.0).abs() < 1e-6);
        assert!((out[[1, 100, 100]] + 1.0).abs() < 1e-6);
    }
}
