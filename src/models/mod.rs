#[cfg(feature = "onnx")]
use crate::ModelInfo;

#[cfg(feature = "onnx")]
pub mod bgem3;
#[cfg(feature = "onnx")]
pub mod image_embedding;
#[cfg(feature = "onnx")]
pub mod model_info;
#[cfg(feature = "onnx")]
pub mod quantization;
#[cfg(feature = "onnx")]
pub mod reranking;
#[cfg(feature = "onnx")]
pub mod sparse;
#[cfg(feature = "onnx")]
pub mod text_embedding;

#[cfg(feature = "qwen3-text")]
pub mod qwen3;
#[cfg(feature = "qwen3-text")]
pub(crate) mod qwen3_qk_norm_rope;
#[cfg(feature = "qwen3-text")]
pub(crate) mod qwen3_sdpa_bthd;
#[cfg(feature = "qwen3-text")]
pub(crate) mod qwen3_silu_mul;
#[cfg(feature = "qwen3-vl")]
pub mod qwen3_vl;

#[cfg(feature = "nomic-v2-moe")]
pub mod nomic_v2_moe;

#[cfg(feature = "onnx")]
pub trait ModelTrait {
    type Model;
    fn get_model_info(model: &Self::Model) -> Option<&ModelInfo<Self::Model>>;
}

#[cfg(feature = "onnx")]
impl ModelTrait for bgem3::Bgem3Model {
    type Model = Self;

    fn get_model_info(model: &Self) -> Option<&ModelInfo<Self>> {
        static ONCE: std::sync::OnceLock<
            std::collections::HashMap<bgem3::Bgem3Model, ModelInfo<bgem3::Bgem3Model>>,
        > = std::sync::OnceLock::new();
        let map = ONCE.get_or_init(|| {
            bgem3::models_list()
                .into_iter()
                .map(|info| (info.model.clone(), info))
                .collect()
        });
        map.get(model)
    }
}
