use crate::ModelInfo;

pub mod bgem3;
pub mod image_embedding;
pub mod model_info;
pub mod quantization;
pub mod reranking;
pub mod sparse;
pub mod text_embedding;

#[cfg(feature = "qwen3")]
pub mod qwen3;
#[cfg(feature = "qwen3")]
pub(crate) mod qwen3_qk_norm_rope;
#[cfg(feature = "qwen3")]
pub(crate) mod qwen3_sdpa_bthd;
#[cfg(feature = "qwen3")]
pub mod qwen3_vl;

#[cfg(feature = "nomic-v2-moe")]
pub mod nomic_v2_moe;

pub trait ModelTrait {
    type Model;
    fn get_model_info(model: &Self::Model) -> Option<&ModelInfo<Self::Model>>;
}

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
