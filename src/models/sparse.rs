use std::{fmt::Display, str::FromStr};

use crate::ModelInfo;

/// Sidecar file with the per-token IDF weights used to embed queries without inference.
pub(crate) const IDF_FILE: &str = "idf.json";

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum SparseModel {
    /// prithivida/Splade_PP_en_v1
    #[default]
    SPLADEPPV1,
    /// BAAI/bge-m3
    BGEM3,
    /// opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte
    OpenSearchNeuralSparseDocV3Gte,
}

pub fn models_list() -> Vec<ModelInfo<SparseModel>> {
    vec![
        ModelInfo {
            model: SparseModel::SPLADEPPV1,
            dim: 0,
            description: String::from("Splade sparse vector model for commercial use, v1"),
            model_code: String::from("Qdrant/Splade_PP_en_v1"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: SparseModel::BGEM3,
            dim: 0,
            description: String::from(
                "BGE-M3 sparse embedding model with an 8192-token context, supporting 100+ languages",
            ),
            model_code: String::from("BAAI/bge-m3"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![
                "onnx/model.onnx_data".to_string(),
                "onnx/Constant_7_attr__value".to_string(),
            ],
            output_key: None,
        },
        ModelInfo {
            model: SparseModel::OpenSearchNeuralSparseDocV3Gte,
            dim: 0,
            description: String::from(
                "Inference-free SPLADE model. Documents are expanded with an ONNX encoder, \
                 queries are encoded with a tokenizer and an IDF lookup table only",
            ),
            model_code: String::from("Qdrant/opensearch-neural-sparse-encoding-doc-v3-gte"),
            model_file: String::from("model.onnx"),
            additional_files: vec![IDF_FILE.to_string()],
            output_key: None,
        },
    ]
}

impl Display for SparseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = models_list()
            .into_iter()
            .find(|model| model.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for SparseModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown sparse model: {s}"))
    }
}

impl TryFrom<String> for SparseModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
pub(crate) fn all_variants() -> Vec<SparseModel> {
    fn _exhaustive_guard(m: &SparseModel) {
        match m {
            SparseModel::SPLADEPPV1 => (),
            SparseModel::BGEM3 => (),
            SparseModel::OpenSearchNeuralSparseDocV3Gte => (),
        }
    }
    vec![
        SparseModel::SPLADEPPV1,
        SparseModel::BGEM3,
        SparseModel::OpenSearchNeuralSparseDocV3Gte,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_model_info() {
        let listed: Vec<_> = models_list().into_iter().map(|i| i.model).collect();
        for variant in all_variants() {
            assert!(
                listed.contains(&variant),
                "{variant:?} is missing from models_list(); get_model_info would panic"
            );
        }
    }
}
