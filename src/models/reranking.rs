use std::{fmt::Display, str::FromStr};

use crate::RerankerModelInfo;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    #[default]
    BGERerankerBase,
    /// rozgo/bge-reranker-v2-m3
    BGERerankerV2M3,
    /// jinaai/jina-reranker-v1-turbo-en
    JINARerankerV1TurboEn,
    /// jinaai/jina-reranker-v2-base-multilingual
    JINARerankerV2BaseMultilingual,
}

impl RerankerModel {
    #[allow(non_upper_case_globals)]
    #[deprecated(note = "use `JINARerankerV2BaseMultilingual` instead")]
    pub const JINARerankerV2BaseMultiligual: Self = Self::JINARerankerV2BaseMultilingual;
}

pub fn reranker_model_list() -> Vec<RerankerModelInfo> {
    let reranker_model_list = vec![
        RerankerModelInfo {
            model: RerankerModel::BGERerankerBase,
            description: String::from("reranker model for English and Chinese"),
            model_code: String::from("BAAI/bge-reranker-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::BGERerankerV2M3,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("rozgo/bge-reranker-v2-m3"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx.data")],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV1TurboEn,
            description: String::from("reranker model for English"),
            model_code: String::from("jinaai/jina-reranker-v1-turbo-en"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultilingual,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
    ];
    reranker_model_list
}

impl Display for RerankerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = reranker_model_list()
            .into_iter()
            .find(|model| model.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for RerankerModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        reranker_model_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown reranker model: {s}"))
    }
}

impl TryFrom<String> for RerankerModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
pub(crate) fn all_variants() -> Vec<RerankerModel> {
    fn _exhaustive_guard(m: &RerankerModel) {
        match m {
            RerankerModel::BGERerankerBase => (),
            RerankerModel::BGERerankerV2M3 => (),
            RerankerModel::JINARerankerV1TurboEn => (),
            RerankerModel::JINARerankerV2BaseMultilingual => (),
        }
    }
    vec![
        RerankerModel::BGERerankerBase,
        RerankerModel::BGERerankerV2M3,
        RerankerModel::JINARerankerV1TurboEn,
        RerankerModel::JINARerankerV2BaseMultilingual,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_model_info() {
        let listed: Vec<_> = reranker_model_list().into_iter().map(|i| i.model).collect();
        for variant in all_variants() {
            assert!(
                listed.contains(&variant),
                "{variant:?} is missing from reranker_model_list(); get_model_info would panic"
            );
        }
    }
}
