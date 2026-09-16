use ndarray::{Array2, ArrayView, Dim, IxDynImpl};

use crate::common::{Error, Result};
use crate::pooling;

use super::{OutputKey, OutputPrecedence};

/// [`SingleBatchOutput`] contains the output of a single batch of inference.
///
/// In the future, each batch will need to deal with its own post-processing, such as
/// pooling etc. This struct should contain all the necessary information for the
/// post-processing to be performed.
pub struct SingleBatchOutput {
    pub outputs: Vec<(String, ort::value::Value)>,
    pub attention_mask_array: Array2<i64>,
}

impl SingleBatchOutput {
    /// Select the output from the session outputs based on the given precedence.
    ///
    /// This returns a view into the tensor, which can be used to perform further
    /// operations.
    pub fn select_output(
        &self,
        precedence: &impl OutputPrecedence,
    ) -> Result<ArrayView<'_, f32, Dim<IxDynImpl>>> {
        let ort_output: &ort::value::Value = precedence
            .key_precedence()
            .find_map(|key| match key {
                // Only select the sole output if and only if there is exactly one.
                OutputKey::OnlyOne => {
                    if self.outputs.len() == 1 {
                        self.outputs.first().map(|(_, v)| v)
                    } else {
                        None
                    }
                }
                OutputKey::ByOrder(idx) => self.outputs.get(*idx).map(|(_, v)| v),
                OutputKey::ByName(name) => {
                    self.outputs.iter().find(|(n, _)| n == name).map(|(_, v)| v)
                }
            })
            .ok_or_else(|| {
                Error::Other(format!(
                    "No suitable output found in the outputs. Available outputs: {:?}",
                    self.outputs.iter().map(|(k, _)| k).collect::<Vec<_>>()
                ))
            })?;

        ort_output.try_extract_array().map_err(Error::from)
    }

    /// Select the output from the session outputs based on the given precedence and pool it.
    ///
    /// This function will pool the output based on the given pooling option, if any.
    pub fn select_and_pool_output(
        &self,
        precedence: &impl OutputPrecedence,
        pooling_opt: Option<pooling::Pooling>,
    ) -> Result<Array2<f32>> {
        let tensor = self.select_output(precedence)?;

        // If there is none pooling, default to cls so as not to break the existing implementations
        // TODO: Consider return output as is to support custom model that has built-in pooling layer:
        // - [] Add model with built-in pooling to the list of supported model in ``models::text_embedding::models_list``
        // - [] Write unit test for new model
        // - [] Update ``pooling::Pooling`` to include None type
        // - [] Change the line below to return output as is
        // - [] Release major version because of breaking changes
        match pooling_opt.unwrap_or_default() {
            pooling::Pooling::Cls => pooling::cls(&tensor),
            pooling::Pooling::Mean => pooling::mean(&tensor, &self.attention_mask_array),
        }
    }
}

/// Container struct with all the outputs from the embedding layer.
///
/// This will contain one [`SingleBatchOutput`] object per batch/inference call.
pub struct EmbeddingOutput {
    batches: Vec<SingleBatchOutput>,
}

impl EmbeddingOutput {
    /// Create a new [`EmbeddingOutput`] from the per-batch session outputs.
    pub fn new(batches: impl IntoIterator<Item = SingleBatchOutput>) -> Self {
        Self {
            batches: batches.into_iter().collect(),
        }
    }

    /// Consume this [`EmbeddingOutput`] and return the raw session outputs.
    ///
    /// This allows the user to perform their custom extractions outside of this
    /// library.
    pub fn into_raw(self) -> Vec<SingleBatchOutput> {
        self.batches
    }

    /// Export the output using the given output transformer.
    ///
    /// The transformer shall be responsible for:
    /// - Selecting the output from the session outputs based on the precedence order,
    /// - Extracting the tensor from the output, then
    /// - Transform the tensor into the desired output.
    ///
    /// The transformer function should take a slice of [`SingleBatchOutput`], and return
    /// the desired output type.
    ///
    /// If any of the steps fail, this function will return an error, including
    /// the session output not containing the expected precedence keys.
    pub fn export_with_transformer<R>(
        &self,
        // TODO: Convert this to a trait alias when it's stabilized.
        // https://github.com/rust-lang/rust/issues/41517
        transformer: impl Fn(&[SingleBatchOutput]) -> Result<R>,
    ) -> Result<R> {
        transformer(&self.batches)
    }
}
