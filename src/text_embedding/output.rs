//! Output types and functions for the [`TextEmbedding`] model.
//!
use crate::{
    common::{normalize, Embedding, Error, Result},
    output::{OutputKey, OutputPrecedence, SingleBatchOutput},
    pooling::Pooling,
};

#[cfg(doc)]
use super::TextEmbedding;

/// The default output precedence for the TextEmbedding model.
pub const OUTPUT_TYPE_PRECEDENCE: &[OutputKey] = &[
    OutputKey::OnlyOne,
    OutputKey::ByName("text_embeds"),
    OutputKey::ByName("last_hidden_state"),
    OutputKey::ByName("sentence_embedding"),
    // Better not to expose this unless the user explicitly asks for it.
    // OutputKey::ByName("token_embeddings"),
];

/// Generates the default array transformer for the [`TextEmbedding`] model using the
/// provided output precedence.
///
pub fn transformer_with_precedence(
    output_precedence: impl OutputPrecedence,
    pooling: Option<Pooling>,
) -> impl Fn(&[SingleBatchOutput]) -> Result<Vec<Embedding>> {
    move |batches| {
        // Not using `par_iter` here: the operations here is probably not
        // computationally expensive enough to warrant spinning up costs of the threads.
        batches
            .iter()
            .map(|batch| {
                batch
                    .select_and_pool_output(&output_precedence, pooling.clone())
                    .and_then(|array| {
                        array
                            .rows()
                            .into_iter()
                            .map(|row| {
                                row.as_slice()
                                    .ok_or_else(|| {
                                        Error::Other("Failed to convert array row to slice".into())
                                    })
                                    .map(normalize)
                            })
                            .collect::<Result<Vec<Embedding>>>()
                    })
            })
            .try_fold(Vec::new(), |mut acc, res| {
                acc.extend(res?);
                Ok(acc)
            })
    }
}
