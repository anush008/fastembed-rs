//! Defines the precedence of the output keys in the session outputs.
//!
//! # Note
//!
//! The purpose of this module is to replicate the existing output key selection mechanism
//! in the library. This is an acceptable solution in lieu of a model-specific solution,
//! e.g. reading the output keys from the model file.

/// Enum for defining the key of the output.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum OutputKey {
    #[default]
    OnlyOne,
    ByOrder(usize),
    ByName(&'static str),
}

/// Trait for defining a precedence of keys in the output.
///
/// This defines the order of precedence for selecting the output from the session outputs.
/// By convention, an ONNX model will have at least one output called `last_hidden_state`,
/// which is however not guaranteed. This trait allows the user to define the order of
/// precedence for selecting the output.
///
/// Any [`OutputPrecedence`] should be usable multiple times, and should not consume itself.
/// [`OutputPrecedence::key_precedence`] is called once per batch.
pub trait OutputPrecedence {
    /// Get the precedence of the keys in the output.
    fn key_precedence(&self) -> impl Iterator<Item = &OutputKey>;
}

/// Any slices of [`OutputKey`] can be used as an [`OutputPrecedence`].
impl OutputPrecedence for &[OutputKey] {
    fn key_precedence(&self) -> impl Iterator<Item = &OutputKey> {
        self.iter()
    }
}

impl OutputPrecedence for &OutputKey {
    fn key_precedence(&self) -> impl Iterator<Item = &OutputKey> {
        std::iter::once(*self)
    }
}
