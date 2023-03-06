//! Error types for the lux lib

use crate::state::StateError;
use thiserror::Error;

/// Parent type for errors occuring in the lux lib
#[derive(Error, Debug, Clone)]
pub enum LuxError {
    /// General error message
    #[error("{0}")]
    Message(&'static str),

    /// Error originating from the [`state`](crate::state) module
    #[error("State error: {0}")]
    StateError(#[from] StateError),
}

impl From<&'static str> for LuxError {
    fn from(val: &'static str) -> Self {
        Self::Message(val)
    }
}
