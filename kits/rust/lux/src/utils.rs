use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Serialize, Deserialize)]
pub struct Players<T> {
    player_0: T,
    player_1: T,
}

pub(crate) struct OpaqueRectArrDbg<'a, T>(pub &'a Vec<Vec<T>>);

impl<'a, T> fmt::Debug for OpaqueRectArrDbg<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let second_dim = self.0.iter().next().map(|arr| arr.len()).unwrap_or(0);
        f.debug_tuple("Array")
            .field(&self.0.len())
            .field(&second_dim)
            .finish()
    }
}
