//! Common utilities for shaping and interacting with data

use serde::{
    de::{Deserializer, Error as DeError},
    ser::Serializer,
    Deserialize, Serialize,
};
use std::fmt;

/// Represents an m x n matrix
#[derive(Clone)]
pub struct RectMat<T> {
    r_dim: usize,
    c_dim: usize,
    buffer: Vec<T>,
}

/// Iterator created from [`RectMat::iter`]
pub struct RectMatIter<'a, T> {
    idx: (usize, usize),
    mat: &'a RectMat<T>,
}

/// Iterator created from [`RectMat::enumerate`]
pub struct RectMatEnumerate<'a, T> {
    idx: (usize, usize),
    mat: &'a RectMat<T>,
}

/// Iterator for values in a column created from [`RectMat::iter_col`]
/// or yielded from the iterator created by [`RectMat::iter_cols`]
pub struct RectMatColIter<'a, T> {
    col_idx: usize,
    idx: usize,
    mat: &'a RectMat<T>,
}

/// Iterator for values in a row created from [`RectMat::iter_row`]
/// or yielded from the iterator created by [`RectMat::iter_rows`]
pub struct RectMatRowIter<'a, T> {
    row_idx: usize,
    idx: usize,
    mat: &'a RectMat<T>,
}

/// Iterator yielding [`RectMatColIter`] for each column in the matrix.
///
/// Created from [`RectMat::iter_cols`]
pub struct RectMatColsIter<'a, T> {
    idx: usize,
    mat: &'a RectMat<T>,
}

/// Iterator yielding [`RectMatRowIter`] for each rowumn in the matrix.
///
/// Created from [`RectMat::iter_rows`]
pub struct RectMatRowsIter<'a, T> {
    idx: usize,
    mat: &'a RectMat<T>,
}

impl<'a, T> Iterator for RectMatIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx.1 >= self.mat.c_dim {
            self.idx.1 -= self.mat.c_dim;
            self.idx.0 += 1;
        }
        if self.idx.0 >= self.mat.r_dim {
            return None;
        }
        let rv = Some(self.mat.get_unchecked(self.idx));
        self.idx.1 += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.mat.c_dim == 0 {
            return (0, Some(0));
        }
        let rows = self.idx.0 + (self.idx.1 / self.mat.c_dim);
        let cols = self.idx.1 % self.mat.c_dim;
        let idx = rows * self.mat.c_dim + cols;
        let elem_count = self.mat.c_dim * self.mat.r_dim;
        let bound = elem_count - idx.min(elem_count);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatIter<'a, T> {}

impl<'a, T> Iterator for RectMatEnumerate<'a, T> {
    type Item = ((usize, usize), &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        while self.idx.1 >= self.mat.c_dim {
            self.idx.1 -= self.mat.c_dim;
            self.idx.0 += 1;
        }
        if self.idx.0 >= self.mat.r_dim {
            return None;
        }
        let rv = Some(((self.idx), self.mat.get_unchecked(self.idx)));
        self.idx.1 += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.mat.c_dim == 0 {
            return (0, Some(0));
        }
        let rows = self.idx.0 + (self.idx.1 / self.mat.c_dim);
        let cols = self.idx.1 % self.mat.c_dim;
        let idx = rows * self.mat.c_dim + cols;
        let elem_count = self.mat.c_dim * self.mat.r_dim;
        let bound = elem_count - idx.min(elem_count);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatEnumerate<'a, T> {}

impl<'a, T> Iterator for RectMatColIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mat.r_dim {
            return None;
        }
        let rv = Some(self.mat.get_unchecked((self.idx, self.col_idx)));
        self.idx += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = self.mat.r_dim - self.idx.min(self.mat.r_dim);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatColIter<'a, T> {}

impl<'a, T> Iterator for RectMatRowIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mat.c_dim {
            return None;
        }
        let rv = Some(self.mat.get_unchecked((self.row_idx, self.idx)));
        self.idx += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = self.mat.c_dim - self.idx.min(self.mat.c_dim);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatRowIter<'a, T> {}

impl<'a, T> Iterator for RectMatColsIter<'a, T> {
    type Item = RectMatColIter<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mat.c_dim {
            return None;
        }
        let rv = Some(self.mat.iter_col(self.idx));
        self.idx += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = self.mat.c_dim - self.idx.min(self.mat.c_dim);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatColsIter<'a, T> {}

impl<'a, T> Iterator for RectMatRowsIter<'a, T> {
    type Item = RectMatRowIter<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.mat.r_dim {
            return None;
        }
        let rv = Some(self.mat.iter_row(self.idx));
        self.idx += 1;
        rv
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = self.mat.r_dim - self.idx.min(self.mat.r_dim);
        (bound, Some(bound))
    }
}

impl<'a, T> ExactSizeIterator for RectMatRowsIter<'a, T> {}

impl<T> TryFrom<Vec<Vec<T>>> for RectMat<T> {
    type Error = &'static str;
    fn try_from(val: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        let r_dim = val.len();
        let c_dim = if val.is_empty() { 0 } else { val[0].len() };
        // FIXME(seamooo) dislike double collect
        let buffer: Vec<_> = val
            .into_iter()
            .map(|x| {
                if x.len() == c_dim {
                    Ok(x)
                } else {
                    Err("Cannot convert a ragged vector to a RectMat")
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Self {
            r_dim,
            c_dim,
            buffer,
        })
    }
}

/// This is a convenience implementation
///
/// TODO(seamooo) make the fast
impl<T: Clone> From<RectMat<T>> for Vec<Vec<T>> {
    fn from(val: RectMat<T>) -> Self {
        val.iter_rows().map(|x| x.cloned().collect()).collect()
    }
}

impl<T: Default> RectMat<T> {
    /// Creates a row-major matrix copying the iterator values to the start,
    /// with default values being used to pad the remaining size
    ///
    /// # Note
    ///
    /// The iterator is allowed to produce more elements than are consumed by the
    /// dimensions of the matrix
    pub fn from_iter_padded<I: Iterator<Item = T>>(
        mut iter: I,
        r_size: usize,
        c_size: usize,
    ) -> Self {
        let buffer = (0..r_size * c_size)
            .map(move |_| iter.next().unwrap_or_default())
            .collect::<Vec<_>>();
        Self {
            buffer,
            r_dim: r_size,
            c_dim: c_size,
        }
    }
    /// Creates a new matrix initialized with default values
    pub fn from_dims_default((r_size, c_size): (usize, usize)) -> Self {
        let buffer = (0..r_size * c_size)
            .map(|_| T::default())
            .collect::<Vec<_>>();
        Self {
            buffer,
            r_dim: r_size,
            c_dim: c_size,
        }
    }
}

impl<T> RectMat<T> {
    /// Creates a row-major matrix from the given iterator, chunked into rows of length `c_size`
    ///
    /// # Note
    ///
    /// The iterator must end (i.e. not be cyclical). Additionally it must produce
    /// `n` elements such that `n` is divisible by `c_size`
    pub fn from_chunkable_iter<I: Iterator<Item = T>>(
        iter: I,
        c_size: usize,
    ) -> Result<Self, &'static str> {
        let mut iter = iter.peekable();
        if c_size == 0 && iter.peek().is_some() {
            return Err("Cannot create zero-sized RectMat from non-empty iterator");
        }
        let buffer = iter.collect::<Vec<_>>();
        let r_dim = buffer.len() / c_size;
        if r_dim * c_size != buffer.len() {
            return Err("Cannot create a RectMat from a ragged iterator");
        }
        Ok(Self {
            buffer,
            r_dim,
            c_dim: c_size,
        })
    }
    fn bounds_check(&self, idx: &(usize, usize)) -> Result<(), String> {
        let row_oob = idx.0 >= self.r_dim;
        let col_oob = idx.1 >= self.c_dim;
        if row_oob || col_oob {
            let mut err_strs = vec![];
            if row_oob {
                err_strs.push(format!(
                    "row index {} out of bounds for mat with {} rows",
                    idx.0, self.r_dim
                ));
            }
            if col_oob {
                err_strs.push(format!(
                    "column index {} out of bounds for mat with {} columns",
                    idx.1, self.c_dim
                ))
            }
            return Err(err_strs.join(", "));
        }
        Ok(())
    }

    /// Gets the number of rows for the matrix
    #[inline(always)]
    pub fn r_len(&self) -> usize {
        self.r_dim
    }

    /// Gets the number of columns for the matrix
    #[inline(always)]
    pub fn c_len(&self) -> usize {
        self.c_dim
    }

    /// Gets with bounds check
    #[inline]
    pub fn get(&self, idx: (usize, usize)) -> Result<&T, String> {
        self.bounds_check(&idx)
            .map(|_| &self.buffer[idx.0 * self.c_dim + idx.1])
    }

    /// Gets mutably with bounds check
    #[inline]
    pub fn get_mut(&mut self, idx: (usize, usize)) -> Result<&mut T, String> {
        self.bounds_check(&idx)
            .map(|_| &mut self.buffer[idx.0 * self.c_dim + idx.1])
    }

    /// Gets without performing bounds check
    #[inline(always)]
    pub fn get_unchecked(&self, idx: (usize, usize)) -> &T {
        &self.buffer[idx.0 * self.c_dim + idx.1]
    }

    /// Gets mutably without performing bounds check
    #[inline(always)]
    pub fn get_mut_unchecked(&mut self, idx: (usize, usize)) -> &mut T {
        &mut self.buffer[idx.0 * self.c_dim + idx.1]
    }

    /// Creates an iterator across all values of the matrix in row-major order
    pub fn iter(&self) -> RectMatIter<'_, T> {
        RectMatIter {
            idx: (0, 0),
            mat: self,
        }
    }

    /// Creates an iterator across all values of the matrix in row-major order,
    /// including the row, column indices in the item
    pub fn enumerate(&self) -> RectMatEnumerate<'_, T> {
        RectMatEnumerate {
            idx: (0, 0),
            mat: self,
        }
    }

    /// Creates an iterator across all values of the indexed column
    ///
    /// # Panics
    ///
    /// Function panics if the requested column is out of range
    pub fn iter_col(&self, col_idx: usize) -> RectMatColIter<'_, T> {
        if col_idx >= self.c_dim {
            panic!("Tried to iterate over out of bounds column");
        }
        RectMatColIter {
            col_idx,
            idx: 0,
            mat: self,
        }
    }

    /// Creates an iterator across all values of the indexed row
    ///
    /// # Panics
    ///
    /// Function panics if the requested row is out of range
    pub fn iter_row(&self, row_idx: usize) -> RectMatRowIter<'_, T> {
        if row_idx >= self.r_dim {
            panic!("Tried to iterate over out of bounds row");
        }
        RectMatRowIter {
            row_idx,
            idx: 0,
            mat: self,
        }
    }

    /// Creates an iterator to yield column iterators for all columns
    pub fn iter_cols(&self) -> RectMatColsIter<'_, T> {
        RectMatColsIter { idx: 0, mat: self }
    }

    /// Creates an iterator to yield row iterators for all rows
    pub fn iter_rows(&self) -> RectMatRowsIter<'_, T> {
        RectMatRowsIter { idx: 0, mat: self }
    }

    /// Creates a new RectMap from applying a transform function to every element
    /// of this RectMat
    pub fn map_new<B, F: FnMut(&T) -> B>(&self, func: F) -> RectMat<B> {
        RectMat::<B>::from_chunkable_iter(self.iter().map(func), self.c_dim).unwrap()
    }
}

/// Elements are in row-major order
impl<T> IntoIterator for RectMat<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.buffer.into_iter()
    }
}

impl<T> fmt::Debug for RectMat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("RectMat")
            .field("r_dim", &self.r_dim)
            .field("c_dim", &self.c_dim)
            .finish()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for RectMat<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec_2d = <Vec<Vec<T>> as Deserialize<'de>>::deserialize(deserializer)?;
        RectMat::try_from(vec_2d).map_err(D::Error::custom)
    }
}

impl<T: Serialize + Clone> Serialize for RectMat<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vec_2d = (0..self.r_dim)
            .map(|row_idx| self.iter_row(row_idx).cloned().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        <Vec<Vec<T>> as Serialize>::serialize(&vec_2d, serializer)
    }
}
