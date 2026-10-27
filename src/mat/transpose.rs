//! Row-major to column-major transpose shared by both write paths.
//!
//! MATLAB stores 2-D arrays column-major; the serializer holds them row-major.
//! Both write paths (`ser::emit` and `ser::emit_with_builder`) and the complex
//! value model need the same transpose, so it lives here once. The
//! implementation is cache-tiled (32x32 blocks) to keep both the strided source
//! reads and the destination writes cache-resident on large matrices.

/// Transpose a row-major matrix of shape `[rows, cols]` into column-major.
///
/// Element `(r, c)` at `row_major[r * cols + c]` lands at `out[c * rows + r]`.
fn transpose_2d<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    // `rows * cols` must not wrap: the loop below writes to `c * rows + r`
    // through a raw pointer, so a wrapped `n` would size `out` far below the
    // largest index written. Checking the length against the *checked* product
    // is what makes the SAFETY comments hold; comparing against a wrapping
    // product would agree with itself and prove nothing.
    let n = rows
        .checked_mul(cols)
        .expect("transpose: rows * cols overflows usize");
    assert_eq!(
        row_major.len(),
        n,
        "transpose: source length {} does not match {rows}x{cols} = {n}",
        row_major.len()
    );
    let mut out: Vec<T> = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }

    const BLK: usize = 32;
    let dst = out.as_mut_ptr();
    for cb in (0..cols).step_by(BLK) {
        let c_end = (cb + BLK).min(cols);
        for rb in (0..rows).step_by(BLK) {
            let r_end = (rb + BLK).min(rows);
            for r in rb..r_end {
                let src_row_base = r * cols;
                for c in cb..c_end {
                    let value = row_major[src_row_base + c];
                    // SAFETY: c < cols and r < rows so c*rows + r < cols*rows = n,
                    // and out has capacity n.
                    unsafe {
                        dst.add(c * rows + r).write(value);
                    }
                }
            }
        }
    }
    // SAFETY: every index 0..n was written above (each (r, c) maps to a unique
    // c * rows + r in 0..n).
    unsafe {
        out.set_len(n);
    }
    out
}

/// Transpose a row-major matrix of scalars into column-major order.
#[inline]
pub(crate) fn transpose_scalars<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
    transpose_2d(rows, cols, row_major)
}

/// Transpose a row-major matrix of `(re, im)` pairs into column-major order.
#[inline]
pub(crate) fn transpose_pairs<T: Copy>(
    rows: usize,
    cols: usize,
    row_major: &[(T, T)],
) -> Vec<(T, T)> {
    transpose_2d(rows, cols, row_major)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition the raw-pointer writes have to honour, computed the slow
    /// and obviously-correct way.
    fn transpose_reference<T: Copy>(rows: usize, cols: usize, row_major: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                out.push(row_major[r * cols + c]);
            }
        }
        out
    }

    /// Shapes chosen around the 32-element blocking: smaller than one block,
    /// exactly one block, and straddling a block boundary in each dimension, so
    /// the partial-block `min` arithmetic is exercised in both directions.
    const SHAPES: &[(usize, usize)] = &[
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 7),
        (7, 1),
        (3, 4),
        (32, 32),
        (33, 32),
        (32, 33),
        (33, 47),
        (65, 3),
    ];

    #[test]
    fn scalars_match_the_reference_transpose() {
        for &(rows, cols) in SHAPES {
            let src: Vec<u32> = (0..(rows * cols) as u32).collect();
            assert_eq!(
                transpose_scalars(rows, cols, &src),
                transpose_reference(rows, cols, &src),
                "shape {rows}x{cols}"
            );
        }
    }

    /// Uses `(f64, f64)`, the 16-byte pair `ComplexVec` actually instantiates,
    /// so the raw-pointer scaling is exercised at an element size larger than a
    /// machine word rather than only at 4 bytes.
    #[test]
    fn pairs_match_the_reference_transpose() {
        for &(rows, cols) in SHAPES {
            let src: Vec<(f64, f64)> = (0..rows * cols).map(|i| (i as f64, -(i as f64))).collect();
            assert_eq!(
                transpose_pairs(rows, cols, &src),
                transpose_reference(rows, cols, &src),
                "shape {rows}x{cols}"
            );
        }
    }

    /// An empty matrix must not write through the dangling pointer a
    /// zero-capacity `Vec` hands out.
    #[test]
    fn an_empty_matrix_yields_an_empty_vec() {
        assert!(transpose_scalars::<u8>(0, 0, &[]).is_empty());
        assert!(transpose_pairs::<f64>(4, 0, &[]).is_empty());
    }

    /// `rows` such that `rows * 4` wraps to exactly 4 — the same value as the
    /// source length below. That is what makes this the dangerous shape rather
    /// than merely a large one: a length check written against the wrapping
    /// product agrees with itself and admits the pair, after which the tiled
    /// loop writes far past a 4-element allocation. Derived from `usize::MAX`
    /// rather than written out, so it wraps at either word width.
    const WRAPS_TO_FOUR: usize = usize::MAX / 4 + 2;

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn a_shape_whose_product_wraps_is_refused() {
        let src = vec![0u8; 4];
        let _ = transpose_scalars(WRAPS_TO_FOUR, 4, &src);
    }

    /// The same refusal on the public constructor, which is the safe entry
    /// point a caller can reach it through.
    #[test]
    #[should_panic(expected = "overflows usize")]
    fn a_matrix_whose_product_wraps_is_refused() {
        let _ = crate::mat::Matrix::from_row_major(WRAPS_TO_FOUR, 4, vec![0u8; 4]);
    }
}
