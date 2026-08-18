//! Internal wordwise operations on dense bit vectors and matrices.

use bitvec::prelude::*;
use std::cmp::Ordering;

/// Returns whether every set bit in `left` is also set in `right`.
pub(crate) fn is_subset(left: &BitVec, right: &BitVec) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let bits_per_word = usize::BITS as usize;
    let full_words = left.len() / bits_per_word;
    let remainder = left.len() % bits_per_word;
    let left_words = left.as_raw_slice();
    let right_words = right.as_raw_slice();

    if left_words[..full_words]
        .iter()
        .zip(&right_words[..full_words])
        .any(|(&left_word, &right_word)| left_word & !right_word != 0)
    {
        return false;
    }
    if remainder == 0 {
        return true;
    }

    let mask = (1usize << remainder) - 1;
    left_words[full_words] & !right_words[full_words] & mask == 0
}

/// Compares equally sized bit vectors by containment of their set bits.
pub(crate) fn set_partial_cmp(left: &BitVec, right: &BitVec) -> Option<Ordering> {
    if left == right {
        Some(Ordering::Equal)
    } else if is_subset(left, right) {
        Some(Ordering::Less)
    } else if is_subset(right, left) {
        Some(Ordering::Greater)
    } else {
        None
    }
}

/// Applies the bitwise union of `right` to `left` in place.
pub(crate) fn union_assign(left: &mut BitVec, right: &BitVec) {
    debug_assert_eq!(left.len(), right.len());
    for (left_word, &right_word) in left.as_raw_mut_slice().iter_mut().zip(right.as_raw_slice()) {
        *left_word |= right_word;
    }
}

/// Returns the bitwise intersection of equally sized vectors.
pub(crate) fn intersection(left: &BitVec, right: &BitVec) -> BitVec {
    debug_assert_eq!(left.len(), right.len());
    let mut intersection = left.clone();
    for (intersection_word, &right_word) in intersection
        .as_raw_mut_slice()
        .iter_mut()
        .zip(right.as_raw_slice())
    {
        *intersection_word &= right_word;
    }
    intersection
}

/// Removes every bit set in `right` from `left` in place.
pub(crate) fn difference_assign(left: &mut BitVec, right: &BitVec) {
    debug_assert_eq!(left.len(), right.len());
    for (left_word, &right_word) in left.as_raw_mut_slice().iter_mut().zip(right.as_raw_slice()) {
        *left_word &= !right_word;
    }
}

/// Transposes a rectangular bit matrix whose rows all have the same length.
pub(crate) fn transpose(rows: &[BitVec]) -> Vec<BitVec> {
    let column_count = rows.first().map_or(0, BitVec::len);
    debug_assert!(rows.iter().all(|row| row.len() == column_count));

    let mut transpose = vec![BitVec::repeat(false, rows.len()); column_count];
    for (row_id, row) in rows.iter().enumerate() {
        for column_id in row.iter_ones() {
            transpose[column_id].set(row_id, true);
        }
    }
    transpose
}
