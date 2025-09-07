use bitvec::prelude::*;

pub fn is_subset(a: &BitVec, b: &BitVec) -> bool {
    if a.len() != b.len() {
        return false; // Different lengths, cannot be subset
    }
    let mut temp = a.clone();
    temp &= b;
    temp == *a
}

/// Determines if any row of the binary matrix x is an intersection of other rows
/// If so, returns the index of the first such row
/// Else, returns None
/// ASSUMES x is a matrix, i.e. each bitvec in x has the same length.
pub fn redundant_row(x: &Vec<BitVec>) -> Option<usize> {
    for i in 0..x.len() {
        let mut best_approx = BitVec::repeat(true, x[0].len());
        for j in 0..x.len() {
            if i != j && is_subset(&x[i], &x[j]) {
                best_approx &= &x[j];
            }
        }
        if best_approx == x[i] {
            // Row i is the intersection of other rows
            return Some(i);
        }
    }
    None
}

// pub fn one_hot(n: usize, i: usize) -> BitVec {
//     let mut result = BitVec::repeat(false, n);
//     result.set(i, true);
//     result
// }
