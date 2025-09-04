//! This module implements Formal Concept Analysis (FCA) structures and algorithms.
//! It includes definitions for FormalContext and FormalConcept, along with methods for
//! computing intents and extents, checking for reduced contexts, and validating concepts.
//! The implementation uses bit vectors for efficient representation of relations.

use bitvec::prelude::*;
use std::sync::Arc;

fn is_subset(a: &BitVec, b: &BitVec) -> bool {
    if a.len() != b.len() {
        return false; // Different lengths, cannot be subset
    }
    let mut temp = a.clone();
    temp &= b;
    temp == *a
}

// Determines if any row of the binary matrix x is an intersection of other rows
// If so, returns the index of the first such row
// Else, returns None
// ASSUMES x is a matrix, i.e. each bitvec in x has the same length.
fn redundant_row(x: &Vec<BitVec>) -> Option<usize> {
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

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FormalContext<A = String, B = String> {
    objects: Vec<A>,                  // A subset of objects is an extent
    attributes: Vec<B>,               // A subset of attributes is an intent
    relation: Vec<BitVec>,            // The intent of each object
    relation_transposed: Vec<BitVec>, // The extent of each attribute
}

impl<A, B> FormalContext<A, B> {
    /// Checks that the formal context is well-formed -- in memory, both
    /// the relation and its transpose are stored. This function makes
    /// sure they are consistent with each other.
    pub fn validate(&self) -> bool {
        // Check if relation is the transpose of relation_transposed
        for i in 0..self.objects.len() {
            for j in 0..self.attributes.len() {
                if self.relation[i][j] != self.relation_transposed[j][i] {
                    return false; // Relation and its transpose do not match
                }
            }
        }
        true
    }
    /// Creates a new formal context where no objects have any attributes.
    pub fn zero_context(objects: Vec<A>, attributes: Vec<B>) -> Self {
        Self {
            relation: vec![BitVec::repeat(false, attributes.len()); objects.len()],
            relation_transposed: vec![BitVec::repeat(false, objects.len()); attributes.len()],
            objects,
            attributes,
        }
    }
    /// Modifies the relation at the given indices.
    pub fn modify_relation_idx(&mut self, obj_idx: usize, attr_idx: usize, value: bool) {
        self.relation[obj_idx].set(attr_idx, value);
        self.relation_transposed[attr_idx].set(obj_idx, value);
    }
    /// Given an extent (a set of objects), induce its intent (the common attributes of those objects).
    pub fn induce_r(&self, extent: &BitVec) -> BitVec {
        let mut intent = BitVec::repeat(true, self.attributes.len());
        for obj in extent.iter_ones() {
            intent &= &self.relation[obj];
        }
        intent
    }
    pub fn induce_l(&self, intent: &BitVec) -> BitVec {
        let mut extent = BitVec::repeat(true, self.objects.len());
        for attr in intent.iter_ones() {
            extent &= &self.relation_transposed[attr];
        }
        extent
    }
    /// Check if the context is reduced, meaning no row or column of the relation is the intersection of other rows or columns (resp).
    /// Note that if a row is the intersection of other rows, then it is the intersection of specifically those rows which are ≥ it.
    pub fn is_reduced(&self) -> bool {
        redundant_row(&self.relation).is_none()
            && redundant_row(&self.relation_transposed).is_none()
    }
    /// Modifies in place! Removes redundant rows and columns to obtain a reduced context
    pub fn reduce(&mut self) {
        while let Some(i) = redundant_row(&self.relation) {
            self.objects.remove(i);
            self.relation.remove(i);
        }
        while let Some(i) = redundant_row(&self.relation_transposed) {
            self.attributes.remove(i);
            self.relation_transposed.remove(i);
        }
    }
}

impl<A: Clone> FormalContext<A, A> {
    /// Creates the 'contranomial scale' on the given objects, where each object has all attributes except itself.
    pub fn contranomial_scale(objects: Vec<A>) -> Self {
        let mut relation = vec![BitVec::repeat(true, objects.len()); objects.len()];
        for i in 0..objects.len() {
            relation[i].set(i, false);
        }
        Self {
            attributes: objects.clone(),
            relation_transposed: relation.clone(),
            relation,
            objects,
        }
    }
}

impl<A: Eq, B: Eq> FormalContext<A, B> {
    pub fn modify_relation(&mut self, obj: &A, attr: &B, value: bool) {
        let obj_idx = self
            .objects
            .iter()
            .position(|o| o == obj)
            .expect("Object not found in context");
        let attr_idx = self
            .attributes
            .iter()
            .position(|a| a == attr)
            .expect("Attribute not found in context");
        self.modify_relation_idx(obj_idx, attr_idx, value);
    }
}

#[derive(Debug, Clone)]
pub struct FormalConcept<A = String, B = String> {
    context: Arc<FormalContext<A, B>>,
    extent: BitVec, // A subset of objects
    intent: BitVec, // A subset of attributes
}

impl<A, B> FormalConcept<A, B> {
    pub fn validate(&self) -> bool {
        self.extent == self.context.induce_l(&self.intent)
            && self.intent == self.context.induce_r(&self.extent)
    }
}

impl<A: PartialEq, B: PartialEq> PartialEq for FormalConcept<A, B> {
    fn eq(&self, other: &Self) -> bool {
        *self.context == *other.context && self.extent == other.extent
    }
}

impl<A: Eq, B: Eq> Eq for FormalConcept<A, B> {}

impl<A: PartialEq, B: PartialEq> PartialOrd for FormalConcept<A, B> {
    // Concepts are ordered by subset containment of their extents, provided they are from the same context.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if *self.context != *other.context {
            return None; // Cannot compare concepts from different contexts
        }
        if self.extent == other.extent {
            return Some(std::cmp::Ordering::Equal);
        }
        if is_subset(&self.extent, &other.extent) {
            return Some(std::cmp::Ordering::Less);
        }
        if is_subset(&other.extent, &self.extent) {
            return Some(std::cmp::Ordering::Greater);
        }
        None
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_subset() {
        let a = bitvec![1, 0, 1];
        let b = bitvec![1, 1, 1];
        assert!(is_subset(&a, &b));
        assert!(!is_subset(&b, &a));
    }
}
