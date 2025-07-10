use std::sync::Arc;

use bitvec::prelude::*;

fn is_subset(a: &BitVec, b: &BitVec) -> bool {
    let mut temp = a.clone();
    temp &= b;
    temp == *a
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct FormalContext<A = String, B = String> {
    objects: Vec<A>,                  // A subset of objects is an extent
    attributes: Vec<B>,               // A subset of attributes is an intent
    relation: Vec<BitVec>,            // The intent of each object
    relation_transposed: Vec<BitVec>, // The extent of each attribute
}

#[derive(Debug, Hash, Clone)]
struct FormalConcept {
    context: Arc<FormalContext>,
    extent: BitVec, // A subset of objects
    intent: BitVec, // A subset of attributes
}

impl FormalContext {
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
    // Check if the context is reduced, meaning no row or column of the relation is the intersection of other rows or columns (resp).
    // Note that if a row is the intersection of other rows, then it is the intersection of specifically those rows which are geq it.
    pub fn is_reduced(&self) -> bool {
        for i in 0..self.objects.len() {
            let mut best_approx = BitVec::repeat(true, self.attributes.len());
            for j in 0..self.objects.len() {
                if i != j && is_subset(&self.relation[i], &self.relation[j]) {
                    best_approx &= &self.relation[j];
                }
            }
            if best_approx == self.relation[i] {
                return false; // Found a row that is the intersection of other rows
            }
        }
        for j in 0..self.attributes.len() {
            let mut best_approx = BitVec::repeat(true, self.objects.len());
            for i in 0..self.attributes.len() {
                if j != i && is_subset(&self.relation_transposed[j], &self.relation_transposed[i]) {
                    best_approx &= &self.relation_transposed[i];
                }
            }
            if best_approx == self.relation_transposed[j] {
                return false; // Found a column that is the intersection of other columns
            }
        }
        true
    }
}

impl FormalConcept {
    pub fn validate(&self) -> bool {
        self.extent == self.context.induce_l(&self.intent)
            && self.intent == self.context.induce_r(&self.extent)
    }
}

impl PartialEq for FormalConcept {
    fn eq(&self, other: &Self) -> bool {
        *self.context == *other.context && self.extent == other.extent
    }
}

impl Eq for FormalConcept {}

impl PartialOrd for FormalConcept {
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
