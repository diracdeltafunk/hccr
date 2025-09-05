use crate::fca::bit_fiddling::*;
use bitvec::prelude::*;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FormalContext<A = String, B = String> {
    pub objects: Vec<A>,              // A subset of objects is an extent
    pub attributes: Vec<B>,           // A subset of attributes is an intent
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
    /// `get_relation_idx(i,j)` returns (i,j) entry of the context matrix
    pub fn get_relation_idx(&self, obj_idx: usize, attr_idx: usize) -> bool {
        self.relation[obj_idx][attr_idx]
    }
    /// Given an extent (a set of objects), induce its intent (the common attributes of those objects).
    pub fn induce_r(&self, extent: &BitVec) -> BitVec {
        let mut intent = BitVec::repeat(true, self.attributes.len());
        for obj in extent.iter_ones() {
            intent &= &self.relation[obj];
        }
        intent
    }
    /// Given an intent (a set of attributes), induce its extent (the set of objects having those attributes).
    pub fn induce_l(&self, intent: &BitVec) -> BitVec {
        let mut extent = BitVec::repeat(true, self.objects.len());
        for attr in intent.iter_ones() {
            extent &= &self.relation_transposed[attr];
        }
        extent
    }
    /// Check if the context is reduced, meaning no row or column of the relation is the intersection of other rows or columns (resp).
    pub fn is_reduced(&self) -> bool {
        redundant_row(&self.relation).is_none()
            && redundant_row(&self.relation_transposed).is_none()
    }
    /// Modifies in place! Removes redundant rows and columns to obtain a reduced context
    pub fn reduce(&mut self) {
        while let Some(i) = redundant_row(&self.relation) {
            self.objects.remove(i);
            self.relation.remove(i);
            for c in &mut self.relation_transposed {
                c.remove(i);
            }
        }
        while let Some(i) = redundant_row(&self.relation_transposed) {
            self.attributes.remove(i);
            self.relation_transposed.remove(i);
            for r in &mut self.relation {
                r.remove(i);
            }
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
    pub fn get_relation(&self, obj: &A, attr: &B) -> bool {
        let Some(obj_idx) = self.objects.iter().position(|o| o == obj) else {
            panic!("Object not found in context");
        };
        let Some(attr_idx) = self.attributes.iter().position(|a| a == attr) else {
            panic!("Attribute not found in context");
        };
        self.relation[obj_idx][attr_idx]
    }
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
    #[test]
    fn test_reduction() {
        let mut context = FormalContext {
            objects: vec!["a", "b", "c"],
            attributes: vec!["1", "2", "3"],
            relation: vec![
                bitvec![1, 0, 1], // a
                bitvec![1, 1, 1], // b
                bitvec![0, 1, 1], // c
            ],
            relation_transposed: vec![
                bitvec![1, 1, 0], // 1
                bitvec![0, 1, 1], // 2
                bitvec![1, 1, 1], // 3
            ],
        };
        assert!(!context.is_reduced());
        context.reduce();
        assert!(context.relation == vec![bitvec![1, 0], bitvec![0, 1]]);
        assert!(context.is_reduced());
    }
}
