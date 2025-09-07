use std::fmt::Display;
use std::fs::File;

use crate::fca::FormalConcept;
use crate::fca::bit_fiddling::*;
use bitvec::prelude::*;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct FormalContext<A = String, B = String> {
    pub objects: Vec<A>,              // A subset of objects is an extent
    pub attributes: Vec<B>,           // A subset of attributes is an intent
    relation: Vec<BitVec>,            // The intent of each object
    relation_transposed: Vec<BitVec>, // The extent of each attribute
}

impl<A: Display, B: Display> Display for FormalContext<A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print header
        write!(f, "{:>10}", "")?;
        for attr in &self.attributes {
            write!(f, "{:>5}", attr)?;
        }
        writeln!(f)?;
        // Print each row
        for (i, obj) in self.objects.iter().enumerate() {
            write!(f, "{:>10}", obj)?;
            for j in 0..self.attributes.len() {
                let mark = if self.relation[i][j] { "1" } else { "0" };
                write!(f, "{:>5}", mark)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<A, B> FormalContext<A, B> {
    /// Constructs a new formal context
    /// The names of objects are given by `objects`
    /// The names of attributes are given by `attributes`
    /// The binary matrix is given by `relation` (`relation[i]` corresponds to `objects[i]`)
    pub fn new(objects: Vec<A>, attributes: Vec<B>, relation: Vec<BitVec>) -> Self {
        assert_eq!(relation.len(), objects.len());
        let mut relation_transposed = vec![BitVec::with_capacity(objects.len()); attributes.len()];
        for i in 0..objects.len() {
            assert_eq!(relation[i].len(), attributes.len());
            for j in 0..attributes.len() {
                relation_transposed[j].push(relation[i][j]);
            }
        }
        Self {
            objects,
            attributes,
            relation,
            relation_transposed,
        }
    }
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
    /// `get_object_intent(i)` returns a reference to the `i`th row of the context matrix
    pub fn get_object_intent(&self, i: usize) -> &BitVec {
        &self.relation[i]
    }
    /// `get_attribute_extent(i)` returns a reference to the `i`th column of the context matrix
    pub fn get_attribute_extent(&self, i: usize) -> &BitVec {
        &self.relation_transposed[i]
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

impl<A: Clone, B: Clone> FormalContext<A, B> {
    /// Get the maximal concept
    pub fn max_concept(&self) -> FormalConcept<A, B> {
        FormalConcept {
            context: std::sync::Arc::new(self.clone()),
            extent: BitVec::repeat(true, self.objects.len()),
            intent: self
                .relation
                .iter()
                .fold(BitVec::repeat(true, self.attributes.len()), |a, b| a & b),
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

impl FormalContext {
    /// Loads a formal context from a .cxt file. The format of a .cxt file is as follows:
    /// ```cxt
    /// B
    ///
    /// <num_objects>
    /// <num_attributes>
    ///
    /// <name of object 1>
    /// <name of object 2>
    /// ...
    /// <name of last object>
    /// <name of attribute 1>
    /// <name of attribute 2>
    /// ...
    /// <name of last attribute>
    /// <first row of context matrix>
    /// <second row of context matrix>
    /// ...
    /// <last row of context matrix>
    /// ```
    /// The blank lines and the first line (containing just the character `B`) *must* be present for the .cxt file to be well-formed!
    /// Each row of the context matrix corresponds to an object, and is a string of `.`s and `X`s. A `.` represents a 0 and an `X` represents a 1.
    pub fn from_cxt(file: File) -> Self {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip the first line (should be "B")
        lines.next().expect("Missing first line").expect("IO error");

        // Skip the blank line
        lines.next().expect("Missing blank line").expect("IO error");

        // Read number of objects and attributes
        let num_objects: usize = lines
            .next()
            .expect("Missing number of objects")
            .expect("IO error")
            .trim()
            .parse()
            .expect("Invalid number of objects");

        let num_attributes: usize = lines
            .next()
            .expect("Missing number of attributes")
            .expect("IO error")
            .trim()
            .parse()
            .expect("Invalid number of attributes");

        // Skip the blank line
        lines.next().expect("Missing blank line").expect("IO error");

        // Read object names
        let mut objects = Vec::with_capacity(num_objects);
        for _ in 0..num_objects {
            let obj_name = lines
                .next()
                .expect("Missing object name")
                .expect("IO error")
                .trim()
                .to_string();
            objects.push(obj_name);
        }

        // Read attribute names
        let mut attributes = Vec::with_capacity(num_attributes);
        for _ in 0..num_attributes {
            let attr_name = lines
                .next()
                .expect("Missing attribute name")
                .expect("IO error")
                .trim()
                .to_string();
            attributes.push(attr_name);
        }

        // Read relation matrix
        let mut relation = Vec::with_capacity(num_objects);
        for _ in 0..num_objects {
            let row_str = lines
                .next()
                .expect("Missing relation row")
                .expect("IO error")
                .trim()
                .to_string();

            let mut row = BitVec::with_capacity(num_attributes);
            for ch in row_str.chars() {
                match ch {
                    'X' => row.push(true),
                    '.' => row.push(false),
                    _ => panic!("Invalid character in matrix!"),
                }
            }

            if row.len() != num_attributes {
                panic!("Row length doesn't match number of attributes");
            }

            relation.push(row);
        }
        return Self::new(objects, attributes, relation);
    }
    /// Loads a formal context from a .dat file. The format of a .dat file is as follows:
    /// Each row corresponds to one object, and is a space-separated list of attributes (usually non-negative integers)
    /// That's it!
    pub fn from_dat(file: File) -> Self {
        use std::collections::HashSet;
        use std::io::{BufRead, BufReader};

        let reader = BufReader::new(file);
        let lines = reader.lines();

        // Collect all unique attributes from all lines
        let mut all_attributes = HashSet::new();
        let mut object_attributes: Vec<Vec<String>> = Vec::new();

        for line_result in lines {
            let attrs: Vec<String> = line_result
                .expect("IO Error")
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            object_attributes.push(attrs.clone());
            for attr in attrs {
                all_attributes.insert(attr);
            }
        }

        let num_objects = object_attributes.len();

        let objects: Vec<String> = (0..num_objects).map(|i| format!("obj{}", i)).collect();

        let attributes: Vec<String> = all_attributes.into_iter().collect();

        let mut relation = vec![BitVec::repeat(false, attributes.len()); num_objects];

        for i in 0..num_objects {
            for att in &object_attributes[i] {
                if let Some(j) = attributes.iter().position(|a| a == att) {
                    relation[i].set(j, true);
                }
            }
        }

        Self::new(objects, attributes, relation)
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
