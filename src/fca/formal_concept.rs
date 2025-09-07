use crate::fca::FormalContext;
use crate::fca::bit_fiddling::*;

use bitvec::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FormalConcept<A = String, B = String> {
    pub context: Arc<FormalContext<A, B>>,
    pub extent: BitVec, // A subset of objects
    pub intent: BitVec, // A subset of attributes
}

impl<A: std::fmt::Debug, B: std::fmt::Debug> std::fmt::Display for FormalConcept<A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extent: Vec<_> = self
            .extent
            .iter_ones()
            .map(|i| &self.context.objects[i])
            .collect();
        let intent: Vec<_> = self
            .intent
            .iter_ones()
            .map(|j| &self.context.attributes[j])
            .collect();
        write!(f, "Extent: {:?}, Intent: {:?}", extent, intent)
    }
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
