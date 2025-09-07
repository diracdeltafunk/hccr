//! This module implements Formal Concept Analysis (FCA) structures and algorithms.
//! It includes definitions for `FormalContext` and `FormalConcept`, along with methods for
//! computing intents and extents, checking for reduced contexts, and validating concepts.
//! The implementation uses bit vectors for efficient representation of relations.

mod bit_fiddling;
mod formal_concept;
mod formal_context;
mod pcbo;

pub use formal_concept::*;
pub use formal_context::*;

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;
    #[test]
    fn test_pcbo_1() {
        let context = FormalContext::new(
            vec!["a", "b", "c"],
            vec!["1", "2", "3"],
            vec![
                bitvec![1, 0, 1], // a
                bitvec![1, 1, 1], // b
                bitvec![0, 1, 1], // c
            ],
        );
        assert_eq!(context.num_concepts(), 4);
    }
    #[test]
    fn test_pcbo_2() {
        // "Lives in Water"
        let context = FormalContext::new(
            vec![
                "fish leech",
                "bream",
                "frog",
                "dog",
                "water weeds",
                "reed",
                "bean",
                "corn",
            ],
            vec![
                "needs water to live",
                "lives in water",
                "lives on land",
                "needs chlorophyll",
                "dicotyledon",
                "monocotyledon",
                "can move",
                "has limbs",
                "breast feeds",
            ],
            vec![
                bitvec![1, 1, 0, 0, 0, 0, 1, 0, 0], // fish leech
                bitvec![1, 1, 0, 0, 0, 0, 1, 1, 0], // bream
                bitvec![1, 1, 1, 0, 0, 0, 1, 1, 0], // frog
                bitvec![1, 0, 1, 0, 0, 0, 1, 1, 1], // dog
                bitvec![1, 1, 0, 1, 0, 1, 0, 0, 0], // water weeds
                bitvec![1, 1, 1, 1, 0, 1, 0, 0, 0], // reed
                bitvec![1, 0, 1, 1, 1, 0, 0, 0, 0], // bean
                bitvec![1, 0, 1, 1, 0, 1, 0, 0, 0], // corn
            ],
        );
        assert_eq!(context.num_concepts(), 19);
    }
}
