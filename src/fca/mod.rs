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
