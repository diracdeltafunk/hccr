#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! `hccr` is a library for finite order-theoretic calculations arising in
//! homotopical combinatorics.
//!
//! The central objects are finite posets, finite lattices, and transfer
//! systems.  Elements of a finite poset are addressed by stable integer
//! indices, while user-supplied labels carry the mathematical names or data
//! attached to those indices.

/// Finite posets, relations, products, coproducts, and relation operations.
pub mod poset;

/// Finite lattices and lattice-theoretic constructions.
pub mod lattice;

/// Monotone maps of posets and homomorphisms of lattices.
pub mod morphism;

/// Rendering finite posets, lattices, and transfer systems as TikZ pictures.
pub mod tikz;

/// Transfer systems on finite lattices, including containment and
/// composition-closed orders.
pub mod transfer_lattice;

#[cfg(feature = "groups")]
/// GAP-backed finite group actions on lattices and G-equivariant transfer
/// systems.
pub mod g_lattice;
