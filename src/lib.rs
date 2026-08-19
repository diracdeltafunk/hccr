#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! `hccr` is a library for finite order-theoretic calculations arising in
//! homotopical combinatorics.
//!
//! The central objects are finite posets, finite lattices, and transfer
//! systems.
//!
//! A **poset** is a set with a reflexive, antisymmetric, transitive relation
//! `<=`; a **lattice** is a nonempty poset in which every two elements have a
//! greatest common lower bound (their meet) and a least common upper bound
//! (their join). See [`poset`] and [`lattice`] for constructors and standard
//! finite constructions.
//!
//! A **transfer system** is an additional partial order on the elements of a
//! lattice. It is contained in the lattice order and obeys a restriction
//! axiom; [`transfer_lattice`] gives the precise definition and explains how
//! the crate enumerates these systems. [`transfer_morphism`] constructs the
//! pushforward and pullback operations induced by monotone maps. The optional
//! `groups` feature adds finite group actions and invariant transfer systems,
//! using GAP for the group theory.
//!
//! # Element coordinates
//!
//! Elements are addressed by stable integer [`poset::ElementId`] values.
//! User-supplied labels carry mathematical names or data, but labels need not
//! be unique and therefore do not identify elements. Relations are oriented
//! from smaller to larger: [`poset::Edge::new`]`(x, y)` denotes `x <= y`.

mod bitvec_utils;

#[cfg(feature = "groups")]
/// GAP-backed finite-group data and permutation-action support.
pub mod group_theory;

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

/// Pushforwards, pullbacks, and induced order maps on transfer systems.
pub mod transfer_morphism;

#[cfg(feature = "groups")]
/// GAP-backed finite group actions on lattices and G-equivariant transfer
/// systems.
pub mod g_lattice;

#[cfg(feature = "groups")]
/// GAP-backed maps between subgroup lattices induced by group homomorphisms.
pub mod subgroup_morphism;

#[cfg(feature = "groups")]
/// Pushforwards and right-adjoint pullbacks of equivariant transfer systems.
pub mod g_transfer_morphism;
