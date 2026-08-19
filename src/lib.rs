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
mod bitvec_utils;

#[cfg(feature = "groups")]
pub mod group_theory;

pub mod poset;

pub mod lattice;

pub mod morphism;

pub mod tikz;

pub mod transfer_lattice;

pub mod transfer_morphism;

#[cfg(feature = "groups")]
pub mod g_lattice;

#[cfg(feature = "groups")]
pub mod subgroup_morphism;

#[cfg(feature = "groups")]
pub mod g_transfer_morphism;
