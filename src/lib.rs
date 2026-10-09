#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! `hccr` is a library for finite order-theoretic calculations arising in
//! homotopical combinatorics.
//!
//! The central objects are finite posets, finite lattices, transfer systems,
//! cotransfer systems, and model structures.
//!
//! # A quick tour
//!
//! Everything needed for everyday work comes from one import:
//!
//! ```
//! use hccr::prelude::*;
//!
//! fn main() -> hccr::Result<()> {
//!     // Build lattices from standard constructions...
//!     let c3 = chain(3);
//!     let fusion = horizontal_join([&c3, &c3, &c3])?;
//!     let square = product([chain(1), chain(1)]);
//!
//!     // ...or from labels and cover relations.
//!     let pentagon = Lattice::from_covers(
//!         ["0", "a", "b", "c", "1"],
//!         [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
//!     )?;
//!
//!     // Elements are named by labels, and found again by label.
//!     let x = fusion.id((1, 2))?;
//!     assert!(fusion.leq(x, fusion.top()));
//!     assert_eq!(square.label(square.top()).to_string(), "(1, 1)");
//!
//!     // Transfer systems.
//!     assert_eq!(fusion.transfer_systems().size(), 298);
//!     assert_eq!(pentagon.transfer_system_count(), 26);
//!     let t = pentagon.transfer_system_generated_by([pentagon.edge("a", "1")?])?;
//!     assert_eq!(t.to_string(), "{0 -> c, a -> b, a -> 1}");
//!
//!     // Maps between lattices, and the induced maps on transfer systems.
//!     let (_, inclusions) = Lattice::horizontal_join_with_inclusions([&c3, &c3, &c3])?;
//!     let middle = &inclusions[1];
//!     let s = c3.transfer_system_generated_by([(0, 3)])?;
//!     assert!(s.pushforward(middle)?.contains_relation((middle.apply(0), middle.apply(3)).into()));
//!     Ok(())
//! }
//! ```
//!
//! # Concepts
//!
//! A **poset** is a set with a reflexive, antisymmetric, transitive relation
//! `<=`; a **lattice** is a nonempty poset in which every two elements have a
//! greatest common lower bound (their meet) and a least common upper bound
//! (their join). See [`poset`] and [`lattice`] for constructors and standard
//! finite constructions. Every element carries a [`Label`]; see [`label`].
//!
//! [`Poset`](poset::Poset), [`Lattice`](lattice::Lattice), and the other
//! main types are cheap-to-clone handles: cloning one shares its data rather
//! than copying it, so there is never any need to wrap them in `Arc` or `Rc`.
//!
//! A **transfer system** is an additional partial order on the elements of a
//! lattice. It is contained in the lattice order and obeys a restriction
//! axiom; [`transfer_lattice`] gives the precise definition and explains how
//! the crate enumerates these systems. [`cotransfer_lattice`] implements their
//! pushout-closed duals, and [`model_structure`] constructs model structures
//! from suitable intervals of transfer systems. [`transfer_morphism`]
//! constructs the pushforward and pullback operations induced by monotone
//! maps, which are described in [`morphism`]. [`tikz`] draws all of these.
//!
//! Errors from every module convert into the single type [`Error`], so
//! `fn main() -> hccr::Result<()>` with `?` works throughout.
//!
//! # Groups
//!
//! The optional `groups` feature adds finite group actions and invariant
//! transfer and cotransfer systems, using GAP for the group theory. GAP is
//! re-exported as `hccr::gap`, so no separate dependency is needed:
//!
//! ```no_run
//! # #[cfg(feature = "groups")]
//! # fn main() -> hccr::Result<()> {
//! use hccr::prelude::*;
//!
//! let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
//! println!("{} transfer systems", s3.transfer_system_count());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "groups"))]
//! # fn main() {}
//! ```
//!
//! [`Label`]: label::Label

mod bitvec_utils;

pub mod error;

pub mod prelude;

pub mod label;

#[cfg(feature = "groups")]
pub mod group_theory;

pub mod poset;

pub mod lattice;

pub mod morphism;

pub mod tikz;

pub mod transfer_lattice;

pub mod transfer_morphism;

pub mod cotransfer_lattice;

pub mod model_structure;

#[cfg(feature = "groups")]
pub mod g_lattice;

#[cfg(feature = "groups")]
pub mod subgroup_morphism;

#[cfg(feature = "groups")]
pub mod g_transfer_morphism;

#[cfg(feature = "groups")]
pub mod g_cotransfer_lattice;

pub use error::{Error, Result};

/// The GAP interface, re-exported from [`gap_sys`] so that it needs no
/// separate dependency. For example, `hccr::gap::eval("SymmetricGroup(4);")`.
#[cfg(feature = "groups")]
pub use gap_sys as gap;
