//! Cotransfer systems on finite lattices.
//!
//! A **cotransfer system** on a finite lattice `L` is a wide suborder `C` of
//! `L` that is closed under pushout: if `x C y` and `x <= z`, then
//! `z C (y \/ z)`. Equivalently, reversing every arrow identifies
//! cotransfer systems on `L` with transfer systems on the opposite lattice
//! `L^op`.
//!
//! This module uses that equivalence internally. It keeps cotransfer arrows in
//! the familiar orientation of `L`, but delegates generation and enumeration
//! to the transfer-system implementation on `L^op`.

use crate::bitvec_utils::{is_subset, set_partial_cmp};
use crate::label::Label;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset};
use crate::transfer_lattice::{TransferSystem, fmt_relations};
use bitvec::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Errors that can occur while constructing one cotransfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CotransferSystemError {
    /// A relation references an element outside the lattice.
    EdgeOutOfBounds {
        /// The invalid relation.
        edge: Edge,
        /// The number of lattice elements.
        lattice_size: usize,
    },
    /// A non-identity relation is not present in the lattice order.
    NotLatticeRelation {
        /// The invalid relation.
        edge: Edge,
    },
    /// A transfer system on the wrong lattice was supplied.
    LatticeMismatch,
}

impl fmt::Display for CotransferSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EdgeOutOfBounds { edge, lattice_size } => write!(
                f,
                "generator {} <= {} is out of bounds for a lattice with {lattice_size} elements",
                edge.from, edge.to
            ),
            Self::NotLatticeRelation { edge } => write!(
                f,
                "generator {} <= {} is not a relation in the lattice order",
                edge.from, edge.to
            ),
            Self::LatticeMismatch => {
                write!(f, "the supplied transfer system is on the wrong lattice")
            }
        }
    }
}

impl std::error::Error for CotransferSystemError {}

/// The coordinates of cotransfer systems on a lattice `L`: the opposite
/// lattice, whose transfer systems they correspond to, and the proper
/// relations of `L` in the bit order of transfer systems on `L^op`.
pub(crate) struct CotransferUniverse {
    opposite: Lattice,
    proper_edges: Vec<Edge>,
}

/// A cotransfer system on a finite lattice.
///
/// It remembers its lattice. Cotransfer systems on the same lattice are
/// ordered by containment, so `<=` can be used to compare them.
#[derive(Clone)]
pub struct CotransferSystem {
    lattice: Lattice,
    arrows: BitVec,
}

/// The lattice of cotransfer systems on a fixed lattice, ordered by
/// containment.
///
/// This dereferences to the [`Lattice`] whose elements are the cotransfer
/// systems (labelled `0, 1, 2, ...`); use [`CotransferLattice::system`] to get
/// the cotransfer system with a given id.
#[derive(Clone)]
pub struct CotransferLattice {
    order: Lattice,
    base: Lattice,
    systems: Arc<[CotransferSystem]>,
    ids: Arc<HashMap<BitVec, ElementId>>,
}

impl Lattice {
    pub(crate) fn cotransfer_universe(&self) -> &CotransferUniverse {
        self.data().cotransfer_universe.get_or_init(|| {
            let opposite = self.opposite();
            let proper_edges = opposite
                .transfer_universe()
                .proper_edges()
                .iter()
                .map(|edge| edge.reversed())
                .collect();
            CotransferUniverse {
                opposite,
                proper_edges,
            }
        })
    }

    /// Constructs the lattice of all cotransfer systems on this lattice,
    /// ordered by containment.
    pub fn cotransfer_systems(&self) -> CotransferLattice {
        let opposite = &self.cotransfer_universe().opposite;
        let systems = opposite.transfer_universe().all_systems();
        let relation = systems
            .iter()
            .map(|left| systems.iter().map(|right| is_subset(left, right)).collect())
            .collect();
        let order = Lattice::new(Poset::from_validated(
            (0..systems.len()).map(Label::from).collect(),
            relation,
        ))
        .expect("cotransfer systems ordered by containment form a lattice");
        let ids = systems
            .iter()
            .enumerate()
            .map(|(id, arrows)| (arrows.clone(), id))
            .collect();
        let systems = systems
            .into_iter()
            .map(|arrows| CotransferSystem::new(self.clone(), arrows))
            .collect();
        CotransferLattice {
            order,
            base: self.clone(),
            systems,
            ids: Arc::new(ids),
        }
    }

    /// Counts the cotransfer systems on this lattice without storing them.
    pub fn cotransfer_system_count(&self) -> usize {
        self.cotransfer_universe().opposite.transfer_system_count()
    }

    /// Constructs the least cotransfer system containing `generators`, which
    /// are relations of this lattice given by element id.
    pub fn cotransfer_system_generated_by<I, E>(
        &self,
        generators: I,
    ) -> Result<CotransferSystem, CotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let mut reversed = Vec::new();
        for generator in generators {
            let edge = generator.into();
            if edge.from >= self.size() || edge.to >= self.size() {
                return Err(CotransferSystemError::EdgeOutOfBounds {
                    edge,
                    lattice_size: self.size(),
                });
            }
            if !self.leq(edge.from, edge.to) {
                return Err(CotransferSystemError::NotLatticeRelation { edge });
            }
            reversed.push(edge.reversed());
        }
        let opposite = self
            .cotransfer_universe()
            .opposite
            .transfer_system_generated_by(reversed)
            .expect("reversed lattice relations are relations of the opposite lattice");
        Ok(CotransferSystem::new(
            self.clone(),
            opposite.arrows().clone(),
        ))
    }

    /// Returns the trivial cotransfer system, containing only identities.
    pub fn trivial_cotransfer_system(&self) -> CotransferSystem {
        let bits = self.cotransfer_universe().proper_edges.len();
        CotransferSystem::new(self.clone(), BitVec::repeat(false, bits))
    }

    /// Returns the complete cotransfer system, containing every relation.
    pub fn complete_cotransfer_system(&self) -> CotransferSystem {
        let bits = self.cotransfer_universe().proper_edges.len();
        CotransferSystem::new(self.clone(), BitVec::repeat(true, bits))
    }

    /// Converts a transfer system on the opposite lattice `L^op` to the
    /// corresponding cotransfer system on this lattice.
    pub fn cotransfer_system_from_opposite(
        &self,
        opposite: &TransferSystem,
    ) -> Result<CotransferSystem, CotransferSystemError> {
        if *opposite.lattice() != self.cotransfer_universe().opposite {
            return Err(CotransferSystemError::LatticeMismatch);
        }
        Ok(CotransferSystem::new(
            self.clone(),
            opposite.arrows().clone(),
        ))
    }
}

impl CotransferSystem {
    pub(crate) fn new(lattice: Lattice, arrows: BitVec) -> Self {
        Self { lattice, arrows }
    }

    fn universe(&self) -> &CotransferUniverse {
        self.lattice.cotransfer_universe()
    }

    /// Returns the lattice this cotransfer system lives on.
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Tests membership of a relation in this cotransfer system.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice.size();
        }
        self.universe()
            .opposite
            .transfer_universe()
            .proper_edge_id(relation.reversed())
            .is_some_and(|edge_id| self.arrows[edge_id])
    }

    /// Returns all selected relations, optionally including identities.
    pub fn edges(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            result.extend(self.lattice.ids().map(|id| Edge::new(id, id)));
        }
        result.extend(self.sorted_proper_edges());
        result
    }

    /// Returns the non-identity relations, sorted by `(from, to)`.
    pub fn sorted_proper_edges(&self) -> Vec<Edge> {
        let universe = self.universe();
        let mut edges = self
            .arrows
            .iter_ones()
            .map(|edge_id| universe.proper_edges[edge_id])
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges
    }

    /// Returns the corresponding transfer system on `L^op`.
    ///
    /// This realizes the containment-preserving isomorphism
    /// `coTr(L) ~= Tr(L^op)`.
    pub fn opposite_transfer_system(&self) -> TransferSystem {
        TransferSystem::new(self.universe().opposite.clone(), self.arrows.clone())
    }

    /// Forms this cotransfer system's right lifting class, a transfer system.
    ///
    /// Together with [`TransferSystem::left_lifting_cotransfer`], this realizes
    /// the order-reversing duality `coTr(L) ~= Tr(L)^op`.
    pub fn right_lifting_transfer(&self) -> TransferSystem {
        let arrows = self.lattice.rlc(&self.edges(true));
        self.lattice
            .transfer_system_generated_by(arrows)
            .expect("a right lifting class consists of lattice relations")
    }
}

impl TransferSystem {
    /// Forms this transfer system's left lifting class, a cotransfer system
    /// on the same lattice.
    ///
    /// Applying [`CotransferSystem::right_lifting_transfer`] recovers `self`.
    pub fn left_lifting_cotransfer(&self) -> CotransferSystem {
        let arrows = self.lattice().llc(&self.edges(true));
        self.lattice()
            .cotransfer_system_generated_by(arrows)
            .expect("a left lifting class consists of lattice relations")
    }
}

impl PartialEq for CotransferSystem {
    fn eq(&self, other: &Self) -> bool {
        self.arrows == other.arrows && self.lattice == other.lattice
    }
}

impl Eq for CotransferSystem {}

impl PartialOrd for CotransferSystem {
    /// Compares cotransfer systems on the same lattice by containment.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.lattice != other.lattice {
            return None;
        }
        set_partial_cmp(&self.arrows, &other.arrows)
    }
}

impl fmt::Display for CotransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_relations(&self.lattice, self.sorted_proper_edges(), f)
    }
}

impl fmt::Debug for CotransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CotransferSystem {self}")
    }
}

impl CotransferLattice {
    /// Returns the lattice whose cotransfer systems these are.
    pub fn base_lattice(&self) -> &Lattice {
        &self.base
    }

    /// Returns the lattice of cotransfer systems itself, with elements
    /// labelled `0, 1, 2, ...`. The same lattice is available through
    /// dereferencing.
    pub fn as_lattice(&self) -> &Lattice {
        &self.order
    }

    /// Returns the cotransfer system with the given element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn system(&self, id: ElementId) -> &CotransferSystem {
        &self.systems[id]
    }

    /// Returns all cotransfer systems, in element-id order.
    pub fn systems(&self) -> &[CotransferSystem] {
        &self.systems
    }

    /// Returns the element id of a cotransfer system, if it belongs to this
    /// lattice.
    pub fn id_of(&self, system: &CotransferSystem) -> Option<ElementId> {
        if system.lattice != self.base {
            return None;
        }
        self.ids.get(&system.arrows).copied()
    }
}

impl Deref for CotransferLattice {
    type Target = Lattice;

    fn deref(&self) -> &Lattice {
        &self.order
    }
}

impl<'a> IntoIterator for &'a CotransferLattice {
    type Item = &'a CotransferSystem;
    type IntoIter = std::slice::Iter<'a, CotransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().iter()
    }
}

impl IntoIterator for CotransferLattice {
    type Item = CotransferSystem;
    type IntoIter = std::vec::IntoIter<CotransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().to_vec().into_iter()
    }
}

impl fmt::Display for CotransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lattice of {} cotransfer systems on {}",
            self.size(),
            self.base
        )
    }
}

impl fmt::Debug for CotransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
