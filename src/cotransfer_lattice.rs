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
use crate::poset::{Edge, EdgeSet, ElementId, Poset, covers_of};
use crate::transfer_lattice::{TransferSystem, fmt_relations};
use bitvec::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Errors that can occur while constructing a cotransfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CotransferSystemError {
    /// A generating relation references an element outside the lattice.
    EdgeOutOfBounds {
        /// The invalid generating relation.
        edge: Edge,
        /// The number of elements in the lattice.
        lattice_size: usize,
    },
    /// A non-identity generating relation is not present in the lattice order.
    NotLatticeRelation {
        /// The invalid generating relation.
        edge: Edge,
    },
    /// The transfer system passed to
    /// [`Lattice::cotransfer_system_from_opposite`] is not on the opposite
    /// lattice `L^op`.
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

/// A cotransfer system on a finite lattice `L`: a partial order on the
/// elements of `L`, contained in the order of `L`, such that a relation
/// `x -> y` and `x <= z` force the relation `z -> (y \/ z)`.
///
/// It remembers its lattice, so its methods need no further context.
/// Cotransfer systems on the same lattice are ordered by containment, so `<=`
/// can be used to compare them. Displaying one lists its non-identity
/// relations as `{a -> b, ...}`, using element labels.
#[derive(Clone)]
pub struct CotransferSystem {
    lattice: Lattice,
    arrows: BitVec,
}

/// The lattice `coTr(L)` of cotransfer systems on a fixed lattice `L`,
/// ordered by containment.
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

    /// Returns the lattice `coTr(L)` of all cotransfer systems on this
    /// lattice `L`, ordered by containment.
    ///
    /// Meet is intersection; join is the cotransfer-system closure of the
    /// union.
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

    /// Returns the cotransfer system generated by the supplied relations: the
    /// least cotransfer system containing all of them.
    ///
    /// Relations are given by element id, as [`Edge`]s or `(from, to)` pairs.
    /// Identity relations may be supplied but need not be.
    ///
    /// # Errors
    ///
    /// Returns [`CotransferSystemError::EdgeOutOfBounds`] if a generator,
    /// identities included, names an id that is not an element of the
    /// lattice, and [`CotransferSystemError::NotLatticeRelation`] if a
    /// generator `x -> y` has `x` not below `y`.
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

    /// Returns the cotransfer system on this lattice `L` corresponding to a
    /// transfer system on the opposite lattice `L^op`, obtained by reversing
    /// every relation.
    ///
    /// This is inverse to [`CotransferSystem::opposite_transfer_system`].
    ///
    /// # Errors
    ///
    /// Returns [`CotransferSystemError::LatticeMismatch`] if `opposite` is not
    /// on the opposite lattice [`Lattice::opposite`] of this lattice.
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

    /// Returns whether a relation belongs to this cotransfer system.
    ///
    /// Every identity relation `x -> x` on an element of the lattice belongs.
    /// Pairs that are not relations of the lattice, and pairs naming ids
    /// outside the lattice, return `false`.
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

    /// Returns every relation of this cotransfer system, identities included.
    pub fn relations(&self) -> EdgeSet {
        let mut result = self.proper_relations();
        result.extend(self.lattice.ids().map(|id| Edge::new(id, id)));
        result
    }

    /// Returns the non-identity relations of this cotransfer system.
    pub fn proper_relations(&self) -> EdgeSet {
        let universe = self.universe();
        self.arrows
            .iter_ones()
            .map(|edge_id| universe.proper_edges[edge_id])
            .collect()
    }

    /// Returns the covers of this cotransfer system, regarded as a partial
    /// order on the elements of the lattice: its non-identity relations
    /// `x -> z` with no `y` other than `x` and `z` such that `x -> y` and
    /// `y -> z` are both relations of the system. These need not be covers in
    /// the lattice.
    pub fn covers(&self) -> EdgeSet {
        covers_of(&self.proper_relations())
    }

    /// Returns whether the supplied relations generate this cotransfer system.
    ///
    /// Identity relations may be supplied but are never needed.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Lattice::cotransfer_system_generated_by`]
    /// when a supplied pair is not a relation of the lattice.
    pub fn is_generated_by<I, E>(&self, generators: I) -> Result<bool, CotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        Ok(self.lattice.cotransfer_system_generated_by(generators)? == *self)
    }

    /// Returns whether the supplied relations form an inclusion-minimal
    /// generating set for this cotransfer system.
    ///
    /// Repetitions and identity relations make a generating collection
    /// non-minimal because deleting one of them does not change its closure.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`CotransferSystem::is_generated_by`].
    pub fn is_minimal_generating_set<I, E>(
        &self,
        generators: I,
    ) -> Result<bool, CotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let generators = generators.into_iter().map(Into::into).collect::<Vec<_>>();
        if !self.is_generated_by(generators.iter().copied())? {
            return Ok(false);
        }
        for removed in 0..generators.len() {
            let remainder = generators
                .iter()
                .enumerate()
                .filter_map(|(index, &edge)| (index != removed).then_some(edge));
            if self.is_generated_by(remainder)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns a deterministic inclusion-minimal generating set.
    ///
    /// It is the reversal of the
    /// [`TransferSystem::minimal_generating_set`] of
    /// [`CotransferSystem::opposite_transfer_system`].
    pub fn minimal_generating_set(&self) -> EdgeSet {
        self.opposite_transfer_system()
            .minimal_generating_set()
            .into_iter()
            .map(Edge::reversed)
            .collect()
    }

    /// Returns a minimum-cardinality generating set.
    ///
    /// As for transfer systems, every inclusion-minimal generating set has the
    /// same cardinality, so this is an explicitly named alias for
    /// [`CotransferSystem::minimal_generating_set`].
    pub fn minimum_generating_set(&self) -> EdgeSet {
        self.minimal_generating_set()
    }

    /// Returns the generator complexity of this cotransfer system: the common
    /// cardinality of its inclusion-minimal generating sets.
    pub fn generator_complexity(&self) -> usize {
        self.opposite_transfer_system().generator_complexity()
    }

    /// Returns the transfer system on the opposite lattice `L^op`, where
    /// `L = self.lattice()`, obtained by reversing every relation.
    ///
    /// This realizes the containment-preserving isomorphism between
    /// `coTr(L)` and `Tr(L^op)`; [`Lattice::cotransfer_system_from_opposite`]
    /// is its inverse.
    pub fn opposite_transfer_system(&self) -> TransferSystem {
        TransferSystem::new(self.universe().opposite.clone(), self.arrows.clone())
    }

    /// Returns the right lifting class of this cotransfer system: the
    /// relations with the right lifting property against every relation in
    /// it (see [`Poset::right_lifting_class`]). It is a transfer system on the
    /// same lattice.
    ///
    /// Together with [`TransferSystem::left_lifting_cotransfer`], this realizes
    /// an order-reversing isomorphism between `coTr(L)` and `Tr(L)`.
    pub fn right_lifting_transfer(&self) -> TransferSystem {
        let arrows = self.lattice.right_lifting_class(&self.relations());
        self.lattice
            .transfer_system_generated_by(arrows)
            .expect("a right lifting class consists of lattice relations")
    }
}

impl TransferSystem {
    /// Returns the left lifting class of this transfer system: the relations
    /// with the left lifting property against every relation in it (see
    /// [`Poset::left_lifting_class`]). It is a cotransfer system on the same
    /// lattice.
    ///
    /// Applying [`CotransferSystem::right_lifting_transfer`] recovers `self`.
    pub fn left_lifting_cotransfer(&self) -> CotransferSystem {
        let arrows = self.lattice().left_lifting_class(&self.relations());
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

impl std::hash::Hash for CotransferSystem {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lattice.hash(state);
        self.arrows.hash(state);
    }
}

impl PartialOrd for CotransferSystem {
    /// Compares cotransfer systems on the same lattice by containment.
    /// Systems on different lattices are incomparable.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.lattice != other.lattice {
            return None;
        }
        set_partial_cmp(&self.arrows, &other.arrows)
    }
}

impl fmt::Display for CotransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_relations(&self.lattice, self.proper_relations(), f)
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

    /// Returns the element id of `system` in this lattice of cotransfer
    /// systems, or `None` if it lives on a different base lattice.
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
