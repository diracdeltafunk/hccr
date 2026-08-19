//! Cotransfer systems on finite lattices.
//!
//! A **cotransfer system** on a finite lattice `L` is a wide suborder `C` of
//! `L` that is closed under pushout: if `x C y` and `x <= z`, then
//! `z C (y \/ z)`. Equivalently, reversing every arrow identifies
//! cotransfer systems on `L` with transfer systems on the opposite lattice
//! `L^op`.
//!
//! This module uses that equivalence internally. It keeps cotransfer arrows in
//! the familiar orientation of `L`, but delegates generation, validation, and
//! enumeration to the existing transfer-system implementation on `L^op`.

use crate::bitvec_utils::set_partial_cmp;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset};
use crate::transfer_lattice::{
    RawTransferSystem, TransferError, TransferSystem, TransferSystemError, TransferUniverse,
};
use bitvec::prelude::*;
use std::fmt;
use std::sync::Arc;

/// An error while constructing a cotransfer-system containment lattice.
///
/// Cotransfer enumeration is transfer enumeration on the opposite lattice, so
/// it has exactly the same construction errors.
pub type CotransferError = TransferError;

/// A cotransfer system stored in the relation coordinates of `L^op`.
///
/// Bit `i` denotes the reversal in `L` of
/// `universe.opposite_transfer_universe().proper_edges()[i]`. Identity arrows
/// are implicit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawCotransferSystem {
    arrows: BitVec,
}

impl PartialOrd for RawCotransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        set_partial_cmp(&self.arrows, &other.arrows)
    }
}

impl RawCotransferSystem {
    pub(crate) fn new(arrows: BitVec) -> Self {
        Self { arrows }
    }

    /// Returns the bitvector of non-identity arrows.
    pub fn arrows(&self) -> &BitVec {
        &self.arrows
    }
}

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
    /// A raw bitvector has the wrong number of coordinates.
    WrongArrowCount {
        /// The number of proper relations in the universe.
        expected: usize,
        /// The number of supplied bits.
        actual: usize,
    },
    /// Raw arrows are not closed under the cotransfer-system axioms.
    RawNotClosed,
    /// A lifting or opposite-conversion input uses a different ambient
    /// transfer universe.
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
            Self::WrongArrowCount { expected, actual } => write!(
                f,
                "raw cotransfer system has {actual} arrow bits, expected {expected}"
            ),
            Self::RawNotClosed => write!(
                f,
                "raw arrow set is not closed under the cotransfer-system axioms"
            ),
            Self::LatticeMismatch => {
                write!(
                    f,
                    "the transfer and cotransfer systems use different ambient universes"
                )
            }
        }
    }
}

impl std::error::Error for CotransferSystemError {}

impl From<TransferSystemError> for CotransferSystemError {
    fn from(error: TransferSystemError) -> Self {
        match error {
            TransferSystemError::EdgeOutOfBounds { edge, lattice_size } => Self::EdgeOutOfBounds {
                edge: reverse(edge),
                lattice_size,
            },
            TransferSystemError::NotLatticeRelation { edge } => Self::NotLatticeRelation {
                edge: reverse(edge),
            },
            TransferSystemError::WrongArrowCount { expected, actual } => {
                Self::WrongArrowCount { expected, actual }
            }
            TransferSystemError::RawNotClosed => Self::RawNotClosed,
        }
    }
}

/// Shared coordinates and closure data for cotransfer systems on a lattice.
#[derive(Debug)]
pub struct CotransferUniverse<A> {
    lattice: Arc<Lattice<A>>,
    transfer_universe: Arc<TransferUniverse<A>>,
    opposite_lattice: Arc<Lattice<ElementId>>,
    opposite_transfer_universe: Arc<TransferUniverse<ElementId>>,
    proper_edges: Vec<Edge>,
}

/// An owned cotransfer system together with its ambient universe.
#[derive(Debug)]
pub struct CotransferSystem<A> {
    raw: RawCotransferSystem,
    universe: Arc<CotransferUniverse<A>>,
}

/// The lattice of cotransfer systems ordered by containment.
#[derive(Debug, Clone)]
pub struct CotransferLattice<A> {
    universe: Arc<CotransferUniverse<A>>,
    lattice: Lattice<RawCotransferSystem>,
}

impl<A> Lattice<A> {
    /// Builds the shared universe for cotransfer systems on this lattice.
    pub fn cotransfer_universe(self: Arc<Self>) -> Arc<CotransferUniverse<A>> {
        Arc::new(CotransferUniverse::new(self))
    }

    /// Constructs the containment lattice of cotransfer systems.
    pub fn cotransfer_systems_containment(
        self: Arc<Self>,
    ) -> Result<CotransferLattice<A>, CotransferError> {
        self.cotransfer_universe().containment_lattice()
    }
}

impl<A> CotransferUniverse<A> {
    /// Constructs a cotransfer universe on `lattice`.
    pub fn new(lattice: Arc<Lattice<A>>) -> Self {
        Self::from_transfer_universe(Arc::new(TransferUniverse::new(lattice)))
    }

    /// Constructs a cotransfer universe dual to an existing transfer universe.
    ///
    /// Retaining the supplied universe makes the right-lifting/left-lifting
    /// duality round-trip preserve the public [`TransferSystem`] value, not
    /// merely its set of arrows.
    pub fn from_transfer_universe(transfer_universe: Arc<TransferUniverse<A>>) -> Self {
        let lattice = Arc::clone(transfer_universe.lattice());
        let opposite_lattice = Arc::new(opposite_lattice_with_id_labels(&lattice));
        let opposite_transfer_universe =
            Arc::new(TransferUniverse::new(Arc::clone(&opposite_lattice)));
        let proper_edges = opposite_transfer_universe
            .proper_edges()
            .iter()
            .copied()
            .map(reverse)
            .collect();
        Self {
            lattice,
            transfer_universe,
            opposite_lattice,
            opposite_transfer_universe,
            proper_edges,
        }
    }

    /// Returns the underlying lattice in its original orientation.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.lattice
    }

    /// Returns the transfer universe dualized by lifting classes.
    pub fn transfer_universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.transfer_universe
    }

    /// Returns `L^op`, with element ids used as labels and coordinates unchanged.
    pub fn opposite_lattice(&self) -> &Arc<Lattice<ElementId>> {
        &self.opposite_lattice
    }

    /// Returns the transfer universe on `L^op` used by this dual representation.
    pub fn opposite_transfer_universe(&self) -> &Arc<TransferUniverse<ElementId>> {
        &self.opposite_transfer_universe
    }

    /// Returns the proper relations in `L` in raw cotransfer-coordinate order.
    pub fn proper_edges(&self) -> &[Edge] {
        &self.proper_edges
    }

    /// Constructs the least cotransfer system containing `generators`.
    pub fn generated_by<I, E>(
        self: &Arc<Self>,
        generators: I,
    ) -> Result<CotransferSystem<A>, CotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let mut reversed = Vec::new();
        for generator in generators {
            let edge = generator.into();
            validate_edge(&self.lattice, edge)?;
            reversed.push(reverse(edge));
        }
        let opposite = self.opposite_transfer_universe.generated_by(reversed)?;
        Ok(CotransferSystem::new(
            RawCotransferSystem::new(opposite.raw().arrows().clone()),
            Arc::clone(self),
        ))
    }

    /// Validates raw data and pairs it with this universe.
    ///
    /// Use [`CotransferUniverse::generated_by`] to add cotransfer closure.
    pub fn try_from_raw(
        self: &Arc<Self>,
        raw: RawCotransferSystem,
    ) -> Result<CotransferSystem<A>, CotransferSystemError> {
        let opposite_raw = RawTransferSystem::new(raw.arrows().clone());
        self.opposite_transfer_universe.try_from_raw(opposite_raw)?;
        Ok(CotransferSystem::new(raw, Arc::clone(self)))
    }

    /// Converts a transfer system on the coordinate-wise opposite lattice to
    /// its corresponding cotransfer system on `L`.
    ///
    /// The input must belong to this cotransfer universe's associated
    /// opposite transfer universe.
    pub fn from_opposite_transfer_system(
        self: &Arc<Self>,
        opposite: &TransferSystem<ElementId>,
    ) -> Result<CotransferSystem<A>, CotransferSystemError> {
        if !Arc::ptr_eq(&self.opposite_transfer_universe, opposite.universe()) {
            return Err(CotransferSystemError::LatticeMismatch);
        }
        self.generated_by(opposite.edges(false).into_iter().map(reverse))
    }

    /// Forms the left lifting class of a transfer system as a cotransfer system.
    pub fn left_lifting_of(
        self: &Arc<Self>,
        right: &TransferSystem<A>,
    ) -> Result<CotransferSystem<A>, CotransferSystemError> {
        if !Arc::ptr_eq(&self.transfer_universe, right.universe()) {
            return Err(CotransferSystemError::LatticeMismatch);
        }
        let arrows = self.lattice.as_poset().llc(&right.edges(true));
        self.generated_by(arrows)
    }

    /// Enumerates all cotransfer systems on the lattice.
    pub fn cotransfer_systems(self: &Arc<Self>) -> Vec<CotransferSystem<A>> {
        self.opposite_transfer_universe
            .transfer_systems()
            .into_iter()
            .map(|opposite| {
                CotransferSystem::new(
                    RawCotransferSystem::new(opposite.raw().arrows().clone()),
                    Arc::clone(self),
                )
            })
            .collect()
    }

    /// Constructs the lattice of cotransfer systems ordered by containment.
    pub fn containment_lattice(self: &Arc<Self>) -> Result<CotransferLattice<A>, CotransferError> {
        let opposite = self.opposite_transfer_universe.containment_lattice()?;
        let lattice = opposite
            .raw_lattice()
            .relabelled(|raw| RawCotransferSystem::new(raw.arrows().clone()));
        Ok(CotransferLattice {
            universe: Arc::clone(self),
            lattice,
        })
    }
}

impl<A> CotransferSystem<A> {
    pub(crate) fn new(raw: RawCotransferSystem, universe: Arc<CotransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    /// Returns the raw bitvector representation.
    pub fn raw(&self) -> &RawCotransferSystem {
        &self.raw
    }

    /// Returns the ambient cotransfer universe.
    pub fn universe(&self) -> &Arc<CotransferUniverse<A>> {
        &self.universe
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    /// Tests membership of a relation in this cotransfer system.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }
        self.universe
            .proper_edges()
            .iter()
            .position(|&edge| edge == relation)
            .is_some_and(|edge_id| self.raw.arrows()[edge_id])
    }

    /// Returns all selected relations, optionally including identities.
    pub fn edges(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            result.extend((0..self.lattice().size()).map(|id| Edge::new(id, id)));
        }
        result.extend(
            self.raw
                .arrows()
                .iter_ones()
                .map(|edge_id| self.universe.proper_edges()[edge_id]),
        );
        result
    }

    /// Returns the corresponding transfer system on `L^op`.
    ///
    /// This realizes the containment-preserving isomorphism
    /// `coTr(L) ~= Tr(L^op)`.
    pub fn opposite_transfer_system(&self) -> TransferSystem<ElementId> {
        TransferSystem::new(
            RawTransferSystem::new(self.raw.arrows().clone()),
            Arc::clone(self.universe.opposite_transfer_universe()),
        )
    }

    /// Forms this cotransfer system's right lifting class.
    ///
    /// Together with [`TransferSystem::left_lifting_cotransfer`], this realizes
    /// the order-reversing duality `coTr(L) ~= Tr(L)^op`.
    pub fn right_lifting_transfer(&self) -> Result<TransferSystem<A>, CotransferSystemError> {
        let arrows = self.lattice().as_poset().rlc(&self.edges(true));
        Ok(self.universe.transfer_universe.generated_by(arrows)?)
    }
}

impl<A> Clone for CotransferSystem<A> {
    fn clone(&self) -> Self {
        Self::new(self.raw.clone(), Arc::clone(&self.universe))
    }
}

impl<A> PartialEq for CotransferSystem<A> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && Arc::ptr_eq(&self.universe, &other.universe)
    }
}

impl<A> Eq for CotransferSystem<A> {}

impl<A> TransferSystem<A> {
    /// Forms this transfer system's left lifting class.
    ///
    /// The result is a cotransfer system on the identical underlying lattice.
    /// Applying [`CotransferSystem::right_lifting_transfer`] recovers `self`.
    pub fn left_lifting_cotransfer(&self) -> Result<CotransferSystem<A>, CotransferSystemError> {
        let universe = Arc::new(CotransferUniverse::from_transfer_universe(Arc::clone(
            self.universe(),
        )));
        universe.left_lifting_of(self)
    }
}

impl<A> CotransferLattice<A> {
    /// Returns the universe shared by all systems in this lattice.
    pub fn universe(&self) -> &Arc<CotransferUniverse<A>> {
        &self.universe
    }

    /// Returns the raw containment lattice.
    pub fn raw_lattice(&self) -> &Lattice<RawCotransferSystem> {
        &self.lattice
    }

    /// Returns the underlying containment poset.
    pub fn as_poset(&self) -> &Poset<RawCotransferSystem> {
        self.lattice.as_poset()
    }

    /// Returns the number of cotransfer systems.
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Returns the meet of two cotransfer systems by id.
    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    /// Returns the join of two cotransfer systems by id.
    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    /// Returns the bottom cotransfer system id.
    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    /// Returns the top cotransfer system id.
    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    /// Returns a cotransfer system by element id.
    pub fn system(&self, id: ElementId) -> Option<CotransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| CotransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Iterates over all cotransfer systems in element-id order.
    pub fn systems(&self) -> impl Iterator<Item = CotransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| CotransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Relabels the raw lattice by user-facing cotransfer systems.
    pub fn to_system_lattice(&self) -> Lattice<CotransferSystem<A>> {
        self.lattice
            .relabelled(|raw| CotransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }

    /// Relabels the containment poset by user-facing cotransfer systems.
    pub fn to_system_poset(&self) -> Poset<CotransferSystem<A>> {
        self.as_poset()
            .relabelled(|raw| CotransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

/// Constructs the opposite of a lattice, preserving element ids and labels.
pub fn opposite_lattice<A: Clone>(lattice: &Lattice<A>) -> Lattice<A> {
    Lattice::new(
        Poset::from_relation(
            lattice.elements().to_vec(),
            lattice.as_poset().relation_matrix_transpose().to_vec(),
        )
        .expect("the transpose of a partial order is a partial order"),
    )
    .expect("the opposite of a lattice is a lattice")
}

fn opposite_lattice_with_id_labels<A>(lattice: &Lattice<A>) -> Lattice<ElementId> {
    Lattice::new(
        Poset::from_relation(
            (0..lattice.size()).collect(),
            lattice.as_poset().relation_matrix_transpose().to_vec(),
        )
        .expect("the transpose of a partial order is a partial order"),
    )
    .expect("the opposite of a lattice is a lattice")
}

fn reverse(edge: Edge) -> Edge {
    Edge::new(edge.to, edge.from)
}

fn validate_edge<A>(lattice: &Lattice<A>, edge: Edge) -> Result<(), CotransferSystemError> {
    if edge.from >= lattice.size() || edge.to >= lattice.size() {
        return Err(CotransferSystemError::EdgeOutOfBounds {
            edge,
            lattice_size: lattice.size(),
        });
    }
    if !lattice.leq(edge.from, edge.to) {
        return Err(CotransferSystemError::NotLatticeRelation { edge });
    }
    Ok(())
}
