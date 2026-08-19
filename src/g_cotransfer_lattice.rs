//! Cotransfer systems invariant under a finite group action.
//!
//! A G-cotransfer system is a cotransfer system whose arrows are closed under
//! the given action. As for G-transfer systems, it is stored as a bitvector of
//! non-identity relation orbits. Its FCA closure is the left-class closure
//! `llc(rlc(-))`, dual to the right-class closure used for G-transfer systems.

use crate::bitvec_utils::{is_subset, set_partial_cmp};
use crate::cotransfer_lattice::{CotransferSystem, CotransferSystemError, CotransferUniverse};
use crate::g_lattice::{
    GLattice, GLatticeError, GTransferSystem, GTransferSystemError, GTransferUniverse,
    RawGTransferSystem, RelationOrbitLabel, SubgroupGLattice,
};
use crate::group_theory::GapSubgroup;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset};
use crate::transfer_lattice::TransferSystem;
use bitvec::prelude::*;
use fcars::FormalContext;
use std::fmt;
use std::sync::Arc;

/// The formal context whose concepts are G-cotransfer systems.
pub type GCotransferContext = FormalContext<RelationOrbitLabel, RelationOrbitLabel>;

/// A G-cotransfer system stored as selected relation orbits.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawGCotransferSystem {
    orbit_arrows: BitVec,
}

impl RawGCotransferSystem {
    pub(crate) fn new(orbit_arrows: BitVec) -> Self {
        Self { orbit_arrows }
    }

    /// Returns the selected non-identity relation-orbit bits.
    pub fn orbit_arrows(&self) -> &BitVec {
        &self.orbit_arrows
    }
}

impl PartialOrd for RawGCotransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        set_partial_cmp(&self.orbit_arrows, &other.orbit_arrows)
    }
}

/// Errors that can occur while constructing a G-cotransfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GCotransferSystemError {
    /// A relation references an element outside the lattice.
    EdgeOutOfBounds {
        /// The invalid relation.
        edge: Edge,
        /// The number of lattice elements.
        lattice_size: usize,
    },
    /// A non-identity relation is not in the lattice order.
    NotLatticeRelation {
        /// The invalid relation.
        edge: Edge,
    },
    /// Raw data has the wrong number of relation-orbit bits.
    WrongOrbitCount {
        /// The required number of bits.
        expected: usize,
        /// The supplied number of bits.
        actual: usize,
    },
    /// Raw orbit data is not cotransfer-closed.
    RawNotClosed,
    /// A lifting-class input belongs to a different ambient G-transfer universe.
    UniverseMismatch,
}

impl fmt::Display for GCotransferSystemError {
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
            Self::WrongOrbitCount { expected, actual } => write!(
                f,
                "raw G-cotransfer system has {actual} relation-orbit bits, expected {expected}"
            ),
            Self::RawNotClosed => write!(
                f,
                "raw relation-orbit set is not closed under the G-cotransfer-system axioms"
            ),
            Self::UniverseMismatch => write!(
                f,
                "the G-transfer and G-cotransfer systems use different ambient universes"
            ),
        }
    }
}

impl std::error::Error for GCotransferSystemError {}

impl From<GTransferSystemError> for GCotransferSystemError {
    fn from(error: GTransferSystemError) -> Self {
        match error {
            GTransferSystemError::EdgeOutOfBounds { edge, lattice_size } => {
                Self::EdgeOutOfBounds { edge, lattice_size }
            }
            GTransferSystemError::NotLatticeRelation { edge } => Self::NotLatticeRelation { edge },
            GTransferSystemError::WrongOrbitCount { expected, actual } => {
                Self::WrongOrbitCount { expected, actual }
            }
            GTransferSystemError::RawNotClosed => Self::RawNotClosed,
        }
    }
}

impl From<CotransferSystemError> for GCotransferSystemError {
    fn from(error: CotransferSystemError) -> Self {
        match error {
            CotransferSystemError::EdgeOutOfBounds { edge, lattice_size } => {
                Self::EdgeOutOfBounds { edge, lattice_size }
            }
            CotransferSystemError::NotLatticeRelation { edge } => Self::NotLatticeRelation { edge },
            CotransferSystemError::WrongArrowCount { expected, actual } => {
                Self::WrongOrbitCount { expected, actual }
            }
            CotransferSystemError::RawNotClosed => Self::RawNotClosed,
            CotransferSystemError::LatticeMismatch => Self::UniverseMismatch,
        }
    }
}

/// Shared orbit coordinates and closure data for G-cotransfer systems.
#[derive(Debug)]
pub struct GCotransferUniverse<A> {
    transfer_universe: Arc<GTransferUniverse<A>>,
    opposite_transfer_universe: Arc<GTransferUniverse<ElementId>>,
    underlying_cotransfer_universe: Arc<CotransferUniverse<A>>,
    context: GCotransferContext,
    relation_orbits: Vec<Vec<Edge>>,
    relation_to_orbit_label: Vec<Vec<Option<usize>>>,
}

/// An owned cotransfer system invariant under the ambient group action.
#[derive(Debug)]
pub struct GCotransferSystem<A> {
    raw: RawGCotransferSystem,
    universe: Arc<GCotransferUniverse<A>>,
}

/// The containment lattice of G-cotransfer systems.
#[derive(Debug, Clone)]
pub struct GCotransferLattice<A> {
    universe: Arc<GCotransferUniverse<A>>,
    lattice: Lattice<RawGCotransferSystem>,
}

impl<A> GLattice<A> {
    /// Builds the shared universe for G-cotransfer systems.
    pub fn cotransfer_universe(&self) -> Arc<GCotransferUniverse<A>> {
        Arc::new(GCotransferUniverse::new(self.transfer_universe()))
    }

    /// Constructs the containment lattice of G-cotransfer systems.
    pub fn cotransfer_systems_containment(&self) -> Result<GCotransferLattice<A>, GLatticeError> {
        self.cotransfer_universe().containment_lattice()
    }
}

impl SubgroupGLattice {
    /// Builds the universe of conjugation-invariant cotransfer systems.
    pub fn cotransfer_universe(&self) -> Arc<GCotransferUniverse<GapSubgroup>> {
        self.g_lattice().cotransfer_universe()
    }

    /// Constructs the containment lattice of conjugation-invariant cotransfer systems.
    pub fn cotransfer_systems_containment(
        &self,
    ) -> Result<GCotransferLattice<GapSubgroup>, GLatticeError> {
        self.g_lattice().cotransfer_systems_containment()
    }
}

impl<A> GCotransferUniverse<A> {
    /// Constructs the cotransfer universe dual to `transfer_universe`.
    pub fn new(transfer_universe: Arc<GTransferUniverse<A>>) -> Self {
        let lattice = transfer_universe.lattice();
        let labels = transfer_universe.relation_orbit_labels().to_vec();
        let relation_orbits = (0..labels.len())
            .map(|orbit_label_id| {
                transfer_universe
                    .relation_orbit_relations(orbit_label_id)
                    .expect("each relation-orbit label should name an orbit")
                    .to_vec()
            })
            .collect::<Vec<_>>();

        // Rows are possible left arrows and columns possible right arrows.
        // Testing one row representative against every arrow in a column
        // orbit is equivalent, by equivariance, to testing both whole orbits.
        let matrix = labels
            .iter()
            .map(|object| {
                labels
                    .iter()
                    .enumerate()
                    .map(|(attribute_label_id, _attribute)| {
                        relation_orbits[attribute_label_id]
                            .iter()
                            .copied()
                            .all(|right| {
                                lifting_condition(lattice, object.canonical_representative(), right)
                            })
                    })
                    .collect()
            })
            .collect();
        let context = FormalContext::new(labels.clone(), labels, matrix);

        let mut relation_to_orbit_label = vec![vec![None; lattice.size()]; lattice.size()];
        for (orbit_label_id, orbit) in relation_orbits.iter().enumerate() {
            for &relation in orbit {
                relation_to_orbit_label[relation.from][relation.to] = Some(orbit_label_id);
            }
        }

        Self {
            underlying_cotransfer_universe: Arc::new(CotransferUniverse::from_transfer_universe(
                Arc::clone(transfer_universe.underlying_transfer_universe()),
            )),
            opposite_transfer_universe: Arc::new(transfer_universe.opposite()),
            transfer_universe,
            context,
            relation_orbits,
            relation_to_orbit_label,
        }
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.transfer_universe.lattice()
    }

    /// Returns the G-transfer universe with the same action coordinates.
    pub fn transfer_universe(&self) -> &Arc<GTransferUniverse<A>> {
        &self.transfer_universe
    }

    /// Returns the G-transfer universe on `L^op` with the induced action.
    pub fn opposite_transfer_universe(&self) -> &Arc<GTransferUniverse<ElementId>> {
        &self.opposite_transfer_universe
    }

    /// Returns the ordinary cotransfer universe on the underlying lattice.
    pub fn underlying_cotransfer_universe(&self) -> &Arc<CotransferUniverse<A>> {
        &self.underlying_cotransfer_universe
    }

    /// Returns the formal context whose concepts enumerate G-cotransfer systems.
    pub fn context(&self) -> &GCotransferContext {
        &self.context
    }

    /// Returns the non-identity relation-orbit labels.
    pub fn relation_orbit_labels(&self) -> &[RelationOrbitLabel] {
        &self.context.objects
    }

    /// Returns all relations in a non-identity orbit.
    pub fn relation_orbit_relations(&self, orbit_label_id: usize) -> Option<&[Edge]> {
        self.relation_orbits.get(orbit_label_id).map(Vec::as_slice)
    }

    /// Constructs the least G-cotransfer system containing `generators`.
    pub fn generated_by<I, E>(
        self: &Arc<Self>,
        generators: I,
    ) -> Result<GCotransferSystem<A>, GCotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let mut orbit_arrows = BitVec::repeat(false, self.relation_orbit_labels().len());
        for generator in generators {
            let edge = generator.into();
            validate_edge(self.lattice(), edge)?;
            if edge.is_identity() {
                continue;
            }
            let Some(orbit_label_id) = self.relation_orbit_label_id(edge) else {
                return Err(GCotransferSystemError::NotLatticeRelation { edge });
            };
            orbit_arrows.set(orbit_label_id, true);
        }
        Ok(GCotransferSystem::new(
            RawGCotransferSystem::new(self.close_orbit_arrows(&orbit_arrows)),
            Arc::clone(self),
        ))
    }

    /// Validates raw relation-orbit data and pairs it with this universe.
    pub fn try_from_raw(
        self: &Arc<Self>,
        raw: RawGCotransferSystem,
    ) -> Result<GCotransferSystem<A>, GCotransferSystemError> {
        let expected = self.relation_orbit_labels().len();
        let actual = raw.orbit_arrows().len();
        if actual != expected {
            return Err(GCotransferSystemError::WrongOrbitCount { expected, actual });
        }
        if self.close_orbit_arrows(raw.orbit_arrows()) != *raw.orbit_arrows() {
            return Err(GCotransferSystemError::RawNotClosed);
        }
        Ok(GCotransferSystem::new(raw, Arc::clone(self)))
    }

    /// Forms the G-invariant left lifting class of a G-transfer system.
    pub fn left_lifting_of(
        self: &Arc<Self>,
        right: &GTransferSystem<A>,
    ) -> Result<GCotransferSystem<A>, GCotransferSystemError> {
        if !Arc::ptr_eq(&self.transfer_universe, right.universe()) {
            return Err(GCotransferSystemError::UniverseMismatch);
        }
        let arrows = self.lattice().as_poset().llc(&right.relations(true));
        self.generated_by(arrows)
    }

    /// Converts a G-transfer system on `L^op` to its G-cotransfer system on `L`.
    pub fn from_opposite_transfer_system(
        self: &Arc<Self>,
        opposite: &GTransferSystem<ElementId>,
    ) -> Result<GCotransferSystem<A>, GCotransferSystemError> {
        if !Arc::ptr_eq(&self.opposite_transfer_universe, opposite.universe()) {
            return Err(GCotransferSystemError::UniverseMismatch);
        }
        self.generated_by(
            opposite
                .relations(false)
                .into_iter()
                .map(|edge| Edge::new(edge.to, edge.from)),
        )
    }

    /// Enumerates all G-cotransfer systems.
    pub fn cotransfer_systems(self: &Arc<Self>) -> Vec<GCotransferSystem<A>> {
        self.context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| {
                GCotransferSystem::new(RawGCotransferSystem::new(concept.extent), Arc::clone(self))
            })
            .collect()
    }

    /// Constructs the containment lattice of G-cotransfer systems.
    pub fn containment_lattice(self: &Arc<Self>) -> Result<GCotransferLattice<A>, GLatticeError> {
        let systems = self
            .context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| RawGCotransferSystem::new(concept.extent))
            .collect::<Vec<_>>();
        let relation = systems
            .iter()
            .map(|left| {
                systems
                    .iter()
                    .map(|right| is_subset(left.orbit_arrows(), right.orbit_arrows()))
                    .collect()
            })
            .collect();
        let poset = Poset::from_relation(systems, relation)?;
        Ok(GCotransferLattice {
            universe: Arc::clone(self),
            lattice: Lattice::new(poset)?,
        })
    }

    pub(crate) fn close_orbit_arrows(&self, orbit_arrows: &BitVec) -> BitVec {
        debug_assert_eq!(orbit_arrows.len(), self.relation_orbit_labels().len());
        self.context.induce_l(&self.context.induce_r(orbit_arrows))
    }

    fn relation_orbit_label_id(&self, relation: Edge) -> Option<usize> {
        self.relation_to_orbit_label
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }
}

impl<A> GCotransferSystem<A> {
    pub(crate) fn new(raw: RawGCotransferSystem, universe: Arc<GCotransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    /// Returns the raw relation-orbit bitvector.
    pub fn raw(&self) -> &RawGCotransferSystem {
        &self.raw
    }

    /// Returns the ambient G-cotransfer universe.
    pub fn universe(&self) -> &Arc<GCotransferUniverse<A>> {
        &self.universe
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    /// Tests membership of a lattice relation.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }
        self.universe
            .relation_orbit_label_id(relation)
            .is_some_and(|orbit_id| self.raw.orbit_arrows()[orbit_id])
    }

    /// Returns the selected non-identity relation-orbit labels.
    pub fn relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.raw
            .orbit_arrows()
            .iter_ones()
            .map(|orbit_id| self.universe.relation_orbit_labels()[orbit_id])
            .collect()
    }

    /// Returns selected relations, optionally including identities.
    pub fn relations(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            result.extend((0..self.lattice().size()).map(|id| Edge::new(id, id)));
        }
        for orbit_id in self.raw.orbit_arrows().iter_ones() {
            result.extend(self.universe.relation_orbits[orbit_id].iter().copied());
        }
        result
    }

    /// Forgets equivariance and returns the underlying cotransfer system.
    pub fn underlying_cotransfer_system(&self) -> CotransferSystem<A> {
        self.universe
            .underlying_cotransfer_universe
            .generated_by(self.relations(false))
            .expect("a G-cotransfer system is an ordinary cotransfer system")
    }

    /// Returns the corresponding G-transfer system on `L^op`.
    ///
    /// This realizes the containment-preserving isomorphism
    /// `G-coTr(L) ~= G-Tr(L^op)` without reconstructing the group action.
    pub fn opposite_transfer_system(&self) -> GTransferSystem<ElementId> {
        GTransferSystem::new(
            RawGTransferSystem::new(self.raw.orbit_arrows().clone()),
            Arc::clone(self.universe.opposite_transfer_universe()),
        )
    }

    /// Returns the underlying G-fixed transfer system on `L^op`.
    ///
    /// This is the arrow-reversal isomorphism between G-cotransfer systems on
    /// `L` and G-fixed transfer systems on the opposite lattice.
    pub fn underlying_opposite_transfer_system(&self) -> TransferSystem<ElementId> {
        self.underlying_cotransfer_system()
            .opposite_transfer_system()
    }

    /// Forms this G-cotransfer system's right lifting class.
    pub fn right_lifting_transfer(&self) -> Result<GTransferSystem<A>, GCotransferSystemError> {
        let arrows = self.lattice().as_poset().rlc(&self.relations(true));
        Ok(self.universe.transfer_universe.generated_by(arrows)?)
    }
}

impl<A> Clone for GCotransferSystem<A> {
    fn clone(&self) -> Self {
        Self::new(self.raw.clone(), Arc::clone(&self.universe))
    }
}

impl<A> PartialEq for GCotransferSystem<A> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && Arc::ptr_eq(&self.universe, &other.universe)
    }
}

impl<A> Eq for GCotransferSystem<A> {}

impl<A> GTransferSystem<A> {
    /// Forms this G-transfer system's G-invariant left lifting class.
    pub fn left_lifting_cotransfer(&self) -> Result<GCotransferSystem<A>, GCotransferSystemError> {
        let universe = Arc::new(GCotransferUniverse::new(Arc::clone(self.universe())));
        universe.left_lifting_of(self)
    }
}

impl<A> GCotransferLattice<A> {
    /// Returns the universe shared by all systems in this lattice.
    pub fn universe(&self) -> &Arc<GCotransferUniverse<A>> {
        &self.universe
    }

    /// Returns the raw containment lattice.
    pub fn raw_lattice(&self) -> &Lattice<RawGCotransferSystem> {
        &self.lattice
    }

    /// Returns the underlying containment poset.
    pub fn as_poset(&self) -> &Poset<RawGCotransferSystem> {
        self.lattice.as_poset()
    }

    /// Returns the number of G-cotransfer systems.
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Returns the meet of two systems by element id.
    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    /// Returns the join of two systems by element id.
    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    /// Returns the bottom element id.
    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    /// Returns the top element id.
    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    /// Returns a G-cotransfer system by id.
    pub fn system(&self, id: ElementId) -> Option<GCotransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| GCotransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Iterates over all G-cotransfer systems in element-id order.
    pub fn systems(&self) -> impl Iterator<Item = GCotransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| GCotransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Relabels the raw lattice by user-facing G-cotransfer systems.
    pub fn to_system_lattice(&self) -> Lattice<GCotransferSystem<A>> {
        self.lattice
            .relabelled(|raw| GCotransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }

    /// Relabels the containment poset by user-facing G-cotransfer systems.
    pub fn to_system_poset(&self) -> Poset<GCotransferSystem<A>> {
        self.as_poset()
            .relabelled(|raw| GCotransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

fn lifting_condition<A>(lattice: &Lattice<A>, left: Edge, right: Edge) -> bool {
    !lattice.leq(left.from, right.from)
        || !lattice.leq(left.to, right.to)
        || lattice.leq(left.to, right.from)
}

fn validate_edge<A>(lattice: &Lattice<A>, edge: Edge) -> Result<(), GCotransferSystemError> {
    if edge.from >= lattice.size() || edge.to >= lattice.size() {
        return Err(GCotransferSystemError::EdgeOutOfBounds {
            edge,
            lattice_size: lattice.size(),
        });
    }
    if !lattice.leq(edge.from, edge.to) {
        return Err(GCotransferSystemError::NotLatticeRelation { edge });
    }
    Ok(())
}
