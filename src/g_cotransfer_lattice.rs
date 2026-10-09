//! Cotransfer systems invariant under a finite group action.
//!
//! A G-cotransfer system is a cotransfer system whose arrows are closed under
//! the given action. As for G-transfer systems, it is stored as a bitvector of
//! non-identity relation orbits. Its FCA closure is the left-class closure
//! `llc(rlc(-))`, dual to the right-class closure used for G-transfer systems.

use crate::bitvec_utils::{is_subset, set_partial_cmp};
use crate::cotransfer_lattice::CotransferSystem;
use crate::g_lattice::{GLattice, GTransferSystem, GTransferUniverse, RelationOrbitLabel};
use crate::label::Label;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset};
use crate::transfer_lattice::fmt_relations;
use bitvec::prelude::*;
use fcars::FormalContext;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

/// The formal context whose concepts are G-cotransfer systems.
pub type GCotransferContext = FormalContext<RelationOrbitLabel, RelationOrbitLabel>;

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
    /// An input belongs to a different G-lattice.
    GLatticeMismatch,
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
            Self::GLatticeMismatch => {
                write!(f, "the supplied system belongs to a different G-lattice")
            }
        }
    }
}

impl std::error::Error for GCotransferSystemError {}

/// Shared coordinates and closure data for G-cotransfer systems.
pub(crate) struct GCotransferUniverse {
    transfer_universe: Rc<GTransferUniverse>,
    opposite_transfer_universe: Rc<GTransferUniverse>,
    context: GCotransferContext,
    relation_orbits: Vec<Vec<Edge>>,
    relation_to_orbit_label: Vec<Vec<Option<usize>>>,
}

/// A cotransfer system on a G-lattice that is invariant under the action.
///
/// G-cotransfer systems on the same G-lattice are ordered by containment, so
/// `<=` can be used to compare them.
#[derive(Clone)]
pub struct GCotransferSystem {
    universe: Rc<GCotransferUniverse>,
    orbit_arrows: BitVec,
}

/// The lattice of G-cotransfer systems on a fixed G-lattice, ordered by
/// containment.
///
/// This dereferences to the [`Lattice`] whose elements are the G-cotransfer
/// systems (labelled `0, 1, 2, ...`); use [`GCotransferLattice::system`] to
/// get the G-cotransfer system with a given id.
#[derive(Clone)]
pub struct GCotransferLattice {
    order: Lattice,
    universe: Rc<GCotransferUniverse>,
    systems: Rc<[GCotransferSystem]>,
    ids: Rc<HashMap<BitVec, ElementId>>,
}

impl GLattice {
    pub(crate) fn cotransfer_universe(&self) -> &Rc<GCotransferUniverse> {
        self.cotransfer_universe_cell().get_or_init(|| {
            Rc::new(GCotransferUniverse::new(Rc::clone(
                self.transfer_universe(),
            )))
        })
    }

    /// Constructs the lattice of G-cotransfer systems, ordered by containment.
    pub fn cotransfer_systems(&self) -> GCotransferLattice {
        let universe = self.cotransfer_universe();
        let systems = universe
            .context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| concept.extent)
            .collect::<Vec<_>>();
        let relation = systems
            .iter()
            .map(|left| systems.iter().map(|right| is_subset(left, right)).collect())
            .collect();
        let order = Lattice::new(Poset::from_validated(
            (0..systems.len()).map(Label::from).collect(),
            relation,
        ))
        .expect("G-cotransfer systems ordered by containment form a lattice");
        let ids = systems
            .iter()
            .enumerate()
            .map(|(id, arrows)| (arrows.clone(), id))
            .collect();
        let systems = systems
            .into_iter()
            .map(|arrows| GCotransferSystem::new(Rc::clone(universe), arrows))
            .collect();
        GCotransferLattice {
            order,
            universe: Rc::clone(universe),
            systems,
            ids: Rc::new(ids),
        }
    }

    /// Counts the G-cotransfer systems without storing them.
    pub fn cotransfer_system_count(&self) -> usize {
        self.cotransfer_universe().context.num_concepts()
    }

    /// Constructs the least G-cotransfer system containing `generators`,
    /// which are relations of the lattice given by element id.
    pub fn cotransfer_system_generated_by<I, E>(
        &self,
        generators: I,
    ) -> Result<GCotransferSystem, GCotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        self.cotransfer_universe().generated_by(generators)
    }

    /// Converts a G-transfer system on the opposite G-lattice to the
    /// corresponding G-cotransfer system.
    ///
    /// The input must have been produced by
    /// [`GCotransferSystem::opposite_transfer_system`] for this G-lattice.
    pub fn cotransfer_system_from_opposite(
        &self,
        opposite: &GTransferSystem,
    ) -> Result<GCotransferSystem, GCotransferSystemError> {
        let universe = self.cotransfer_universe();
        if !universe
            .opposite_transfer_universe
            .same_coordinates(opposite.universe())
        {
            return Err(GCotransferSystemError::GLatticeMismatch);
        }
        universe.generated_by(opposite.relations(false).into_iter().map(Edge::reversed))
    }
}

impl GCotransferUniverse {
    fn new(transfer_universe: Rc<GTransferUniverse>) -> Self {
        let lattice = transfer_universe.lattice();
        let labels = transfer_universe.relation_orbit_labels().to_vec();
        let relation_orbits = (0..labels.len())
            .map(|orbit_label_id| {
                transfer_universe
                    .relation_orbit_relations(orbit_label_id)
                    .to_vec()
            })
            .collect::<Vec<_>>();

        // Rows are possible left arrows and columns possible right arrows.
        // Testing one row representative against every arrow in a column
        // orbit is equivalent, by equivariance, to testing both whole orbits.
        let matrix = labels
            .iter()
            .map(|object| {
                relation_orbits
                    .iter()
                    .map(|orbit| {
                        orbit.iter().copied().all(|right| {
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
            opposite_transfer_universe: Rc::new(transfer_universe.opposite()),
            transfer_universe,
            context,
            relation_orbits,
            relation_to_orbit_label,
        }
    }

    fn lattice(&self) -> &Lattice {
        self.transfer_universe.lattice()
    }

    fn generated_by<I, E>(
        self: &Rc<Self>,
        generators: I,
    ) -> Result<GCotransferSystem, GCotransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let lattice = self.lattice();
        let mut orbit_arrows = BitVec::repeat(false, self.relation_orbits.len());
        for generator in generators {
            let edge = generator.into();
            if edge.from >= lattice.size() || edge.to >= lattice.size() {
                return Err(GCotransferSystemError::EdgeOutOfBounds {
                    edge,
                    lattice_size: lattice.size(),
                });
            }
            if !lattice.leq(edge.from, edge.to) {
                return Err(GCotransferSystemError::NotLatticeRelation { edge });
            }
            if edge.is_identity() {
                continue;
            }
            let orbit_label_id = self
                .relation_orbit_label_id(edge)
                .expect("every proper lattice relation lies in a relation orbit");
            orbit_arrows.set(orbit_label_id, true);
        }
        Ok(GCotransferSystem::new(
            Rc::clone(self),
            self.context.induce_l(&self.context.induce_r(&orbit_arrows)),
        ))
    }

    fn relation_orbit_label_id(&self, relation: Edge) -> Option<usize> {
        self.relation_to_orbit_label
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }
}

impl GCotransferSystem {
    fn new(universe: Rc<GCotransferUniverse>, orbit_arrows: BitVec) -> Self {
        Self {
            universe,
            orbit_arrows,
        }
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub fn lattice(&self) -> &Lattice {
        self.universe.lattice()
    }

    /// Tests membership of a relation in this G-cotransfer system.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }
        self.universe
            .relation_orbit_label_id(relation)
            .is_some_and(|orbit_id| self.orbit_arrows[orbit_id])
    }

    /// Returns the selected non-identity relation-orbit labels.
    pub fn relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.orbit_arrows
            .iter_ones()
            .map(|orbit_id| self.universe.context.objects[orbit_id])
            .collect()
    }

    /// Returns all selected relations, optionally including identities.
    pub fn relations(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            result.extend(self.lattice().ids().map(|id| Edge::new(id, id)));
        }
        for orbit_id in self.orbit_arrows.iter_ones() {
            result.extend(self.universe.relation_orbits[orbit_id].iter().copied());
        }
        result
    }

    /// Expands this system to an ordinary cotransfer system on the lattice.
    pub fn underlying_cotransfer_system(&self) -> CotransferSystem {
        self.lattice()
            .cotransfer_system_generated_by(self.relations(false))
            .expect("a G-cotransfer system is an ordinary cotransfer system")
    }

    /// Returns the corresponding G-transfer system on the opposite lattice.
    pub fn opposite_transfer_system(&self) -> GTransferSystem {
        GTransferSystem::new(
            Rc::clone(&self.universe.opposite_transfer_universe),
            self.orbit_arrows.clone(),
        )
    }

    /// Forms the right lifting class, a G-transfer system.
    ///
    /// Together with [`GTransferSystem::left_lifting_cotransfer`], this
    /// realizes the order-reversing duality between G-cotransfer and
    /// G-transfer systems.
    pub fn right_lifting_transfer(&self) -> GTransferSystem {
        let arrows = self.lattice().rlc(&self.relations(true));
        self.universe
            .transfer_universe
            .generated_by(arrows)
            .expect("a right lifting class consists of lattice relations")
    }
}

impl GTransferSystem {
    /// Forms the left lifting class, a G-cotransfer system on the same
    /// G-lattice.
    pub fn left_lifting_cotransfer(&self) -> GCotransferSystem {
        let universe = Rc::new(GCotransferUniverse::new(Rc::clone(self.universe())));
        let arrows = self.lattice().llc(&self.relations(true));
        universe
            .generated_by(arrows)
            .expect("a left lifting class consists of lattice relations")
    }
}

impl PartialEq for GCotransferSystem {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_arrows == other.orbit_arrows
            && self
                .universe
                .transfer_universe
                .same_coordinates(&other.universe.transfer_universe)
    }
}

impl Eq for GCotransferSystem {}

impl PartialOrd for GCotransferSystem {
    /// Compares G-cotransfer systems on the same G-lattice by containment.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self
            .universe
            .transfer_universe
            .same_coordinates(&other.universe.transfer_universe)
        {
            return None;
        }
        set_partial_cmp(&self.orbit_arrows, &other.orbit_arrows)
    }
}

impl fmt::Display for GCotransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut relations = self.relations(false).into_iter().collect::<Vec<_>>();
        relations.sort_unstable();
        fmt_relations(self.lattice(), relations, f)
    }
}

impl fmt::Debug for GCotransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GCotransferSystem {self}")
    }
}

impl GCotransferLattice {
    /// Returns the underlying lattice of the G-lattice whose G-cotransfer
    /// systems these are.
    pub fn base_lattice(&self) -> &Lattice {
        self.universe.lattice()
    }

    /// Returns the lattice of G-cotransfer systems itself, with elements
    /// labelled `0, 1, 2, ...`. The same lattice is available through
    /// dereferencing.
    pub fn as_lattice(&self) -> &Lattice {
        &self.order
    }

    /// Returns the G-cotransfer system with the given element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn system(&self, id: ElementId) -> &GCotransferSystem {
        &self.systems[id]
    }

    /// Returns all G-cotransfer systems, in element-id order.
    pub fn systems(&self) -> &[GCotransferSystem] {
        &self.systems
    }

    /// Returns the element id of a G-cotransfer system, if it belongs to this
    /// lattice.
    pub fn id_of(&self, system: &GCotransferSystem) -> Option<ElementId> {
        if !system
            .universe
            .transfer_universe
            .same_coordinates(&self.universe.transfer_universe)
        {
            return None;
        }
        self.ids.get(&system.orbit_arrows).copied()
    }
}

impl Deref for GCotransferLattice {
    type Target = Lattice;

    fn deref(&self) -> &Lattice {
        &self.order
    }
}

impl<'a> IntoIterator for &'a GCotransferLattice {
    type Item = &'a GCotransferSystem;
    type IntoIter = std::slice::Iter<'a, GCotransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().iter()
    }
}

impl IntoIterator for GCotransferLattice {
    type Item = GCotransferSystem;
    type IntoIter = std::vec::IntoIter<GCotransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().to_vec().into_iter()
    }
}

impl fmt::Display for GCotransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lattice of {} G-cotransfer systems on {}",
            self.size(),
            self.base_lattice()
        )
    }
}

impl fmt::Debug for GCotransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn lifting_condition(lattice: &Lattice, left: Edge, right: Edge) -> bool {
    !lattice.leq(left.from, right.from)
        || !lattice.leq(left.to, right.to)
        || lattice.leq(left.to, right.from)
}
