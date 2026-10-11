//! Transfer systems on a G-lattice that are invariant under the action.
//!
//! A **G-transfer system** is an ordinary transfer system invariant under the
//! action of `G`: whenever it contains one relation, it contains the
//! relation's whole orbit. It can therefore be stored compactly as a set of
//! non-identity relation orbits. The same formal-concept closure used for
//! ordinary transfer systems is applied at orbit level.
//!
//! ```no_run
//! use hccr::g_lattice::SubgroupGLattice;
//!
//! let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
//! let tr = s3.transfer_systems();
//! assert_eq!(tr.size(), 9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::bitvec_utils::{is_subset, set_partial_cmp};
use crate::g_lattice::{GLattice, RelationOrbitLabel, relation_id_matrix};
use crate::label::Label;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, covers_of};
use crate::transfer_lattice::{TransferSystem, fmt_relations};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

/// The formal context whose concepts are transfer systems on a G-lattice.
///
/// Objects and attributes are both non-identity relation orbits, represented by
/// [`RelationOrbitLabel`] values.
pub type GTransferContext = FormalContext<RelationOrbitLabel, RelationOrbitLabel>;

/// Shared data that gives G-transfer-system bitvectors their meaning: the
/// underlying lattice, the ordered list of non-identity relation orbits, and
/// the formal context whose concepts are the G-transfer systems.
pub(crate) struct GTransferUniverse {
    action_coordinates: Rc<()>,
    lattice: Lattice,
    context: GTransferContext,
    relation_orbits: Vec<Vec<Edge>>,
    relation_to_orbit_label: Vec<Vec<Option<usize>>>,
}

/// A transfer system on a G-lattice: an ordinary transfer system that is
/// invariant under the group action.
///
/// It is stored as a set of relation orbits. G-transfer systems on the same
/// G-lattice are ordered by containment, so `<=` can be used to compare them.
#[derive(Clone)]
pub struct GTransferSystem {
    universe: Rc<GTransferUniverse>,
    orbit_arrows: BitVec,
}

/// The lattice of G-transfer systems on a fixed G-lattice, ordered by
/// containment.
///
/// This dereferences to the [`Lattice`] whose elements are the G-transfer
/// systems (labelled `0, 1, 2, ...`); use [`GTransferLattice::system`] to get
/// the G-transfer system with a given id.
#[derive(Clone)]
pub struct GTransferLattice {
    order: Lattice,
    universe: Rc<GTransferUniverse>,
    systems: Rc<[GTransferSystem]>,
    ids: Rc<HashMap<BitVec, ElementId>>,
}

/// A concrete obstruction to compatibility of two G-transfer systems.
///
/// Compatibility is oriented: the receiver is the additive transfer system
/// and the argument is the multiplicative transfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GCompatibilityFailure {
    /// The two systems are on different G-lattices.
    DifferentGLattices,
    /// A multiplicative transfer is absent from the additive system.
    MultiplicativeNotAdditive {
        /// The multiplicative relation missing from the additive system.
        relation: Edge,
    },
    /// The distributivity condition fails for the displayed relations.
    Distributivity {
        /// A multiplicative relation `K -> H`.
        multiplicative: Edge,
        /// The additive relation `(K /\ L) -> K`.
        additive: Edge,
        /// The required additive relation `L -> H`.
        required: Edge,
    },
}

/// Errors that can occur while constructing an individual G-transfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GTransferSystemError {
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
}

impl fmt::Display for GTransferSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GTransferSystemError::EdgeOutOfBounds { edge, lattice_size } => write!(
                f,
                "generator {} <= {} is out of bounds for a lattice with {lattice_size} elements",
                edge.from, edge.to
            ),
            GTransferSystemError::NotLatticeRelation { edge } => write!(
                f,
                "generator {} <= {} is not a relation in the lattice order",
                edge.from, edge.to
            ),
        }
    }
}

impl std::error::Error for GTransferSystemError {}

impl GTransferUniverse {
    /// Constructs the G-transfer universe of a G-lattice.
    fn new(g_lattice: &GLattice) -> Self {
        let context = g_lattice.transfer_context();
        let relation_orbits = context
            .objects
            .iter()
            .map(|label| {
                g_lattice.data.relation_orbits[label.orbit_id()]
                    .relations
                    .clone()
            })
            .collect::<Vec<_>>();
        Self::from_packed_orbits(
            g_lattice.lattice().clone(),
            Rc::clone(g_lattice.action_coordinates()),
            context,
            relation_orbits,
        )
    }

    /// Constructs the transfer universe for the induced action on `L^op`.
    ///
    /// Element ids and action coordinates are unchanged, while every relation
    /// orbit is reversed. This is the internal coordinate bridge used by
    /// G-cotransfer systems.
    pub(crate) fn opposite(&self) -> GTransferUniverse {
        let opposite_lattice = self.lattice.opposite();
        let relation_orbits = self
            .relation_orbits
            .iter()
            .map(|orbit| orbit.iter().map(|edge| edge.reversed()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let opposite_relations = opposite_lattice.relations().into_iter().collect::<Vec<_>>();
        let opposite_relation_ids =
            relation_id_matrix(opposite_lattice.size(), &opposite_relations);
        let labels = self
            .relation_orbit_labels()
            .iter()
            .enumerate()
            .map(|(orbit_label_id, label)| {
                let (canonical_relation_id, canonical_representative) = relation_orbits
                    [orbit_label_id]
                    .iter()
                    .copied()
                    .map(|relation| {
                        (
                            opposite_relation_ids[relation.from][relation.to]
                                .expect("every reversed orbit arrow is an opposite relation"),
                            relation,
                        )
                    })
                    .min_by_key(|&(relation_id, _)| relation_id)
                    .expect("every non-identity relation orbit is nonempty");
                RelationOrbitLabel::new(
                    label.orbit_id(),
                    canonical_relation_id,
                    canonical_representative,
                )
            })
            .collect::<Vec<_>>();
        let context = packed_g_transfer_context(&opposite_lattice, labels, &relation_orbits);

        GTransferUniverse::from_packed_orbits(
            opposite_lattice,
            Rc::clone(&self.action_coordinates),
            context,
            relation_orbits,
        )
    }

    fn from_packed_orbits(
        lattice: Lattice,
        action_coordinates: Rc<()>,
        context: GTransferContext,
        relation_orbits: Vec<Vec<Edge>>,
    ) -> Self {
        let mut relation_to_orbit_label = vec![vec![None; lattice.size()]; lattice.size()];
        for (orbit_label_id, relations) in relation_orbits.iter().enumerate() {
            for &relation in relations {
                relation_to_orbit_label[relation.from][relation.to] = Some(orbit_label_id);
            }
        }

        Self {
            action_coordinates,
            lattice,
            context,
            relation_orbits,
            relation_to_orbit_label,
        }
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub(crate) fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Returns whether two universes use the same relation-orbit coordinates:
    /// the same group action on the same (or an equal) lattice.
    pub(crate) fn same_coordinates(&self, other: &GTransferUniverse) -> bool {
        std::ptr::eq(self, other)
            || (Rc::ptr_eq(&self.action_coordinates, &other.action_coordinates)
                && self.lattice == other.lattice)
    }

    /// Returns the opaque identity of the group action defining these orbit coordinates.
    pub(crate) fn action_coordinates(&self) -> &Rc<()> {
        &self.action_coordinates
    }

    /// Returns the non-identity relation-orbit labels, in bit order.
    pub(crate) fn relation_orbit_labels(&self) -> &[RelationOrbitLabel] {
        &self.context.objects
    }

    /// Returns all underlying lattice relations in a non-identity orbit.
    pub(crate) fn relation_orbit_relations(&self, orbit_label_id: usize) -> &[Edge] {
        &self.relation_orbits[orbit_label_id]
    }

    pub(crate) fn relation_orbit_label_id(&self, relation: Edge) -> Option<usize> {
        self.relation_to_orbit_label
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }

    pub(crate) fn generated_by<I, E>(
        self: &Rc<Self>,
        generators: I,
    ) -> Result<GTransferSystem, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let orbit_arrows = self.generator_orbit_bits(generators)?;
        Ok(GTransferSystem::new(
            Rc::clone(self),
            self.close(&orbit_arrows),
        ))
    }

    fn generator_orbit_bits<I, E>(&self, generators: I) -> Result<BitVec, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let lattice_size = self.lattice.size();
        let mut orbit_arrows = BitVec::repeat(false, self.relation_orbit_labels().len());

        for generator in generators {
            let edge = generator.into();
            if edge.from >= lattice_size || edge.to >= lattice_size {
                return Err(GTransferSystemError::EdgeOutOfBounds { edge, lattice_size });
            }
            if edge.is_identity() {
                continue;
            }

            let Some(orbit_label_id) = self.relation_orbit_label_id(edge) else {
                return Err(GTransferSystemError::NotLatticeRelation { edge });
            };
            orbit_arrows.set(orbit_label_id, true);
        }

        Ok(orbit_arrows)
    }

    /// Closes a correctly sized orbit bitvector under the G-transfer-system axioms.
    pub(crate) fn close(&self, orbit_arrows: &BitVec) -> BitVec {
        debug_assert_eq!(orbit_arrows.len(), self.relation_orbit_labels().len());
        self.context.induce_l(&self.context.induce_r(orbit_arrows))
    }

    /// Enumerates the bitvectors of all G-transfer systems.
    pub(crate) fn all_systems(&self) -> Vec<BitVec> {
        self.context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| concept.extent)
            .collect()
    }
}

impl GTransferSystem {
    pub(crate) fn new(universe: Rc<GTransferUniverse>, orbit_arrows: BitVec) -> Self {
        Self {
            universe,
            orbit_arrows,
        }
    }

    pub(crate) fn universe(&self) -> &Rc<GTransferUniverse> {
        &self.universe
    }

    pub(crate) fn orbit_arrows(&self) -> &BitVec {
        &self.orbit_arrows
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub fn lattice(&self) -> &Lattice {
        self.universe.lattice()
    }

    /// Returns whether a relation belongs to this G-transfer system.
    ///
    /// Every in-range identity relation is present.  A proper relation belongs
    /// exactly when its entire G-orbit is one of the selected orbits.
    /// Non-relations and out-of-range edges return `false`.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }

        self.universe
            .relation_orbit_label_id(relation)
            .is_some_and(|orbit_label_id| self.orbit_arrows[orbit_label_id])
    }

    /// Returns the selected non-identity relation-orbit labels.
    pub fn relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.orbit_arrows
            .iter_ones()
            .map(|orbit_label_id| self.universe.relation_orbit_labels()[orbit_label_id])
            .collect()
    }

    /// Returns every relation of this G-transfer system, identities included.
    pub fn relations(&self) -> EdgeSet {
        let mut result = self.proper_relations();
        result.extend(self.lattice().ids().map(|id| Edge::new(id, id)));
        result
    }

    /// Returns the non-identity relations of this G-transfer system.
    pub fn proper_relations(&self) -> EdgeSet {
        // Each selected orbit contributes all of its relations.
        self.orbit_arrows
            .iter_ones()
            .flat_map(|orbit_label_id| {
                self.universe
                    .relation_orbit_relations(orbit_label_id)
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// Returns the covers of this G-transfer system, regarded as a partial
    /// order on the elements of the lattice: the relations `x -> z` in it with
    /// no `y` such that `x -> y` and `y -> z` are both in it. These need not
    /// be covers in the lattice.
    pub fn covers(&self) -> EdgeSet {
        covers_of(&self.proper_relations())
    }

    /// Expands this G-transfer system to an ordinary transfer system.
    ///
    /// The resulting system is on the underlying lattice and contains the
    /// union of all selected relation orbits.  It is fixed by the G-action.
    /// Conversely, every G-fixed ordinary transfer system is obtained uniquely
    /// in this way.
    pub fn underlying_transfer_system(&self) -> TransferSystem {
        self.lattice()
            .transfer_system_generated_by(self.proper_relations())
            .expect("a G-transfer system is an ordinary transfer system")
    }

    /// Returns whether this G-transfer system is saturated.
    ///
    /// Saturation requires `K -> H` and `K <= L <= H` to imply `L -> H`.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        raw_g_transfer_system_is_saturated(&self.universe, &self.orbit_arrows)
    }

    /// Returns the least saturated G-transfer system containing this one.
    pub fn saturated_closure(&self) -> GTransferSystem {
        let universe = &self.universe;
        let lattice = universe.lattice();
        let mut current = self.orbit_arrows.clone();

        loop {
            let mut generators = current.clone();
            for orbit_id in current.iter_ones() {
                for &relation in universe.relation_orbit_relations(orbit_id) {
                    for middle in lattice.ids() {
                        if lattice.leq(relation.from, middle)
                            && lattice.leq(middle, relation.to)
                            && middle != relation.to
                        {
                            let required = Edge::new(middle, relation.to);
                            if let Some(required_orbit) = universe.relation_orbit_label_id(required)
                            {
                                generators.set(required_orbit, true);
                            }
                        }
                    }
                }
            }

            let next = universe.close(&generators);
            if next == current {
                return GTransferSystem::new(Rc::clone(universe), next);
            }
            current = next;
        }
    }

    /// Returns whether this G-transfer system is cosaturated, or disklike.
    ///
    /// A disklike system is generated by its relations whose target is the
    /// top element of the underlying lattice.
    #[must_use]
    pub fn is_cosaturated(&self) -> bool {
        self.cosaturated_coclosure() == *self
    }

    /// Alias for [`GTransferSystem::is_cosaturated`].
    #[must_use]
    pub fn is_disklike(&self) -> bool {
        self.is_cosaturated()
    }

    /// Returns the greatest cosaturated G-transfer system contained in this one.
    ///
    /// It is generated by all relations in this system with codomain the top
    /// element, and is therefore also called the cosaturation coclosure.
    pub fn cosaturated_coclosure(&self) -> GTransferSystem {
        let top = self.lattice().top();
        let generators = self
            .lattice()
            .ids()
            .filter(|&source| self.contains_relation(Edge::new(source, top)))
            .map(|source| Edge::new(source, top));
        self.universe
            .generated_by(generators)
            .expect("relations already in a G-transfer system must be valid generators")
    }

    /// Returns whether the supplied relations generate this G-transfer system.
    pub fn is_generated_by<I, E>(&self, generators: I) -> Result<bool, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        Ok(self.universe.generated_by(generators)? == *self)
    }

    /// Returns whether the supplied relations are an irredundant generating set.
    ///
    /// Supplying identities, duplicate relations, or two relations in the same
    /// G-orbit is redundant and therefore returns `false`.
    pub fn is_minimal_generating_set<I, E>(
        &self,
        generators: I,
    ) -> Result<bool, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let generators = generators.into_iter().map(Into::into).collect::<Vec<_>>();
        if generators.iter().any(|edge| edge.is_identity()) {
            return Ok(false);
        }

        let bits = self
            .universe
            .generator_orbit_bits(generators.iter().copied())?;
        if bits.count_ones() != generators.len() || self.universe.close(&bits) != self.orbit_arrows
        {
            return Ok(false);
        }

        for orbit_id in bits.iter_ones() {
            let mut smaller = bits.clone();
            smaller.set(orbit_id, false);
            if self.universe.close(&smaller) == self.orbit_arrows {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns a deterministic minimal set of relation-orbit generators.
    ///
    /// Each returned label represents its entire G-orbit. All minimal bases of
    /// a transfer system have the same cardinality, though the basis itself
    /// need not be unique.
    pub fn minimal_generating_set(&self) -> Vec<RelationOrbitLabel> {
        let mut basis = self.orbit_arrows.clone();
        let selected = basis.iter_ones().collect::<Vec<_>>();
        for orbit_id in selected.into_iter().rev() {
            basis.set(orbit_id, false);
            if self.universe.close(&basis) != self.orbit_arrows {
                basis.set(orbit_id, true);
            }
        }
        basis
            .iter_ones()
            .map(|orbit_id| self.universe.relation_orbit_labels()[orbit_id])
            .collect()
    }

    /// Returns a minimum-cardinality relation-orbit generating set.
    ///
    /// This is an explicit alias for [`GTransferSystem::minimal_generating_set`]:
    /// all irredundant bases of a transfer system have the same size.
    pub fn minimum_generating_set(&self) -> Vec<RelationOrbitLabel> {
        self.minimal_generating_set()
    }

    /// Returns the common cardinality of the minimal generating sets.
    #[must_use]
    pub fn generator_complexity(&self) -> usize {
        self.minimal_generating_set().len()
    }

    /// Returns a witness when `(self, multiplicative)` is not compatible.
    ///
    /// The receiver is the additive transfer system. Compatibility requires
    /// every multiplicative transfer to be additive and the distributivity
    /// implication from compatible indexing systems.
    pub fn compatibility_failure(
        &self,
        multiplicative: &GTransferSystem,
    ) -> Option<GCompatibilityFailure> {
        if !self.universe.same_coordinates(&multiplicative.universe) {
            return Some(GCompatibilityFailure::DifferentGLattices);
        }

        let mut multiplicative_relations = multiplicative
            .proper_relations()
            .into_iter()
            .collect::<Vec<_>>();
        multiplicative_relations.sort_unstable();
        for &relation in &multiplicative_relations {
            if !self.contains_relation(relation) {
                return Some(GCompatibilityFailure::MultiplicativeNotAdditive { relation });
            }
        }

        let lattice = self.lattice();
        for multiplicative_relation in multiplicative_relations {
            let k = multiplicative_relation.from;
            let h = multiplicative_relation.to;
            for l in lattice.ids() {
                if !lattice.leq(l, h) {
                    continue;
                }
                let additive_relation = Edge::new(lattice.meet(k, l), k);
                let required = Edge::new(l, h);
                if self.contains_relation(additive_relation) && !self.contains_relation(required) {
                    return Some(GCompatibilityFailure::Distributivity {
                        multiplicative: multiplicative_relation,
                        additive: additive_relation,
                        required,
                    });
                }
            }
        }
        None
    }

    /// Returns whether `(self, multiplicative)` is a compatible pair.
    ///
    /// The receiver is the additive transfer system and the argument is the
    /// multiplicative transfer system.
    #[must_use]
    pub fn is_compatible_with(&self, multiplicative: &GTransferSystem) -> bool {
        self.compatibility_failure(multiplicative).is_none()
    }
}

impl PartialEq for GTransferSystem {
    fn eq(&self, other: &Self) -> bool {
        self.orbit_arrows == other.orbit_arrows && self.universe.same_coordinates(&other.universe)
    }
}

impl Eq for GTransferSystem {}

impl std::hash::Hash for GTransferSystem {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Equal systems have the same coordinates, so the orbit bits suffice.
        self.orbit_arrows.hash(state);
    }
}

impl PartialOrd for GTransferSystem {
    /// Compares G-transfer systems on the same G-lattice by containment.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.universe.same_coordinates(&other.universe) {
            return None;
        }
        set_partial_cmp(&self.orbit_arrows, &other.orbit_arrows)
    }
}

impl fmt::Display for GTransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut relations = self.proper_relations().into_iter().collect::<Vec<_>>();
        relations.sort_unstable();
        fmt_relations(self.lattice(), relations, f)
    }
}

impl fmt::Debug for GTransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GTransferSystem {self}")
    }
}

impl GTransferLattice {
    fn from_raw(universe: &Rc<GTransferUniverse>, systems: Vec<BitVec>) -> Self {
        let relation = systems
            .iter()
            .map(|left| systems.iter().map(|right| is_subset(left, right)).collect())
            .collect();
        let order = Lattice::new(Poset::from_validated(
            (0..systems.len()).map(Label::from).collect(),
            relation,
        ))
        .expect("G-transfer systems ordered by containment form a lattice");
        let ids = systems
            .iter()
            .enumerate()
            .map(|(id, arrows)| (arrows.clone(), id))
            .collect();
        let systems = systems
            .into_iter()
            .map(|arrows| GTransferSystem::new(Rc::clone(universe), arrows))
            .collect();
        Self {
            order,
            universe: Rc::clone(universe),
            systems,
            ids: Rc::new(ids),
        }
    }

    pub(crate) fn universe(&self) -> &Rc<GTransferUniverse> {
        &self.universe
    }

    /// Returns the underlying lattice of the G-lattice whose G-transfer
    /// systems these are.
    pub fn base_lattice(&self) -> &Lattice {
        self.universe.lattice()
    }

    /// Returns the lattice of G-transfer systems itself, with elements
    /// labelled `0, 1, 2, ...`. The same lattice is available through
    /// dereferencing.
    pub fn as_lattice(&self) -> &Lattice {
        &self.order
    }

    /// Returns the G-transfer system with the given element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn system(&self, id: ElementId) -> &GTransferSystem {
        &self.systems[id]
    }

    /// Returns all G-transfer systems, in element-id order.
    pub fn systems(&self) -> &[GTransferSystem] {
        &self.systems
    }

    /// Returns the element id of a G-transfer system, if it belongs to this
    /// lattice.
    pub fn id_of(&self, system: &GTransferSystem) -> Option<ElementId> {
        if !system.universe.same_coordinates(&self.universe) {
            return None;
        }
        self.ids.get(&system.orbit_arrows).copied()
    }
}

impl Deref for GTransferLattice {
    type Target = Lattice;

    fn deref(&self) -> &Lattice {
        &self.order
    }
}

impl<'a> IntoIterator for &'a GTransferLattice {
    type Item = &'a GTransferSystem;
    type IntoIter = std::slice::Iter<'a, GTransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().iter()
    }
}

impl IntoIterator for GTransferLattice {
    type Item = GTransferSystem;
    type IntoIter = std::vec::IntoIter<GTransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().to_vec().into_iter()
    }
}

impl fmt::Display for GTransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lattice of {} G-transfer systems on {}",
            self.size(),
            self.base_lattice()
        )
    }
}

impl fmt::Debug for GTransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn transfer_context_relation(
    lattice: &Lattice,
    attribute_relation: Edge,
    object_relation: Edge,
) -> bool {
    lattice.leq(attribute_relation.to, object_relation.from)
        || !lattice.leq(attribute_relation.to, object_relation.to)
        || !lattice.leq(attribute_relation.from, object_relation.from)
}

fn packed_g_transfer_context(
    lattice: &Lattice,
    labels: Vec<RelationOrbitLabel>,
    relation_orbits: &[Vec<Edge>],
) -> GTransferContext {
    let mut attribute_coordinates = (0..labels.len()).collect::<Vec<_>>();
    attribute_coordinates.sort_unstable_by_key(|&coordinate| {
        let relation = labels[coordinate].canonical_representative();
        (relation.to, relation.from)
    });
    let matrix = labels
        .iter()
        .map(|object| {
            attribute_coordinates
                .iter()
                .map(|&coordinate| {
                    let attribute_orbit = &relation_orbits[coordinate];
                    attribute_orbit.iter().copied().all(|attribute| {
                        transfer_context_relation(
                            lattice,
                            attribute,
                            object.canonical_representative(),
                        )
                    })
                })
                .collect()
        })
        .collect();
    let attributes = attribute_coordinates
        .into_iter()
        .map(|coordinate| labels[coordinate])
        .collect();
    FormalContext::new(labels, attributes, matrix)
}

fn raw_g_transfer_system_is_saturated(universe: &GTransferUniverse, raw: &BitVec) -> bool {
    let lattice = universe.lattice();
    for orbit_id in raw.iter_ones() {
        for &relation in universe.relation_orbit_relations(orbit_id) {
            for middle in lattice.ids() {
                if lattice.leq(relation.from, middle)
                    && lattice.leq(middle, relation.to)
                    && middle != relation.to
                {
                    let required = Edge::new(middle, relation.to);
                    let Some(required_orbit) = universe.relation_orbit_label_id(required) else {
                        debug_assert!(required.is_identity());
                        continue;
                    };
                    if !raw[required_orbit] {
                        return false;
                    }
                }
            }
        }
    }
    true
}

impl GLattice {
    /// Constructs the formal context whose concepts are G-transfer systems.
    ///
    /// Objects and attributes are non-identity relation orbits.  For an object
    /// orbit represented by `r` and an attribute orbit `O`, incidence holds
    /// when the ordinary transfer-system lifting relation holds between `r`
    /// and every relation in `O`. Requiring incidence against a whole orbit is
    /// what builds `G`-invariance into the resulting closed sets.
    ///
    /// Object order is the stable relation-orbit coordinate order used by raw
    /// G-transfer systems. Attributes are ordered by target and then source;
    /// this target-major lectic order reduces PCbO search work without changing
    /// the closed extents or their coordinates.
    pub fn transfer_context(&self) -> GTransferContext {
        let labels = self.non_identity_relation_orbit_labels();
        let mut attribute_coordinates = (0..labels.len()).collect::<Vec<_>>();
        attribute_coordinates.sort_unstable_by_key(|&coordinate| {
            let relation = labels[coordinate].canonical_representative();
            (relation.to, relation.from)
        });
        let matrix = labels
            .iter()
            .map(|object| {
                attribute_coordinates
                    .iter()
                    .map(|&coordinate| {
                        let attribute = labels[coordinate];
                        self.data.relation_orbits[attribute.orbit_id()]
                            .relations()
                            .iter()
                            .all(|&relation| {
                                transfer_context_relation(
                                    &self.data.lattice,
                                    relation,
                                    object.canonical_representative(),
                                )
                            })
                    })
                    .collect()
            })
            .collect();
        let attributes = attribute_coordinates
            .into_iter()
            .map(|coordinate| labels[coordinate])
            .collect();
        FormalContext::new(labels, attributes, matrix)
    }

    /// Returns the cached G-transfer universe, building it on first use.
    pub(crate) fn transfer_universe(&self) -> &Rc<GTransferUniverse> {
        self.data
            .transfer_universe
            .get_or_init(|| Rc::new(GTransferUniverse::new(self)))
    }

    /// Constructs the lattice of G-transfer systems, ordered by containment.
    pub fn transfer_systems(&self) -> GTransferLattice {
        let universe = self.transfer_universe();
        GTransferLattice::from_raw(universe, universe.all_systems())
    }

    /// Counts the G-transfer systems without storing them.
    ///
    /// This traverses the formal concepts of the orbit-level transfer context
    /// but does not construct the transfer systems or their containment lattice.
    pub fn transfer_system_count(&self) -> usize {
        self.transfer_universe().context.num_concepts()
    }

    /// Constructs the containment lattice of saturated G-transfer systems.
    ///
    /// Its meet is intersection and its join is saturated closure of the
    /// ordinary G-transfer-system join.
    pub fn saturated_transfer_systems(&self) -> GTransferLattice {
        let universe = self.transfer_universe();
        let systems = universe
            .all_systems()
            .into_iter()
            .filter(|raw| raw_g_transfer_system_is_saturated(universe, raw))
            .collect();
        GTransferLattice::from_raw(universe, systems)
    }

    /// Returns the maximum size of a minimal generating set of a G-transfer system.
    pub fn transfer_system_complexity(&self) -> usize {
        self.transfer_systems()
            .systems()
            .iter()
            .map(GTransferSystem::generator_complexity)
            .max()
            .unwrap_or(0)
    }

    /// Returns the size of a minimal generating set of the complete G-transfer system.
    pub fn transfer_system_width(&self) -> usize {
        self.complete_transfer_system().generator_complexity()
    }

    /// Constructs the G-transfer system generated by the supplied relations.
    ///
    /// Relations are given by element id. Identity relations may be supplied
    /// but need not be. A proper relation selects its entire G-orbit, and the
    /// result is then closed under the G-transfer-system axioms.
    pub fn transfer_system_generated_by<I, E>(
        &self,
        generators: I,
    ) -> Result<GTransferSystem, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        self.transfer_universe().generated_by(generators)
    }

    /// Returns the trivial G-transfer system, containing only identities.
    pub fn trivial_transfer_system(&self) -> GTransferSystem {
        let universe = self.transfer_universe();
        GTransferSystem::new(
            Rc::clone(universe),
            BitVec::repeat(false, universe.relation_orbit_labels().len()),
        )
    }

    /// Returns the complete G-transfer system, containing every relation.
    pub fn complete_transfer_system(&self) -> GTransferSystem {
        let universe = self.transfer_universe();
        GTransferSystem::new(
            Rc::clone(universe),
            BitVec::repeat(true, universe.relation_orbit_labels().len()),
        )
    }

    /// Enumerates compatible `(additive, multiplicative)` pairs of
    /// G-transfer systems.
    pub fn compatible_transfer_system_pairs(&self) -> Vec<(GTransferSystem, GTransferSystem)> {
        let systems = self.transfer_systems();
        let mut pairs = Vec::new();
        for additive in systems.systems() {
            for multiplicative in systems.systems() {
                if additive.is_compatible_with(multiplicative) {
                    pairs.push((additive.clone(), multiplicative.clone()));
                }
            }
        }
        pairs
    }
}
