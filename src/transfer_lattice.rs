//! Transfer systems on finite lattices.
//!
//! Let `L` be a finite lattice. A **transfer system** is a partial order `R`
//! on the elements of `L` with two extra requirements:
//!
//! 1. `x R y` implies `x <= y` in the lattice; and
//! 2. if `x R y` and `z <= y`, then `(x /\ z) R z`.
//!
//! The second condition is called restriction closure. Together with
//! transitivity, it says that a permitted arrow remains permitted after
//! restricting its target to a smaller element.
//!
//! The main entry points are methods on [`Lattice`]:
//!
//! ```
//! use hccr::lattice::Lattice;
//!
//! let l = Lattice::boolean(2);
//! let tr = l.transfer_systems();
//! assert_eq!(tr.size(), 10);
//!
//! let t = l.transfer_system_generated_by([(l.id(hccr::label::Label::set([0]))?, l.top())])?;
//! assert!(t.contains_relation((l.bottom(), l.id(hccr::label::Label::set([1]))?).into()));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Enumeration algorithm
//!
//! Computationally, identity relations are implicit and a system is a
//! bitvector selecting proper lattice relations `x < y`. The crate encodes
//! restriction and transitivity as a formal context from formal concept
//! analysis (FCA), using the proper relations of `L` as both objects and
//! attributes. In FCA, applying derivation twice to a set of objects gives a
//! closure operator. For this particular context, the closed object sets are
//! exactly the transfer systems. Thus generation is one double-derivation and
//! enumeration uses the formal concepts of the context, rather than testing
//! all subsets of lattice relations independently. The context is built the
//! first time it is needed and then cached inside the lattice.

use crate::bitvec_utils::{intersection, intersects, is_subset, set_partial_cmp};
use crate::label::Label;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, compose};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

type TransferContext = FormalContext<Edge, Edge>;

/// Shared data that gives transfer-system bitvectors their meaning.
///
/// It fixes a row-major indexing of the proper relations of a lattice and the
/// formal context whose concepts are its transfer systems. One universe is
/// cached per lattice.
pub(crate) struct TransferUniverse {
    proper_edges: Vec<Edge>,
    proper_edge_ids: Vec<Vec<Option<usize>>>,
    context: TransferContext,
}

/// A transfer system on a finite lattice.
///
/// It remembers its lattice, so all of its methods need no further context.
/// Transfer systems on the same lattice are ordered by containment, so `<=`
/// can be used to compare them.
#[derive(Clone)]
pub struct TransferSystem {
    lattice: Lattice,
    arrows: BitVec,
}

/// A lattice of transfer systems on a fixed lattice, such as the containment
/// lattice `Tr(L)`.
///
/// This dereferences to the [`Lattice`] whose elements are the transfer
/// systems (labelled `0, 1, 2, ...`), so all lattice methods apply; use
/// [`TransferLattice::system`] to get the transfer system with a given id.
#[derive(Clone)]
pub struct TransferLattice {
    order: Lattice,
    systems: SystemTable,
}

/// A poset of transfer systems on a fixed lattice, such as the
/// composition-closed order, which need not be a lattice.
///
/// This dereferences to the [`Poset`] whose elements are the transfer
/// systems (labelled `0, 1, 2, ...`); use [`TransferPoset::system`] to get the
/// transfer system with a given id.
#[derive(Clone)]
pub struct TransferPoset {
    order: Poset,
    systems: SystemTable,
}

#[derive(Clone)]
struct SystemTable {
    base: Lattice,
    systems: Arc<[TransferSystem]>,
    ids: Arc<HashMap<BitVec, ElementId>>,
}

/// A concrete reason that an additive and multiplicative transfer system are
/// not compatible.
///
/// Compatibility uses the convention that the first system is additive and
/// the second is multiplicative.  In particular, every multiplicative
/// transfer must also be additive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityFailure {
    /// The two transfer systems are on different lattices.
    DifferentLattices,
    /// A multiplicative transfer is absent from the additive system.
    MultiplicativeNotAdditive {
        /// The multiplicative relation missing from the additive system.
        relation: Edge,
    },
    /// The distributivity condition fails for a triple of lattice elements.
    Distributivity {
        /// The multiplicative transfer `K -> H`.
        multiplicative: Edge,
        /// The additive transfer `(K /\ J) -> K`.
        additive: Edge,
        /// The required but absent additive transfer `J -> H`.
        required: Edge,
    },
}

/// Errors that can occur while constructing an individual transfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSystemError {
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

impl fmt::Display for TransferSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferSystemError::EdgeOutOfBounds { edge, lattice_size } => write!(
                f,
                "generator {} <= {} is out of bounds for a lattice with {lattice_size} elements",
                edge.from, edge.to
            ),
            TransferSystemError::NotLatticeRelation { edge } => write!(
                f,
                "generator {} <= {} is not a relation in the lattice order",
                edge.from, edge.to
            ),
        }
    }
}

impl std::error::Error for TransferSystemError {}

impl Lattice {
    /// Returns the cached transfer-system universe, building it on first use.
    pub(crate) fn transfer_universe(&self) -> &TransferUniverse {
        self.data()
            .transfer_universe
            .get_or_init(|| TransferUniverse::new(self))
    }

    /// Constructs the lattice `Tr(L)` of all transfer systems on this lattice,
    /// ordered by containment.
    ///
    /// Meet is intersection; join is the transfer-system closure of the union.
    pub fn transfer_systems(&self) -> TransferLattice {
        TransferLattice::from_raw(self, self.transfer_universe().all_systems())
    }

    /// Counts the transfer systems on this lattice without storing them.
    pub fn transfer_system_count(&self) -> usize {
        self.transfer_universe().context.num_concepts()
    }

    /// Constructs the lattice of saturated transfer systems, ordered by
    /// containment.
    ///
    /// Meets are intersections.  Joins are obtained by taking ordinary
    /// transfer-system closure of the union and then saturated closure.
    pub fn saturated_transfer_systems(&self) -> TransferLattice {
        let systems = self
            .transfer_universe()
            .all_systems()
            .into_iter()
            .filter(|arrows| is_saturated_raw(self, arrows))
            .collect();
        TransferLattice::from_raw(self, systems)
    }

    /// Constructs the composition-closed order on transfer systems.
    ///
    /// A comparison `R <= R'` requires ordinary containment and the
    /// factorization condition: each relevant square formed by one arrow from
    /// `R` and one from `R'` must admit an intermediate factorization using
    /// those two systems. Unlike containment, this order need not be a
    /// lattice.
    pub fn transfer_systems_composition_closed(&self) -> TransferPoset {
        let systems = self.transfer_universe().all_systems();
        let order = self.relation_matrix();
        let geometry = FactorizationGeometry::new(self);
        let partial_orders = systems
            .iter()
            .map(|arrows| as_partial_order(self, arrows))
            .collect::<Vec<_>>();
        let relation = (0..systems.len())
            .map(|left| {
                (0..systems.len())
                    .map(|right| {
                        is_subset(&systems[left], &systems[right])
                            && factorization_condition(
                                order,
                                &geometry,
                                &partial_orders[left],
                                &partial_orders[right],
                            )
                    })
                    .collect()
            })
            .collect();
        TransferPoset::from_raw(self, systems, relation)
    }

    /// Constructs the model-structure order on transfer systems.
    ///
    /// For transfer systems `R <= R'`, the corresponding premodel structure
    /// has weak equivalences `R ∘ llc(R')`. The pair is included exactly when
    /// those weak equivalences satisfy 2-out-of-3. Thus the intervals of this
    /// order are precisely the model structures on the lattice.
    pub fn transfer_systems_model_structure_order(&self) -> TransferPoset {
        let systems = self.transfer_universe().all_systems();
        let universe = self.transfer_universe();
        let right_classes = systems
            .iter()
            .map(|arrows| {
                let mut edges = EdgeSet::from_iter(
                    arrows
                        .iter_ones()
                        .map(|edge_id| universe.proper_edges[edge_id]),
                );
                edges.extend(self.ids().map(|id| Edge::new(id, id)));
                edges
            })
            .collect::<Vec<_>>();
        let left_classes = right_classes
            .iter()
            .map(|right| self.llc(right))
            .collect::<Vec<_>>();
        let relation = (0..systems.len())
            .map(|lower| {
                (0..systems.len())
                    .map(|upper| {
                        is_subset(&systems[lower], &systems[upper])
                            && self.two_out_of_three(&compose(
                                &right_classes[lower],
                                &left_classes[upper],
                            ))
                    })
                    .collect()
            })
            .collect();
        TransferPoset::from_raw(self, systems, relation)
    }

    /// Returns the transfer-system complexity of this lattice: the largest
    /// size of a minimal generating set of a transfer system on it.
    pub fn transfer_system_complexity(&self) -> usize {
        self.transfer_universe()
            .all_systems()
            .iter()
            .map(|arrows| minimal_generator_bits(self, arrows).count_ones())
            .max()
            .unwrap_or(0)
    }

    /// Returns the transfer-system width of this lattice: the size of a
    /// minimal generating set of the complete transfer system.
    pub fn transfer_system_width(&self) -> usize {
        self.complete_transfer_system().generator_complexity()
    }

    /// Constructs the transfer system generated by the supplied relations.
    ///
    /// Relations are given by element id, as [`Edge`]s or `(from, to)` pairs.
    /// Identity relations may be supplied but need not be. Every non-identity
    /// generator must be a relation of the lattice. The result is the least
    /// transfer system containing all of the generators.
    pub fn transfer_system_generated_by<I, E>(
        &self,
        generators: I,
    ) -> Result<TransferSystem, TransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let universe = self.transfer_universe();
        let mut arrows = BitVec::repeat(false, universe.proper_edges.len());
        for generator in generators {
            let edge = generator.into();
            if edge.from >= self.size() || edge.to >= self.size() {
                return Err(TransferSystemError::EdgeOutOfBounds {
                    edge,
                    lattice_size: self.size(),
                });
            }
            if edge.is_identity() {
                continue;
            }
            let Some(edge_id) = universe.proper_edge_id(edge) else {
                return Err(TransferSystemError::NotLatticeRelation { edge });
            };
            arrows.set(edge_id, true);
        }
        Ok(TransferSystem::new(self.clone(), universe.close(&arrows)))
    }

    /// Returns the trivial transfer system, containing only identities.
    pub fn trivial_transfer_system(&self) -> TransferSystem {
        let arrows = BitVec::repeat(false, self.transfer_universe().proper_edges.len());
        TransferSystem::new(self.clone(), arrows)
    }

    /// Returns the complete transfer system, containing every relation.
    pub fn complete_transfer_system(&self) -> TransferSystem {
        let arrows = BitVec::repeat(true, self.transfer_universe().proper_edges.len());
        TransferSystem::new(self.clone(), arrows)
    }

    /// Enumerates all compatible `(additive, multiplicative)` pairs of
    /// transfer systems, in the sense of
    /// [`TransferSystem::is_compatible_with`].
    pub fn compatible_transfer_system_pairs(&self) -> Vec<(TransferSystem, TransferSystem)> {
        let systems = self.transfer_universe().all_systems();
        let mut pairs = Vec::new();
        for additive in &systems {
            for multiplicative in &systems {
                if compatibility_failure_for_raw(self, additive, multiplicative).is_none() {
                    pairs.push((
                        TransferSystem::new(self.clone(), additive.clone()),
                        TransferSystem::new(self.clone(), multiplicative.clone()),
                    ));
                }
            }
        }
        pairs
    }
}

impl TransferUniverse {
    fn new(lattice: &Lattice) -> Self {
        let proper_edges = lattice.proper_relations_iter().collect::<Vec<_>>();
        let mut proper_edge_ids = vec![vec![None; lattice.size()]; lattice.size()];
        for (edge_id, &edge) in proper_edges.iter().enumerate() {
            proper_edge_ids[edge.from][edge.to] = Some(edge_id);
        }
        let context = build_transfer_context(lattice, &proper_edges);
        Self {
            proper_edges,
            proper_edge_ids,
            context,
        }
    }

    /// Returns the proper relations in deterministic row-major order; bit `i`
    /// of a transfer system's bitvector selects `proper_edges()[i]`.
    pub(crate) fn proper_edges(&self) -> &[Edge] {
        &self.proper_edges
    }

    /// Returns the bit index of a proper relation, or `None` for identities,
    /// non-relations, and out-of-range edges.
    pub(crate) fn proper_edge_id(&self, edge: Edge) -> Option<usize> {
        self.proper_edge_ids
            .get(edge.from)
            .and_then(|row| row.get(edge.to))
            .copied()
            .flatten()
    }

    /// Closes a correctly sized bitvector under the transfer-system axioms.
    ///
    /// Transfer systems are precisely the closed extents of the formal
    /// context, so the closure is the FCA double-prime operation.
    pub(crate) fn close(&self, arrows: &BitVec) -> BitVec {
        debug_assert_eq!(arrows.len(), self.proper_edges.len());
        self.context.induce_l(&self.context.induce_r(arrows))
    }

    /// Enumerates the bitvectors of all transfer systems.
    pub(crate) fn all_systems(&self) -> Vec<BitVec> {
        self.context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| concept.extent)
            .collect()
    }
}

impl TransferSystem {
    pub(crate) fn new(lattice: Lattice, arrows: BitVec) -> Self {
        debug_assert_eq!(
            arrows.len(),
            lattice.transfer_universe().proper_edges().len()
        );
        Self { lattice, arrows }
    }

    pub(crate) fn arrows(&self) -> &BitVec {
        &self.arrows
    }

    fn universe(&self) -> &TransferUniverse {
        self.lattice.transfer_universe()
    }

    /// Returns the lattice this transfer system lives on.
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Returns whether a relation belongs to this transfer system.
    ///
    /// Every in-range identity relation belongs to a transfer system.  A
    /// non-identity relation belongs precisely when it was selected;
    /// non-relations and out-of-range edges return `false`.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        raw_contains_relation(&self.lattice, &self.arrows, relation)
    }

    /// Returns the relations belonging to this transfer system.
    ///
    /// If `include_identities` is true, the identity relations `x <= x` are
    /// included along with the non-identity relations.
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
        self.arrows
            .iter_ones()
            .map(|edge_id| universe.proper_edges[edge_id])
            .collect()
    }

    /// Returns whether every relation of this system belongs to `other`.
    ///
    /// Returns `false` when the two systems are on different lattices.
    pub fn is_contained_in(&self, other: &TransferSystem) -> bool {
        self.lattice == other.lattice && is_subset(&self.arrows, &other.arrows)
    }

    /// Returns whether this transfer system is saturated.
    ///
    /// Saturation says that whenever `x -> z` belongs to the system and
    /// `x <= y <= z`, the relation `y -> z` also belongs to the system.  The
    /// other relation `x -> y` is already forced by restriction closure.
    pub fn is_saturated(&self) -> bool {
        is_saturated_raw(&self.lattice, &self.arrows)
    }

    /// Returns the least saturated transfer system containing this one.
    pub fn saturated_closure(&self) -> TransferSystem {
        let lattice = &self.lattice;
        let universe = self.universe();
        let mut arrows = self.arrows.clone();
        loop {
            let previous = arrows.clone();
            for edge_id in previous.iter_ones() {
                let edge = universe.proper_edges[edge_id];
                for middle in lattice.ids() {
                    let saturated_edge = Edge::new(middle, edge.to);
                    if lattice.leq(edge.from, middle)
                        && lattice.leq(middle, edge.to)
                        && !saturated_edge.is_identity()
                    {
                        let saturated_id = universe
                            .proper_edge_id(saturated_edge)
                            .expect("a proper intermediate relation should be indexed");
                        arrows.set(saturated_id, true);
                    }
                }
            }
            arrows = universe.close(&arrows);
            if arrows == previous {
                return TransferSystem::new(lattice.clone(), arrows);
            }
        }
    }

    /// Returns whether this transfer system is cosaturated.
    ///
    /// A cosaturated transfer system is generated by its relations whose
    /// target is the top of the lattice.  Such systems are also called
    /// *disklike* in the transfer-system literature.
    pub fn is_cosaturated(&self) -> bool {
        self.cosaturated_coclosure().arrows == self.arrows
    }

    /// Returns whether this transfer system is disklike.
    ///
    /// This is an alias for [`TransferSystem::is_cosaturated`].
    pub fn is_disklike(&self) -> bool {
        self.is_cosaturated()
    }

    /// Returns the greatest cosaturated transfer system contained in this one.
    ///
    /// It is generated by all relations in this system whose target is the
    /// top of the lattice.  Consequently this operation is contractive,
    /// monotone, and idempotent.
    pub fn cosaturated_coclosure(&self) -> TransferSystem {
        let universe = self.universe();
        let top = self.lattice.top();
        let mut top_arrows = BitVec::repeat(false, universe.proper_edges.len());
        for edge_id in self.arrows.iter_ones() {
            if universe.proper_edges[edge_id].to == top {
                top_arrows.set(edge_id, true);
            }
        }
        TransferSystem::new(self.lattice.clone(), universe.close(&top_arrows))
    }

    /// Returns whether the supplied relations generate this transfer system.
    ///
    /// Identity relations may be supplied but are never needed.  Invalid
    /// lattice relations produce the same errors as
    /// [`Lattice::transfer_system_generated_by`].
    pub fn is_generated_by<I, E>(&self, generators: I) -> Result<bool, TransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        Ok(self
            .lattice
            .transfer_system_generated_by(generators)?
            .arrows
            == self.arrows)
    }

    /// Returns whether the supplied relations form an inclusion-minimal
    /// generating set for this transfer system.
    ///
    /// Repetitions and identity relations make a generating collection
    /// non-minimal because deleting one of them does not change its closure.
    pub fn is_minimal_generating_set<I, E>(
        &self,
        generators: I,
    ) -> Result<bool, TransferSystemError>
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
    /// Relations are considered in reverse row-major order and deleted
    /// whenever the remaining relations still generate this system.
    pub fn minimal_generating_set(&self) -> EdgeSet {
        let universe = self.universe();
        minimal_generator_bits(&self.lattice, &self.arrows)
            .iter_ones()
            .map(|edge_id| universe.proper_edges[edge_id])
            .collect()
    }

    /// Returns a minimum-cardinality generating set.
    ///
    /// Every inclusion-minimal generating set of a transfer system has the
    /// same cardinality, so the deterministic set returned here is also
    /// minimum.  This method is an explicitly named alias for
    /// [`TransferSystem::minimal_generating_set`].
    pub fn minimum_generating_set(&self) -> EdgeSet {
        self.minimal_generating_set()
    }

    /// Returns the generator complexity of this transfer system.
    ///
    /// This is the common cardinality of its inclusion-minimal generating
    /// sets.
    pub fn generator_complexity(&self) -> usize {
        minimal_generator_bits(&self.lattice, &self.arrows).count_ones()
    }

    /// Returns a concrete failure when this additive transfer system is not
    /// compatible with `multiplicative`.
    ///
    /// Compatibility first requires every multiplicative relation to be
    /// additive.  It then requires that, for `K, J <= H`, a multiplicative
    /// `K -> H` and additive `(K /\ J) -> K` force additive `J -> H`.
    pub fn compatibility_failure(
        &self,
        multiplicative: &TransferSystem,
    ) -> Option<CompatibilityFailure> {
        if self.lattice != multiplicative.lattice {
            return Some(CompatibilityFailure::DifferentLattices);
        }
        compatibility_failure_for_raw(&self.lattice, &self.arrows, &multiplicative.arrows)
    }

    /// Returns whether this additive transfer system is compatible with the
    /// supplied multiplicative transfer system.
    pub fn is_compatible_with(&self, multiplicative: &TransferSystem) -> bool {
        self.compatibility_failure(multiplicative).is_none()
    }
}

impl PartialEq for TransferSystem {
    fn eq(&self, other: &Self) -> bool {
        self.arrows == other.arrows && self.lattice == other.lattice
    }
}

impl Eq for TransferSystem {}

impl PartialOrd for TransferSystem {
    /// Compares transfer systems on the same lattice by containment. Systems
    /// on different lattices are incomparable.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.lattice != other.lattice {
            return None;
        }
        set_partial_cmp(&self.arrows, &other.arrows)
    }
}

impl fmt::Display for TransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_relations(&self.lattice, self.sorted_proper_edges(), f)
    }
}

impl fmt::Debug for TransferSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransferSystem {self}")
    }
}

/// Writes a set of relations as `{a -> b, c -> d}` using element labels.
pub(crate) fn fmt_relations(
    poset: &Poset,
    edges: impl IntoIterator<Item = Edge>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    f.write_str("{")?;
    for (index, edge) in edges.into_iter().enumerate() {
        if index > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{} -> {}", poset.label(edge.from), poset.label(edge.to))?;
    }
    f.write_str("}")
}

impl SystemTable {
    fn new(base: &Lattice, systems: Vec<BitVec>) -> Self {
        let ids = systems
            .iter()
            .enumerate()
            .map(|(id, arrows)| (arrows.clone(), id))
            .collect();
        let systems = systems
            .into_iter()
            .map(|arrows| TransferSystem::new(base.clone(), arrows))
            .collect();
        Self {
            base: base.clone(),
            systems,
            ids: Arc::new(ids),
        }
    }

    fn id_of(&self, system: &TransferSystem) -> Option<ElementId> {
        if system.lattice != self.base {
            return None;
        }
        self.ids.get(&system.arrows).copied()
    }
}

fn containment_relation(systems: &[BitVec]) -> Vec<BitVec> {
    systems
        .iter()
        .map(|left| systems.iter().map(|right| is_subset(left, right)).collect())
        .collect()
}

impl TransferLattice {
    fn from_raw(base: &Lattice, systems: Vec<BitVec>) -> Self {
        let relation = containment_relation(&systems);
        let order = Lattice::new(Poset::from_validated(
            (0..systems.len()).map(Label::from).collect(),
            relation,
        ))
        .expect("transfer systems ordered by containment form a lattice");
        Self {
            order,
            systems: SystemTable::new(base, systems),
        }
    }

    /// Returns the lattice whose transfer systems these are.
    pub fn base_lattice(&self) -> &Lattice {
        &self.systems.base
    }

    /// Returns the lattice of transfer systems itself, with elements labelled
    /// `0, 1, 2, ...`. The same lattice is available through dereferencing.
    pub fn as_lattice(&self) -> &Lattice {
        &self.order
    }

    /// Returns the transfer system with the given element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn system(&self, id: ElementId) -> &TransferSystem {
        &self.systems.systems[id]
    }

    /// Returns all transfer systems, in element-id order.
    pub fn systems(&self) -> &[TransferSystem] {
        &self.systems.systems
    }

    /// Returns the element id of a transfer system, if it belongs to this
    /// lattice.
    pub fn id_of(&self, system: &TransferSystem) -> Option<ElementId> {
        self.systems.id_of(system)
    }
}

impl TransferPoset {
    fn from_raw(base: &Lattice, systems: Vec<BitVec>, relation: Vec<BitVec>) -> Self {
        let order = Poset::from_relation((0..systems.len()).map(Label::from), relation)
            .expect("the transfer-system orders are partial orders");
        Self {
            order,
            systems: SystemTable::new(base, systems),
        }
    }

    /// Returns the lattice whose transfer systems these are.
    pub fn base_lattice(&self) -> &Lattice {
        &self.systems.base
    }

    /// Returns the poset of transfer systems itself, with elements labelled
    /// `0, 1, 2, ...`. The same poset is available through dereferencing.
    pub fn as_poset(&self) -> &Poset {
        &self.order
    }

    /// Returns the transfer system with the given element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn system(&self, id: ElementId) -> &TransferSystem {
        &self.systems.systems[id]
    }

    /// Returns all transfer systems, in element-id order.
    pub fn systems(&self) -> &[TransferSystem] {
        &self.systems.systems
    }

    /// Returns the element id of a transfer system, if it belongs to this
    /// poset.
    pub fn id_of(&self, system: &TransferSystem) -> Option<ElementId> {
        self.systems.id_of(system)
    }
}

impl Deref for TransferLattice {
    type Target = Lattice;

    fn deref(&self) -> &Lattice {
        &self.order
    }
}

impl Deref for TransferPoset {
    type Target = Poset;

    fn deref(&self) -> &Poset {
        &self.order
    }
}

impl<'a> IntoIterator for &'a TransferLattice {
    type Item = &'a TransferSystem;
    type IntoIter = std::slice::Iter<'a, TransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().iter()
    }
}

impl IntoIterator for TransferLattice {
    type Item = TransferSystem;
    type IntoIter = std::vec::IntoIter<TransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().to_vec().into_iter()
    }
}

impl fmt::Display for TransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lattice of {} transfer systems on {}",
            self.size(),
            self.systems.base
        )
    }
}

impl fmt::Debug for TransferLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<'a> IntoIterator for &'a TransferPoset {
    type Item = &'a TransferSystem;
    type IntoIter = std::slice::Iter<'a, TransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().iter()
    }
}

impl IntoIterator for TransferPoset {
    type Item = TransferSystem;
    type IntoIter = std::vec::IntoIter<TransferSystem>;

    fn into_iter(self) -> Self::IntoIter {
        self.systems().to_vec().into_iter()
    }
}

impl fmt::Display for TransferPoset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "poset of {} transfer systems on {}",
            self.size(),
            self.systems.base
        )
    }
}

impl fmt::Debug for TransferPoset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn raw_contains_relation(lattice: &Lattice, arrows: &BitVec, relation: Edge) -> bool {
    if relation.is_identity() {
        return relation.from < lattice.size();
    }
    lattice
        .transfer_universe()
        .proper_edge_id(relation)
        .is_some_and(|edge_id| arrows[edge_id])
}

fn is_saturated_raw(lattice: &Lattice, arrows: &BitVec) -> bool {
    let universe = lattice.transfer_universe();
    for edge_id in arrows.iter_ones() {
        let edge = universe.proper_edges[edge_id];
        for middle in lattice.ids() {
            if lattice.leq(edge.from, middle)
                && lattice.leq(middle, edge.to)
                && !raw_contains_relation(lattice, arrows, Edge::new(middle, edge.to))
            {
                return false;
            }
        }
    }
    true
}

fn minimal_generator_bits(lattice: &Lattice, arrows: &BitVec) -> BitVec {
    let universe = lattice.transfer_universe();
    let mut generators = arrows.clone();
    for edge_id in (0..generators.len()).rev() {
        if !generators[edge_id] {
            continue;
        }
        generators.set(edge_id, false);
        if universe.close(&generators) != *arrows {
            generators.set(edge_id, true);
        }
    }
    generators
}

fn compatibility_failure_for_raw(
    lattice: &Lattice,
    additive: &BitVec,
    multiplicative: &BitVec,
) -> Option<CompatibilityFailure> {
    let universe = lattice.transfer_universe();
    if let Some(edge_id) = multiplicative
        .iter_ones()
        .find(|&edge_id| !additive[edge_id])
    {
        return Some(CompatibilityFailure::MultiplicativeNotAdditive {
            relation: universe.proper_edges[edge_id],
        });
    }

    for edge_id in multiplicative.iter_ones() {
        let multiplicative_edge = universe.proper_edges[edge_id];
        let k = multiplicative_edge.from;
        let h = multiplicative_edge.to;
        for j in lattice.ids() {
            if !lattice.leq(j, h) {
                continue;
            }
            let additive_edge = Edge::new(lattice.meet(k, j), k);
            let required = Edge::new(j, h);
            if raw_contains_relation(lattice, additive, additive_edge)
                && !raw_contains_relation(lattice, additive, required)
            {
                return Some(CompatibilityFailure::Distributivity {
                    multiplicative: multiplicative_edge,
                    additive: additive_edge,
                    required,
                });
            }
        }
    }
    None
}

fn build_transfer_context(lattice: &Lattice, proper_edges: &[Edge]) -> TransferContext {
    let mut attributes = proper_edges.to_vec();
    attributes.sort_unstable_by_key(|edge| (edge.to, edge.from));
    let matrix = proper_edges
        .iter()
        .map(|edge1| {
            attributes
                .iter()
                .map(|edge2| {
                    lattice.leq(edge2.to, edge1.from)
                        || !lattice.leq(edge2.to, edge1.to)
                        || !lattice.leq(edge2.from, edge1.from)
                })
                .collect()
        })
        .collect();
    FormalContext::new(proper_edges.to_vec(), attributes, matrix)
}

fn as_partial_order(lattice: &Lattice, arrows: &BitVec) -> PartialOrder {
    let universe = lattice.transfer_universe();
    let n = lattice.size();
    let mut pairs = Vec::with_capacity(n + arrows.count_ones());
    let mut matrix_transpose = vec![BitVec::repeat(false, n); n];

    for (id, column) in matrix_transpose.iter_mut().enumerate() {
        pairs.push(Edge::new(id, id));
        column.set(id, true);
    }

    for edge_id in arrows.iter_ones() {
        let edge = universe.proper_edges[edge_id];
        pairs.push(edge);
        matrix_transpose[edge.to].set(edge.from, true);
    }

    PartialOrder {
        pairs,
        matrix_transpose,
    }
}

/// Encodes a partial order on {1,...,n} in two ways:
/// First, as a list of pairs (i,j).
///
/// Second, as a list of columns of a binary matrix, where columns[j][i] is true
/// if and only if (i,j) is in the partial order.
#[derive(Debug, Clone)]
struct PartialOrder {
    pairs: Vec<Edge>,
    matrix_transpose: Vec<BitVec>,
}

/// Ambient intervals reused by every transfer-system factorization check.
struct FactorizationGeometry {
    intervals: Vec<Vec<BitVec>>,
}

impl FactorizationGeometry {
    fn new(poset: &Poset) -> Self {
        let intervals = poset
            .relation_matrix()
            .iter()
            .map(|upper_set| {
                poset
                    .relation_matrix_transpose()
                    .iter()
                    .map(|lower_set| intersection(upper_set, lower_set))
                    .collect()
            })
            .collect();
        Self { intervals }
    }

    fn interval(&self, lower: ElementId, upper: ElementId) -> &BitVec {
        &self.intervals[lower][upper]
    }
}

/// A pair of arrows witnessing failure of the composition-closed
/// factorization condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FactorizationFailure {
    /// The arrow in the smaller transfer system.
    pub(crate) first: Edge,
    /// The arrow in the larger transfer system.
    pub(crate) second: Edge,
}

fn factorization_condition(
    order: &[BitVec],
    geometry: &FactorizationGeometry,
    left: &PartialOrder,
    right: &PartialOrder,
) -> bool {
    factorization_failure(order, geometry, left, right).is_none()
}

fn factorization_failure(
    order: &[BitVec],
    geometry: &FactorizationGeometry,
    left: &PartialOrder,
    right: &PartialOrder,
) -> Option<FactorizationFailure> {
    for &first in &left.pairs {
        for &second in &right.pairs {
            if order[first.from][second.from]
                && order[first.to][second.to]
                && !has_factorization_witness(order, geometry, left, right, first, second)
            {
                return Some(FactorizationFailure { first, second });
            }
        }
    }
    None
}

/// Returns a failed pair of arrows when two transfer systems do not satisfy
/// the composition-closed factorization condition.
pub(crate) fn factorization_failure_between(
    left: &TransferSystem,
    right: &TransferSystem,
) -> Option<FactorizationFailure> {
    let lattice = &left.lattice;
    let geometry = FactorizationGeometry::new(lattice);
    factorization_failure(
        lattice.relation_matrix(),
        &geometry,
        &as_partial_order(lattice, &left.arrows),
        &as_partial_order(lattice, &right.arrows),
    )
}

fn has_factorization_witness(
    order: &[BitVec],
    geometry: &FactorizationGeometry,
    left: &PartialOrder,
    right: &PartialOrder,
    first: Edge,
    second: Edge,
) -> bool {
    let possible_z_primes = geometry.interval(first.from, second.from);
    for w_prime in right.matrix_transpose[second.to].iter_ones() {
        if order[first.to][w_prime]
            && intersects(&left.matrix_transpose[w_prime], possible_z_primes)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_context_uses_stable_objects_and_target_major_attributes() {
        let lattice = Lattice::chain(3);
        let universe = lattice.transfer_universe();

        assert_eq!(universe.context.objects, universe.proper_edges());
        assert!(
            universe
                .context
                .attributes
                .windows(2)
                .all(|edges| (edges[0].to, edges[0].from) <= (edges[1].to, edges[1].from))
        );
        assert_ne!(universe.context.objects, universe.context.attributes);
    }

    #[test]
    fn generated_transfer_system_has_restriction_closure() {
        let lattice = Lattice::chain(2);
        let generated = lattice
            .transfer_system_generated_by([Edge::new(0, 2), Edge::new(1, 1), Edge::new(0, 2)])
            .expect("all generators are lattice relations");
        assert_eq!(
            generated.edges(false),
            EdgeSet::from([Edge::new(0, 1), Edge::new(0, 2)])
        );
    }

    #[test]
    fn composition_order_enforces_the_factorization_condition() {
        let lattice = Lattice::chain(2);
        let left = lattice
            .transfer_system_generated_by([Edge::new(0, 1)])
            .expect("the generator is a lattice relation");
        let right = lattice
            .transfer_system_generated_by([Edge::new(0, 1), Edge::new(0, 2)])
            .expect("the generators are lattice relations");

        let order = lattice.relation_matrix();
        let geometry = FactorizationGeometry::new(&lattice);
        let left_order = as_partial_order(&lattice, left.arrows());
        let right_order = as_partial_order(&lattice, right.arrows());
        assert!(!factorization_condition(
            order,
            &geometry,
            &left_order,
            &right_order
        ));
        assert!(factorization_condition(
            order,
            &geometry,
            &left_order,
            &left_order
        ));
    }

    #[test]
    fn model_structure_order_has_the_expected_intervals_on_two_chain() {
        let lattice = Lattice::chain(2);
        let model_order = lattice.transfer_systems_model_structure_order();
        let composition_order = lattice.transfer_systems_composition_closed();

        assert_eq!(model_order.systems(), composition_order.systems());
        assert_eq!(model_order.all_relations_iter().count(), 10);
        assert!(
            model_order
                .all_relations_iter()
                .all(|edge| composition_order.leq(edge.from, edge.to))
        );
    }

    #[test]
    fn transfer_systems_print_with_labels() {
        let lattice = Lattice::from_covers(["0", "a", "1"], [("0", "a"), ("a", "1")]).unwrap();
        let system = lattice
            .transfer_system_generated_by([lattice.edge("a", "1").unwrap()])
            .unwrap();
        assert_eq!(system.to_string(), "{a -> 1}");
    }
}
