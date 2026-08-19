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
//! Computationally, identity relations are implicit and a system is a
//! bitvector selecting proper lattice relations `x < y`. A
//! [`RelationIndex`] specifies which relation each bit denotes, and a
//! [`TransferUniverse`] bundles that indexing with the closure machinery.
//!
//! # Enumeration algorithm
//!
//! The crate encodes restriction and transitivity as a formal context from
//! formal concept analysis (FCA), using the proper relations of `L` as both
//! objects and attributes. In FCA, applying derivation twice to a set of
//! objects gives a closure operator. For this particular context, the closed
//! object sets are exactly the transfer systems. Thus generation is one
//! double-derivation and enumeration uses the formal concepts of the context,
//! rather than testing all subsets of lattice relations independently.

use crate::bitvec_utils::{intersection, intersects, is_subset, set_partial_cmp};
use crate::lattice::{Lattice, LatticeError};
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError, compose};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::fmt;
use std::sync::Arc;

type TransferContext = FormalContext<Edge, Edge>;

/// A transfer system stored as a bitvector of non-identity lattice relations.
///
/// This raw form has no self-contained mathematical meaning: the ambient
/// lattice and the ordering of proper relations live in
/// [`TransferUniverse`].  Identity relations `x <= x` are not stored in the
/// bitvector; they are mathematically part of every transfer system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawTransferSystem {
    /// Bitmask of the non-identity arrows in the transfer system.
    arrows: BitVec,
}

impl PartialOrd for RawTransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        set_partial_cmp(&self.arrows, &other.arrows)
    }
}

/// A stable indexing of the proper relations of a finite lattice.
///
/// The index is the common coordinate system for ordinary and equivariant
/// transfer systems.  It orders the non-identity relations `x < y` of a
/// lattice in row-major order and provides constant-time lookup of the bit
/// assigned to a relation.  In particular, a G-transfer system can expand its
/// selected relation orbits into the raw bitvector of an ordinary transfer
/// system without allocating an intermediate relation set.
#[derive(Debug)]
pub struct RelationIndex<A> {
    lattice: Arc<Lattice<A>>,
    proper_edges: Vec<Edge>,
    proper_edge_ids: Vec<Vec<Option<usize>>>,
}

/// Shared ambient data that gives raw transfer-system bitsets their meaning.
///
/// A universe fixes the lattice `L`, a shared [`RelationIndex`], and the
/// formal context whose concepts enumerate transfer systems.  Constructing a
/// universe eagerly prepares that formal context, so generation, validation,
/// and enumeration reuse the same relation coordinates and closure operator.
#[derive(Debug)]
pub struct TransferUniverse<A> {
    relation_index: Arc<RelationIndex<A>>,
    /// The formal context whose concepts correspond to transfer systems.
    context: TransferContext,
}

/// An owned transfer system together with its ambient lattice data.
///
/// This is the user-facing form of a transfer system: it pairs a raw bitvector
/// with the universe needed to interpret each bit as a relation in the
/// underlying lattice.
#[derive(Debug)]
pub struct TransferSystem<A> {
    raw: RawTransferSystem,
    universe: Arc<TransferUniverse<A>>,
}

/// A poset of transfer systems on a fixed lattice.
///
/// This wrapper is used for orders on transfer systems that need not themselves
/// be lattices, such as the composition-closed order.
#[derive(Debug, Clone)]
pub struct TransferPoset<A> {
    universe: Arc<TransferUniverse<A>>,
    poset: Poset<RawTransferSystem>,
}

/// A lattice of transfer systems on a fixed lattice.
///
/// For the full containment lattice, meet is intersection and join is the
/// transfer-system closure of union.  The same wrapper also represents the
/// containment lattice of saturated systems, where join additionally takes
/// saturated closure.
#[derive(Debug, Clone)]
pub struct TransferLattice<A> {
    universe: Arc<TransferUniverse<A>>,
    lattice: Lattice<RawTransferSystem>,
}

/// A concrete reason that an additive and multiplicative transfer system are
/// not compatible.
///
/// Compatibility uses the convention that the first system is additive and
/// the second is multiplicative.  In particular, every multiplicative
/// transfer must also be additive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityFailure {
    /// The two transfer systems do not share the same transfer universe.
    DifferentUniverses,
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

/// Errors that can occur while constructing transfer-system orders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// Construction of an underlying finite poset failed.
    Poset(PosetError),
    /// Construction of an underlying finite lattice failed.
    Lattice(LatticeError),
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
    /// A raw bitvector does not have one bit for every proper lattice relation.
    WrongArrowCount {
        /// The number of proper lattice relations in the universe.
        expected: usize,
        /// The number of bits in the raw representation.
        actual: usize,
    },
    /// A raw bitvector is not closed under the transfer-system axioms.
    RawNotClosed,
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferError::Poset(error) => write!(f, "{error}"),
            TransferError::Lattice(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TransferError {}

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
            TransferSystemError::WrongArrowCount { expected, actual } => write!(
                f,
                "raw transfer system has {actual} arrow bits, expected {expected}"
            ),
            TransferSystemError::RawNotClosed => {
                write!(
                    f,
                    "raw arrow set is not closed under the transfer-system axioms"
                )
            }
        }
    }
}

impl std::error::Error for TransferSystemError {}

impl From<PosetError> for TransferError {
    fn from(error: PosetError) -> Self {
        Self::Poset(error)
    }
}

impl From<LatticeError> for TransferError {
    fn from(error: LatticeError) -> Self {
        Self::Lattice(error)
    }
}

impl<A> Lattice<A> {
    /// Builds the shared universe used to enumerate transfer systems on `self`.
    ///
    /// The universe records all non-identity relations of the lattice and the
    /// formal context whose concepts correspond to transfer systems.
    pub fn transfer_universe(self: Arc<Self>) -> Arc<TransferUniverse<A>> {
        Arc::new(TransferUniverse::new(self))
    }

    /// Constructs the lattice of transfer systems ordered by containment.
    ///
    /// A transfer system is below another precisely when its set of
    /// non-identity relations is a subset of the other's.
    pub fn transfer_systems_containment(
        self: Arc<Self>,
    ) -> Result<TransferLattice<A>, TransferError> {
        self.transfer_universe().containment_lattice()
    }

    /// Counts the transfer systems on this lattice without storing them.
    ///
    /// This traverses the formal concepts of the transfer context but does not
    /// construct the transfer systems or their containment lattice.
    pub fn transfer_system_count(self: Arc<Self>) -> usize {
        self.transfer_universe().transfer_system_count()
    }

    /// Constructs the lattice of saturated transfer systems under
    /// containment.
    pub fn saturated_transfer_systems_containment(
        self: Arc<Self>,
    ) -> Result<TransferLattice<A>, TransferError> {
        self.transfer_universe().saturated_containment_lattice()
    }

    /// Returns the complexity of this lattice.
    ///
    /// This is the largest generator complexity among all transfer systems on
    /// the lattice.
    pub fn transfer_system_complexity(self: Arc<Self>) -> usize {
        self.transfer_universe().complexity()
    }

    /// Returns the transfer-system width of this lattice.
    ///
    /// The width is the generator complexity of the complete transfer system.
    pub fn transfer_system_width(self: Arc<Self>) -> usize {
        self.transfer_universe().width()
    }

    /// Constructs the composition-closed order on transfer systems.
    ///
    /// This order requires both containment and the factorization condition
    /// described by [`TransferUniverse::composition_closed_order`].
    pub fn transfer_systems_composition_closed(
        self: Arc<Self>,
    ) -> Result<TransferPoset<A>, TransferError> {
        self.transfer_universe().composition_closed_order()
    }

    /// Constructs the model-structure order on transfer systems.
    ///
    /// An interval `R <= R'` belongs to this order precisely when the weak
    /// equivalences `R ∘ llc(R')` have the 2-out-of-3 property.
    pub fn transfer_systems_model_structure_order(
        self: Arc<Self>,
    ) -> Result<TransferPoset<A>, TransferError> {
        self.transfer_universe().model_structure_order()
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
    fn new<A>(lattice: &Lattice<A>) -> Self {
        let poset = lattice.as_poset();
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

impl RawTransferSystem {
    pub(crate) fn new(arrows: BitVec) -> Self {
        Self { arrows }
    }

    /// Returns the bitvector of non-identity relations.
    ///
    /// Bit `i` corresponds to `universe.proper_edges()[i]` for any
    /// [`TransferUniverse`] interpreting this raw transfer system.
    pub fn arrows(&self) -> &BitVec {
        &self.arrows
    }

    fn as_partial_order<A>(&self, universe: &TransferUniverse<A>) -> PartialOrder {
        let n = universe.underlying_lattice().size();
        let mut pairs = Vec::with_capacity(n + self.arrows().count_ones());
        let mut matrix_transpose = vec![BitVec::repeat(false, n); n];

        for (id, column) in matrix_transpose.iter_mut().enumerate() {
            pairs.push(Edge::new(id, id));
            column.set(id, true);
        }

        for edge_id in self.arrows().iter_ones() {
            let edge = universe.proper_edges()[edge_id];
            pairs.push(edge);
            matrix_transpose[edge.to].set(edge.from, true);
        }

        PartialOrder {
            pairs,
            matrix_transpose,
        }
    }
}

impl<A> RelationIndex<A> {
    /// Constructs the row-major index of all proper relations of `lattice`.
    pub fn new(lattice: Arc<Lattice<A>>) -> Self {
        let proper_edges = lattice
            .as_poset()
            .proper_relations_iter()
            .collect::<Vec<_>>();
        let mut proper_edge_ids = vec![vec![None; lattice.size()]; lattice.size()];
        for (edge_id, &edge) in proper_edges.iter().enumerate() {
            proper_edge_ids[edge.from][edge.to] = Some(edge_id);
        }

        Self {
            lattice,
            proper_edges,
            proper_edge_ids,
        }
    }

    /// Returns the lattice whose proper relations are indexed.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.lattice
    }

    /// Returns the proper lattice relations in deterministic row-major order.
    pub fn proper_edges(&self) -> &[Edge] {
        &self.proper_edges
    }

    /// Returns the bit index assigned to a proper relation.
    ///
    /// Identity relations, non-relations, and out-of-range edges have no bit
    /// index and therefore return `None`.
    pub fn proper_edge_id(&self, edge: Edge) -> Option<usize> {
        self.proper_edge_ids
            .get(edge.from)
            .and_then(|row| row.get(edge.to))
            .copied()
            .flatten()
    }
}

impl<A> TransferUniverse<A> {
    /// Constructs the transfer-system universe for a lattice.
    ///
    /// This indexes every proper relation and builds the FCA incidence matrix
    /// used for closure and enumeration. The potentially substantial setup is
    /// therefore paid once and shared by all systems in the universe.
    pub fn new(underlying_lattice: Arc<Lattice<A>>) -> Self {
        Self::from_relation_index(Arc::new(RelationIndex::new(underlying_lattice)))
    }

    /// Constructs a transfer-system universe using an existing relation index.
    ///
    /// This crate-private constructor lets a G-transfer universe share the
    /// ordinary relation coordinates needed for the fixed-point expansion.
    pub(crate) fn from_relation_index(relation_index: Arc<RelationIndex<A>>) -> Self {
        let context =
            build_transfer_context(relation_index.lattice(), relation_index.proper_edges());
        Self {
            relation_index,
            context,
        }
    }

    /// Returns the underlying lattice.
    pub fn underlying_lattice(&self) -> &Arc<Lattice<A>> {
        self.relation_index.lattice()
    }

    /// Returns the underlying lattice.
    ///
    /// This is an alias for [`TransferUniverse::underlying_lattice`].
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.underlying_lattice()
    }

    /// Returns the shared indexing of the proper lattice relations.
    pub fn relation_index(&self) -> &Arc<RelationIndex<A>> {
        &self.relation_index
    }

    fn context(&self) -> &TransferContext {
        &self.context
    }

    /// Returns the proper lattice relations used as transfer-system generators.
    ///
    /// These are the non-identity relations `x < y`, in deterministic
    /// row-major order inherited from the underlying poset.
    pub fn proper_edges(&self) -> &[Edge] {
        self.relation_index.proper_edges()
    }

    /// Constructs the transfer system generated by the supplied relations.
    ///
    /// Identity relations may be supplied but need not be: they belong to
    /// every transfer system and are not stored in the raw bitvector.  Every
    /// non-identity generator must be a relation in the underlying lattice
    /// order.  The returned system is the least transfer system containing all
    /// of the generators.
    pub fn generated_by<I, E>(
        self: &Arc<Self>,
        generators: I,
    ) -> Result<TransferSystem<A>, TransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let lattice_size = self.underlying_lattice().size();
        let mut arrows = BitVec::repeat(false, self.proper_edges().len());

        for generator in generators {
            let edge = generator.into();
            if edge.from >= lattice_size || edge.to >= lattice_size {
                return Err(TransferSystemError::EdgeOutOfBounds { edge, lattice_size });
            }
            if edge.is_identity() {
                continue;
            }

            let Some(edge_id) = self.relation_index().proper_edge_id(edge) else {
                return Err(TransferSystemError::NotLatticeRelation { edge });
            };
            arrows.set(edge_id, true);
        }

        let raw = RawTransferSystem::new(self.close_arrows(&arrows));
        Ok(TransferSystem::new(raw, Arc::clone(self)))
    }

    /// Validates raw bitvector data and pairs it with this universe.
    ///
    /// The raw bitvector must have exactly one bit for each proper lattice
    /// relation and must already be closed under the transfer-system axioms.
    /// Use [`TransferUniverse::generated_by`] when closure should be added
    /// automatically.
    pub fn try_from_raw(
        self: &Arc<Self>,
        raw: RawTransferSystem,
    ) -> Result<TransferSystem<A>, TransferSystemError> {
        let expected = self.proper_edges().len();
        let actual = raw.arrows().len();
        if actual != expected {
            return Err(TransferSystemError::WrongArrowCount { expected, actual });
        }
        if self.close_arrows(raw.arrows()) != *raw.arrows() {
            return Err(TransferSystemError::RawNotClosed);
        }

        Ok(TransferSystem::new(raw, Arc::clone(self)))
    }

    /// Closes a correctly sized bitvector under the transfer-system axioms.
    ///
    /// Transfer systems are precisely the closed extents of the universe's
    /// formal context, so the closure is the FCA double-prime operation.
    pub(crate) fn close_arrows(&self, arrows: &BitVec) -> BitVec {
        debug_assert_eq!(arrows.len(), self.proper_edges().len());
        self.context.induce_l(&self.context.induce_r(arrows))
    }

    /// Enumerates all transfer systems on the underlying lattice.
    ///
    /// Each result shares this universe, so its raw bitvector can be decoded
    /// using [`TransferUniverse::proper_edges`]. Enumeration delegates to the
    /// formal-context concept algorithm; each concept extent is one closed
    /// relation set.
    pub fn transfer_systems(self: &Arc<Self>) -> Vec<TransferSystem<A>> {
        all_transfer_systems(self)
            .into_iter()
            .map(|raw| TransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    /// Counts all transfer systems without storing one value per system.
    pub fn transfer_system_count(&self) -> usize {
        self.context.num_concepts()
    }

    /// Enumerates all saturated transfer systems on the underlying lattice.
    ///
    /// A transfer system is saturated when `x -> z` and `x <= y <= z`
    /// force `y -> z`.
    pub fn saturated_transfer_systems(self: &Arc<Self>) -> Vec<TransferSystem<A>> {
        all_saturated_transfer_systems(self)
            .into_iter()
            .map(|raw| TransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    /// Enumerates all compatible additive--multiplicative pairs.
    ///
    /// Each tuple is ordered `(additive, multiplicative)`.  The systems share
    /// this universe and satisfy the containment and distributivity conditions
    /// checked by [`TransferSystem::is_compatible_with`].
    pub fn compatible_pairs(self: &Arc<Self>) -> Vec<(TransferSystem<A>, TransferSystem<A>)> {
        let systems = all_transfer_systems(self);
        let mut pairs = Vec::new();
        for additive in &systems {
            for multiplicative in &systems {
                if compatibility_failure_for_raw(self, additive, multiplicative).is_none() {
                    pairs.push((
                        TransferSystem::new(additive.clone(), Arc::clone(self)),
                        TransferSystem::new(multiplicative.clone(), Arc::clone(self)),
                    ));
                }
            }
        }
        pairs
    }

    /// Returns the complexity of the underlying lattice.
    ///
    /// The complexity is the maximum size of a minimal generating set over
    /// all transfer systems.  Minimal generating sets of a transfer system all
    /// have the same cardinality.
    pub fn complexity(self: &Arc<Self>) -> usize {
        all_transfer_systems(self)
            .iter()
            .map(|raw| minimal_generator_bits(self, raw).count_ones())
            .max()
            .unwrap_or(0)
    }

    /// Returns the transfer-system width of the underlying lattice.
    ///
    /// This is the size of a minimal generating set of the complete transfer
    /// system.
    pub fn width(self: &Arc<Self>) -> usize {
        let complete = RawTransferSystem::new(BitVec::repeat(true, self.proper_edges().len()));
        minimal_generator_bits(self, &complete).count_ones()
    }

    /// Constructs the lattice of transfer systems ordered by containment.
    ///
    /// Under this order, meet is intersection. Join starts with the union and
    /// then adds all relations forced by transitivity and restriction closure.
    pub fn containment_lattice(self: &Arc<Self>) -> Result<TransferLattice<A>, TransferError> {
        Ok(containment_lattice(
            Arc::clone(self),
            all_transfer_systems(self),
        )?)
    }

    /// Constructs the lattice of saturated transfer systems under
    /// containment.
    ///
    /// Meets are intersections.  Joins are obtained by taking ordinary
    /// transfer-system closure of the union and then saturated closure.
    pub fn saturated_containment_lattice(
        self: &Arc<Self>,
    ) -> Result<TransferLattice<A>, TransferError> {
        Ok(containment_lattice(
            Arc::clone(self),
            all_saturated_transfer_systems(self),
        )?)
    }

    /// Constructs the composition-closed order on transfer systems.
    ///
    /// The algorithm first enumerates the systems, then compares every ordered
    /// pair. A comparison requires ordinary containment and the factorization
    /// condition: each relevant square formed by one arrow from the smaller
    /// system and one from the larger must admit an intermediate
    /// factorization using those two systems. Unlike containment, this order
    /// need not be a lattice.
    pub fn composition_closed_order(self: &Arc<Self>) -> Result<TransferPoset<A>, TransferError> {
        Ok(composition_closed_order(
            Arc::clone(self),
            all_transfer_systems(self),
        )?)
    }

    /// Constructs the model-structure order on transfer systems.
    ///
    /// For transfer systems `R <= R'`, the corresponding premodel structure
    /// has weak equivalences `R ∘ llc(R')`. The pair is included exactly when
    /// those weak equivalences satisfy 2-out-of-3. Thus the intervals of this
    /// order are precisely the model structures on the underlying lattice.
    pub fn model_structure_order(self: &Arc<Self>) -> Result<TransferPoset<A>, TransferError> {
        Ok(model_structure_order(
            Arc::clone(self),
            all_transfer_systems(self),
        )?)
    }
}

fn all_transfer_systems<A>(universe: &TransferUniverse<A>) -> Vec<RawTransferSystem> {
    universe
        .context()
        .all_concepts_raw()
        .into_iter()
        .map(|concept| RawTransferSystem::new(concept.extent))
        .collect()
}

fn all_saturated_transfer_systems<A>(universe: &TransferUniverse<A>) -> Vec<RawTransferSystem> {
    all_transfer_systems(universe)
        .into_iter()
        .filter(|raw| is_saturated_raw(universe, raw))
        .collect()
}

impl<A> TransferSystem<A> {
    /// Pairs raw transfer-system data with its ambient universe.
    pub(crate) fn new(raw: RawTransferSystem, universe: Arc<TransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    /// Returns the raw bitvector representation.
    pub fn raw(&self) -> &RawTransferSystem {
        &self.raw
    }

    /// Returns the ambient universe.
    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    /// Returns whether a relation belongs to this transfer system.
    ///
    /// Every in-range identity relation belongs to a transfer system.  A
    /// non-identity relation belongs precisely when its corresponding bit is
    /// selected; non-relations and out-of-range edges return `false`.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }

        self.universe
            .relation_index()
            .proper_edge_id(relation)
            .is_some_and(|edge_id| self.raw.arrows()[edge_id])
    }

    /// Returns the relations belonging to this transfer system.
    ///
    /// If `include_identities` is true, the identity relations `x <= x` are
    /// included along with the stored non-identity relations.
    pub fn edges(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            for id in 0..self.lattice().size() {
                result.insert(Edge::new(id, id));
            }
        }
        result.extend(
            self.raw
                .arrows
                .iter_ones()
                .map(|edge_id| self.universe.proper_edges()[edge_id]),
        );
        result
    }

    /// Returns whether this transfer system is saturated.
    ///
    /// Saturation says that whenever `x -> z` belongs to the system and
    /// `x <= y <= z`, the relation `y -> z` also belongs to the system.  The
    /// other relation `x -> y` is already forced by restriction closure.
    pub fn is_saturated(&self) -> bool {
        is_saturated_raw(&self.universe, &self.raw)
    }

    /// Returns the least saturated transfer system containing this one.
    pub fn saturated_closure(&self) -> TransferSystem<A> {
        let raw = saturated_closure_raw(&self.universe, &self.raw);
        TransferSystem::new(raw, Arc::clone(&self.universe))
    }

    /// Returns whether this transfer system is cosaturated.
    ///
    /// A cosaturated transfer system is generated by its relations whose
    /// target is the top of the lattice.  Such systems are also called
    /// *disklike* in the transfer-system literature.
    pub fn is_cosaturated(&self) -> bool {
        self.raw == cosaturated_coclosure_raw(&self.universe, &self.raw)
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
    pub fn cosaturated_coclosure(&self) -> TransferSystem<A> {
        let raw = cosaturated_coclosure_raw(&self.universe, &self.raw);
        TransferSystem::new(raw, Arc::clone(&self.universe))
    }

    /// Returns whether the supplied relations generate this transfer system.
    ///
    /// Identity relations may be supplied but are never needed.  Invalid
    /// lattice relations produce the same errors as
    /// [`TransferUniverse::generated_by`].
    pub fn is_generated_by<I, E>(&self, generators: I) -> Result<bool, TransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        Ok(self.universe.generated_by(generators)?.raw.eq(&self.raw))
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
    /// Relations are considered in reverse order of the universe's stable
    /// relation index and deleted whenever the remaining relations still
    /// generate this system.
    pub fn minimal_generating_set(&self) -> EdgeSet {
        generator_edges_from_bits(
            &self.universe,
            &minimal_generator_bits(&self.universe, &self.raw),
        )
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
        minimal_generator_bits(&self.universe, &self.raw).count_ones()
    }

    /// Returns a concrete failure when this additive transfer system is not
    /// compatible with `multiplicative`.
    ///
    /// Compatibility first requires every multiplicative relation to be
    /// additive.  It then requires that, for `K, J <= H`, a multiplicative
    /// `K -> H` and additive `(K /\ J) -> K` force additive `J -> H`.
    pub fn compatibility_failure(
        &self,
        multiplicative: &TransferSystem<A>,
    ) -> Option<CompatibilityFailure> {
        if !Arc::ptr_eq(&self.universe, &multiplicative.universe) {
            return Some(CompatibilityFailure::DifferentUniverses);
        }
        compatibility_failure_for_raw(&self.universe, &self.raw, &multiplicative.raw)
    }

    /// Returns whether this additive transfer system is compatible with the
    /// supplied multiplicative transfer system.
    pub fn is_compatible_with(&self, multiplicative: &TransferSystem<A>) -> bool {
        self.compatibility_failure(multiplicative).is_none()
    }
}

fn raw_contains_relation<A>(
    universe: &TransferUniverse<A>,
    raw: &RawTransferSystem,
    relation: Edge,
) -> bool {
    if relation.is_identity() {
        return relation.from < universe.lattice().size();
    }
    universe
        .relation_index()
        .proper_edge_id(relation)
        .is_some_and(|edge_id| raw.arrows()[edge_id])
}

fn is_saturated_raw<A>(universe: &TransferUniverse<A>, raw: &RawTransferSystem) -> bool {
    let lattice = universe.lattice();
    for edge_id in raw.arrows().iter_ones() {
        let edge = universe.proper_edges()[edge_id];
        for middle in 0..lattice.size() {
            if lattice.leq(edge.from, middle)
                && lattice.leq(middle, edge.to)
                && !raw_contains_relation(universe, raw, Edge::new(middle, edge.to))
            {
                return false;
            }
        }
    }
    true
}

fn saturated_closure_raw<A>(
    universe: &TransferUniverse<A>,
    raw: &RawTransferSystem,
) -> RawTransferSystem {
    let lattice = universe.lattice();
    let mut arrows = raw.arrows().clone();
    loop {
        let previous = arrows.clone();
        for edge_id in previous.iter_ones() {
            let edge = universe.proper_edges()[edge_id];
            for middle in 0..lattice.size() {
                let saturated_edge = Edge::new(middle, edge.to);
                if lattice.leq(edge.from, middle)
                    && lattice.leq(middle, edge.to)
                    && !saturated_edge.is_identity()
                {
                    let saturated_id = universe
                        .relation_index()
                        .proper_edge_id(saturated_edge)
                        .expect("a proper intermediate relation should be indexed");
                    arrows.set(saturated_id, true);
                }
            }
        }
        arrows = universe.close_arrows(&arrows);
        if arrows == previous {
            return RawTransferSystem::new(arrows);
        }
    }
}

fn cosaturated_coclosure_raw<A>(
    universe: &TransferUniverse<A>,
    raw: &RawTransferSystem,
) -> RawTransferSystem {
    let top = universe.lattice().top();
    let mut top_arrows = BitVec::repeat(false, universe.proper_edges().len());
    for edge_id in raw.arrows().iter_ones() {
        if universe.proper_edges()[edge_id].to == top {
            top_arrows.set(edge_id, true);
        }
    }
    RawTransferSystem::new(universe.close_arrows(&top_arrows))
}

fn minimal_generator_bits<A>(universe: &TransferUniverse<A>, raw: &RawTransferSystem) -> BitVec {
    let mut generators = raw.arrows().clone();
    for edge_id in (0..generators.len()).rev() {
        if !generators[edge_id] {
            continue;
        }
        generators.set(edge_id, false);
        if universe.close_arrows(&generators) != *raw.arrows() {
            generators.set(edge_id, true);
        }
    }
    generators
}

fn generator_edges_from_bits<A>(universe: &TransferUniverse<A>, generators: &BitVec) -> EdgeSet {
    generators
        .iter_ones()
        .map(|edge_id| universe.proper_edges()[edge_id])
        .collect()
}

fn compatibility_failure_for_raw<A>(
    universe: &TransferUniverse<A>,
    additive: &RawTransferSystem,
    multiplicative: &RawTransferSystem,
) -> Option<CompatibilityFailure> {
    if let Some(edge_id) = multiplicative
        .arrows()
        .iter_ones()
        .find(|&edge_id| !additive.arrows()[edge_id])
    {
        return Some(CompatibilityFailure::MultiplicativeNotAdditive {
            relation: universe.proper_edges()[edge_id],
        });
    }

    let lattice = universe.lattice();
    for edge_id in multiplicative.arrows().iter_ones() {
        let multiplicative_edge = universe.proper_edges()[edge_id];
        let k = multiplicative_edge.from;
        let h = multiplicative_edge.to;
        for j in 0..lattice.size() {
            if !lattice.leq(j, h) {
                continue;
            }
            let additive_edge = Edge::new(lattice.meet_id(k, j), k);
            let required = Edge::new(j, h);
            if raw_contains_relation(universe, additive, additive_edge)
                && !raw_contains_relation(universe, additive, required)
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

fn build_transfer_context<A>(lattice: &Lattice<A>, proper_edges: &[Edge]) -> TransferContext {
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

impl<A> Clone for TransferSystem<A> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            universe: Arc::clone(&self.universe),
        }
    }
}

impl<A> PartialEq for TransferSystem<A> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && Arc::ptr_eq(&self.universe, &other.universe)
    }
}

impl<A> Eq for TransferSystem<A> {}

impl<A> TransferPoset<A> {
    fn new(universe: Arc<TransferUniverse<A>>, poset: Poset<RawTransferSystem>) -> Self {
        Self { universe, poset }
    }

    /// Returns the universe shared by all systems in this poset.
    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    /// Returns the raw poset whose labels are bitvector transfer systems.
    pub fn raw_poset(&self) -> &Poset<RawTransferSystem> {
        &self.poset
    }

    /// Returns the number of transfer systems in the poset.
    pub fn size(&self) -> usize {
        self.poset.size()
    }

    /// Returns the cover relations in the Hasse diagram of this poset.
    pub fn cover_relations(&self) -> EdgeSet {
        self.poset.cover_relations()
    }

    /// Returns a transfer system by element id in this poset.
    pub fn system(&self, id: ElementId) -> Option<TransferSystem<A>> {
        self.poset
            .element(id)
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Iterates over all transfer systems in element-id order.
    pub fn systems(&self) -> impl Iterator<Item = TransferSystem<A>> + '_ {
        self.poset
            .elements()
            .iter()
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Relabels the raw poset by user-facing [`TransferSystem`] values.
    pub fn to_system_poset(&self) -> Poset<TransferSystem<A>> {
        self.poset
            .relabelled(|raw| TransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

impl<A> TransferLattice<A> {
    fn new(universe: Arc<TransferUniverse<A>>, lattice: Lattice<RawTransferSystem>) -> Self {
        Self { universe, lattice }
    }

    /// Returns the universe shared by all systems in this lattice.
    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    /// Returns the raw lattice whose labels are bitvector transfer systems.
    pub fn raw_lattice(&self) -> &Lattice<RawTransferSystem> {
        &self.lattice
    }

    /// Returns the underlying poset of the transfer-system lattice.
    pub fn as_poset(&self) -> &Poset<RawTransferSystem> {
        self.lattice.as_poset()
    }

    /// Returns the number of transfer systems in the lattice.
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Returns the meet of two transfer systems by element id.
    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    /// Returns the join of two transfer systems by element id.
    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    /// Returns the bottom transfer system.
    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    /// Returns the top transfer system.
    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    /// Returns a transfer system by element id in this lattice.
    pub fn system(&self, id: ElementId) -> Option<TransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Iterates over all transfer systems in element-id order.
    pub fn systems(&self) -> impl Iterator<Item = TransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Relabels the raw lattice by user-facing [`TransferSystem`] values.
    pub fn to_system_lattice(&self) -> Lattice<TransferSystem<A>> {
        self.lattice
            .relabelled(|raw| TransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }

    /// Relabels the underlying containment poset by user-facing
    /// [`TransferSystem`] values.
    ///
    /// This avoids copying the lattice's meet and join tables when only its
    /// order is needed.
    pub fn to_system_poset(&self) -> Poset<TransferSystem<A>> {
        self.as_poset()
            .relabelled(|raw| TransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
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

/// Returns a failed pair of arrows when two raw systems do not satisfy the
/// composition-closed factorization condition.
///
/// Callers are responsible for supplying raw systems belonging to `universe`.
pub(crate) fn factorization_failure_for_raw<A>(
    universe: &TransferUniverse<A>,
    left: &RawTransferSystem,
    right: &RawTransferSystem,
) -> Option<FactorizationFailure> {
    let left = left.as_partial_order(universe);
    let right = right.as_partial_order(universe);
    let geometry = FactorizationGeometry::new(universe.underlying_lattice());
    factorization_failure(
        universe.underlying_lattice().as_poset().relation_matrix(),
        &geometry,
        &left,
        &right,
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

fn containment_lattice<A>(
    universe: Arc<TransferUniverse<A>>,
    systems: Vec<RawTransferSystem>,
) -> Result<TransferLattice<A>, LatticeError> {
    let poset = transfer_systems_ordered_by(systems, |left, right| {
        is_subset(left.arrows(), right.arrows())
    })?;
    Ok(TransferLattice::new(universe, Lattice::new(poset)?))
}

fn transfer_systems_ordered_by<F>(
    systems: Vec<RawTransferSystem>,
    predicate: F,
) -> Result<Poset<RawTransferSystem>, PosetError>
where
    F: Fn(&RawTransferSystem, &RawTransferSystem) -> bool,
{
    let relation = systems
        .iter()
        .map(|left| systems.iter().map(|right| predicate(left, right)).collect())
        .collect();
    Poset::from_relation(systems, relation)
}

fn composition_closed_order<A>(
    universe: Arc<TransferUniverse<A>>,
    systems: Vec<RawTransferSystem>,
) -> Result<TransferPoset<A>, PosetError> {
    let order = universe.underlying_lattice().as_poset().relation_matrix();
    let geometry = FactorizationGeometry::new(universe.underlying_lattice());
    let partial_orders = systems
        .iter()
        .map(|raw| raw.as_partial_order(&universe))
        .collect::<Vec<_>>();

    let relation = (0..systems.len())
        .map(|left| {
            (0..systems.len())
                .map(|right| {
                    is_subset(systems[left].arrows(), systems[right].arrows())
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
    Ok(TransferPoset::new(
        universe,
        Poset::from_relation(systems, relation)?,
    ))
}

fn model_structure_order<A>(
    universe: Arc<TransferUniverse<A>>,
    systems: Vec<RawTransferSystem>,
) -> Result<TransferPoset<A>, PosetError> {
    let ambient = universe.underlying_lattice().as_poset();
    let right_classes = systems
        .iter()
        .map(|system| {
            let mut edges = EdgeSet::from_iter(
                system
                    .arrows()
                    .iter_ones()
                    .map(|edge_id| universe.proper_edges()[edge_id]),
            );
            edges.extend((0..ambient.size()).map(|id| Edge::new(id, id)));
            edges
        })
        .collect::<Vec<_>>();
    let left_classes = right_classes
        .iter()
        .map(|right| ambient.llc(right))
        .collect::<Vec<_>>();

    let relation = (0..systems.len())
        .map(|lower| {
            (0..systems.len())
                .map(|upper| {
                    is_subset(systems[lower].arrows(), systems[upper].arrows())
                        && ambient
                            .two_out_of_three(&compose(&right_classes[lower], &left_classes[upper]))
                })
                .collect()
        })
        .collect();
    Ok(TransferPoset::new(
        universe,
        Poset::from_relation(systems, relation)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_context_uses_stable_objects_and_target_major_attributes() {
        let lattice = Arc::new(Lattice::chain(3).expect("the finite chain is a lattice"));
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
        let lattice = Arc::new(Lattice::chain(2).expect("the finite chain is a lattice"));
        let universe = lattice.transfer_universe();

        let generated = universe
            .generated_by([Edge::new(0, 2), Edge::new(1, 1), Edge::new(0, 2)])
            .expect("all generators are lattice relations");
        assert_eq!(
            generated.edges(false),
            EdgeSet::from([Edge::new(0, 1), Edge::new(0, 2)])
        );
    }

    #[test]
    fn composition_order_enforces_the_factorization_condition() {
        let lattice = Arc::new(Lattice::chain(2).expect("the finite chain is a lattice"));
        let universe = lattice.transfer_universe();
        let left = universe
            .generated_by([Edge::new(0, 1)])
            .expect("the generator is a lattice relation");
        let right = universe
            .generated_by([Edge::new(0, 1), Edge::new(0, 2)])
            .expect("the generators are lattice relations");

        let order = universe.lattice().as_poset().relation_matrix();
        let geometry = FactorizationGeometry::new(universe.lattice());
        let left_order = left.raw().as_partial_order(&universe);
        let right_order = right.raw().as_partial_order(&universe);
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
        let universe =
            Arc::new(Lattice::chain(2).expect("the finite chain is a lattice")).transfer_universe();
        let model_order = universe
            .model_structure_order()
            .expect("model structures define a partial order");
        let composition_order = universe
            .composition_closed_order()
            .expect("composition-closed premodels define a partial order");

        assert_eq!(
            model_order.raw_poset().elements(),
            composition_order.raw_poset().elements()
        );
        assert_eq!(model_order.raw_poset().all_relations_iter().count(), 10);
        assert!(
            model_order
                .raw_poset()
                .all_relations_iter()
                .all(|edge| composition_order.raw_poset().leq(edge.from, edge.to))
        );
    }
}
