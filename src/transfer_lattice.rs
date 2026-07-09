//! Transfer systems on finite lattices.
//!
//! For a finite lattice `L`, a transfer system is represented here as a
//! reflexive partial order on the elements of `L` that is contained in the
//! lattice order and satisfies the usual restriction/factorization closure
//! condition.  Computationally, the identity relations are implicit and a
//! transfer system is stored as a bitvector of selected non-identity relations
//! `x < y`.
//!
//! The enumeration uses formal concept analysis: the proper relations of `L`
//! are used as both objects and attributes in a formal context, and formal
//! concepts of this context correspond to transfer systems.

use crate::lattice::{Lattice, LatticeError};
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::fmt;
use std::sync::Arc;

type TransferContext = FormalContext<Edge, Edge>;

/// A transfer system stored as a bitvector of non-identity lattice relations.
///
/// The ambient lattice and the ordering of proper relations live in
/// [`TransferUniverse`].  Identity relations `x <= x` are not stored in the
/// bitvector; they are mathematically part of every transfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTransferSystem {
    /// Bitmask of the non-identity arrows in the transfer system.
    arrows: BitVec,
}

impl PartialOrd for RawTransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.arrows == other.arrows {
            Some(std::cmp::Ordering::Equal)
        } else if bitvec_subset(&self.arrows, &other.arrows) {
            Some(std::cmp::Ordering::Less)
        } else if bitvec_subset(&other.arrows, &self.arrows) {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
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
/// universe eagerly prepares that formal context, so the universe is always a
/// complete ambient object for formal-concept calculations.
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
/// For the containment order, meet is intersection of transfer systems and join
/// is the transfer-system closure of union.
#[derive(Debug, Clone)]
pub struct TransferLattice<A> {
    universe: Arc<TransferUniverse<A>>,
    lattice: Lattice<RawTransferSystem>,
}

/// Errors that can occur while constructing transfer-system orders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// Construction of an underlying finite poset failed.
    Poset(PosetError),
    /// Construction of an underlying finite lattice failed.
    Lattice(LatticeError),
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

    /// Constructs the composition-closed order on transfer systems.
    ///
    /// This order refines containment by the factorization condition used in
    /// the transfer-system literature.
    pub fn transfer_systems_composition_closed(
        self: Arc<Self>,
    ) -> Result<TransferPoset<A>, TransferError> {
        self.transfer_universe().composition_closed_order()
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

    /// Enumerates all transfer systems on the underlying lattice.
    ///
    /// Each result shares this universe, so its raw bitvector can be decoded
    /// using [`TransferUniverse::proper_edges`].
    pub fn transfer_systems(self: &Arc<Self>) -> Vec<TransferSystem<A>> {
        all_transfer_systems(self)
            .into_iter()
            .map(|raw| TransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    /// Constructs the lattice of transfer systems ordered by containment.
    pub fn containment_lattice(self: &Arc<Self>) -> Result<TransferLattice<A>, TransferError> {
        Ok(containment_lattice(
            Arc::clone(self),
            all_transfer_systems(self),
        )?)
    }

    /// Constructs the composition-closed order on transfer systems.
    pub fn composition_closed_order(self: &Arc<Self>) -> Result<TransferPoset<A>, TransferError> {
        Ok(composition_closed_order(
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

impl<A> TransferSystem<A> {
    /// Pairs raw transfer-system data with its ambient universe.
    pub fn new(raw: RawTransferSystem, universe: Arc<TransferUniverse<A>>) -> Self {
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
}

fn build_transfer_context<A>(lattice: &Lattice<A>, proper_edges: &[Edge]) -> TransferContext {
    let matrix = proper_edges
        .iter()
        .map(|edge1| {
            proper_edges
                .iter()
                .map(|edge2| {
                    lattice.leq(edge2.to, edge1.from)
                        || !lattice.leq(edge2.to, edge1.to)
                        || !lattice.leq(edge2.from, edge1.from)
                })
                .collect()
        })
        .collect();
    FormalContext::new(proper_edges.to_vec(), proper_edges.to_vec(), matrix)
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
}

fn factorization_condition(order: &[BitVec], left: &PartialOrder, right: &PartialOrder) -> bool {
    left.pairs.iter().all(|&first| {
        right.pairs.iter().all(|&second| {
            !order[first.from][second.from]
                || !order[first.to][second.to]
                || has_factorization_witness(order, left, right, first, second)
        })
    })
}

fn has_factorization_witness(
    order: &[BitVec],
    left: &PartialOrder,
    right: &PartialOrder,
    first: Edge,
    second: Edge,
) -> bool {
    for w_prime in right.matrix_transpose[second.to].iter_ones() {
        if !order[first.to][w_prime] {
            continue;
        }
        for z_prime in left.matrix_transpose[w_prime].iter_ones() {
            if order[first.from][z_prime] && order[z_prime][second.from] {
                return true;
            }
        }
    }
    false
}

fn containment_lattice<A>(
    universe: Arc<TransferUniverse<A>>,
    systems: Vec<RawTransferSystem>,
) -> Result<TransferLattice<A>, LatticeError> {
    let poset = transfer_systems_ordered_by(systems, |left, right| {
        bitvec_subset(left.arrows(), right.arrows())
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
    let order = universe
        .underlying_lattice()
        .as_poset()
        .relation_matrix()
        .to_vec();
    let partial_orders = systems
        .iter()
        .map(|raw| raw.as_partial_order(&universe))
        .collect::<Vec<_>>();

    let relation = (0..systems.len())
        .map(|left| {
            (0..systems.len())
                .map(|right| {
                    bitvec_subset(systems[left].arrows(), systems[right].arrows())
                        && factorization_condition(
                            &order,
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

///  Checks if the set of bits in `left` is a subset of the set of bits in `right` (and that they have the same number of bits). To do this we loop over the underlying words of the bit vectors and check that no bit is set in `left` that is not also set in `right`.
pub(crate) fn bitvec_subset(left: &BitVec, right: &BitVec) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let bits_per_word = usize::BITS as usize;
    let full_words = left.len() / bits_per_word;
    let remainder = left.len() % bits_per_word;
    let left_words = left.as_raw_slice();
    let right_words = right.as_raw_slice();

    for i in 0..full_words {
        if left_words[i] & !right_words[i] != 0 {
            return false;
        }
    }

    if remainder == 0 {
        return true;
    }

    let mask = (1usize << remainder) - 1;
    (left_words[full_words] & !right_words[full_words] & mask) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_index_and_formal_context_are_precomputed_together() {
        let lattice = Arc::new(Lattice::chain(2).expect("the finite chain is a lattice"));
        let universe = lattice.transfer_universe();

        assert_eq!(
            universe.proper_edges(),
            &[Edge::new(0, 1), Edge::new(0, 2), Edge::new(1, 2)]
        );
        assert_eq!(universe.context.objects, universe.proper_edges());
        assert_eq!(universe.context.attributes, universe.proper_edges());
        assert_eq!(
            universe.relation_index().proper_edge_id(Edge::new(0, 2)),
            Some(1)
        );
        assert_eq!(
            universe.relation_index().proper_edge_id(Edge::new(1, 1)),
            None
        );
        let systems = universe.transfer_systems();
        assert!(!systems.is_empty());

        let top = systems
            .iter()
            .max_by_key(|system| system.raw().arrows().count_ones())
            .expect("the chain has a top transfer system");
        assert!(top.contains_relation(Edge::new(0, 0)));
        assert!(top.contains_relation(Edge::new(0, 2)));
        assert!(!top.contains_relation(Edge::new(2, 0)));
        assert!(!top.contains_relation(Edge::new(3, 3)));
    }
}
