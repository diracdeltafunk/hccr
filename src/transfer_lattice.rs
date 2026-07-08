use crate::lattice::{Lattice, LatticeError};
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::fmt;
use std::sync::Arc;

type TransferContext = FormalContext<Edge, Edge>;

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

/// Shared ambient data that gives raw transfer-system bitsets their meaning.
#[derive(Debug)]
pub struct TransferUniverse<A> {
    /// The lattice on which we might have a transfer system.
    underlying_lattice: Arc<Lattice<A>>,
    /// The formal context whose concepts correspond to transfer systems on `underlying_lattice`.
    context: TransferContext,
}

/// An owned transfer system together with its ambient lattice data.
#[derive(Debug)]
pub struct TransferSystem<A> {
    raw: RawTransferSystem,
    universe: Arc<TransferUniverse<A>>,
}

/// A poset of transfer systems on a fixed lattice.
#[derive(Debug, Clone)]
pub struct TransferPoset<A> {
    universe: Arc<TransferUniverse<A>>,
    poset: Poset<RawTransferSystem>,
}

/// A lattice of transfer systems on a fixed lattice.
#[derive(Debug, Clone)]
pub struct TransferLattice<A> {
    universe: Arc<TransferUniverse<A>>,
    lattice: Lattice<RawTransferSystem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    Poset(PosetError),
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
    pub fn transfer_universe(self: Arc<Self>) -> Arc<TransferUniverse<A>> {
        Arc::new(TransferUniverse::new(self))
    }

    fn transfer_context(&self) -> TransferContext {
        let proper_edges: Vec<_> = self.as_poset().proper_relations_iter().collect();
        let matrix = proper_edges
            .iter()
            .map(|edge1| {
                proper_edges
                    .iter()
                    .map(|edge2| {
                        self.leq(edge2.to, edge1.from)
                            || !self.leq(edge2.to, edge1.to)
                            || !self.leq(edge2.from, edge1.from)
                    })
                    .collect()
            })
            .collect();
        FormalContext::new(proper_edges.clone(), proper_edges, matrix)
    }

    pub fn transfer_systems_containment(
        self: Arc<Self>,
    ) -> Result<TransferLattice<A>, TransferError> {
        self.transfer_universe().containment_lattice()
    }

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
    fn new(arrows: BitVec) -> Self {
        Self { arrows }
    }

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

impl<A> TransferUniverse<A> {
    pub fn new(underlying_lattice: Arc<Lattice<A>>) -> Self {
        let context = underlying_lattice.transfer_context();
        Self {
            underlying_lattice,
            context,
        }
    }

    pub fn underlying_lattice(&self) -> &Arc<Lattice<A>> {
        &self.underlying_lattice
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.underlying_lattice
    }

    fn context(&self) -> &TransferContext {
        &self.context
    }

    pub fn proper_edges(&self) -> &[Edge] {
        &self.context.objects
    }

    pub fn transfer_systems(self: &Arc<Self>) -> Vec<TransferSystem<A>> {
        all_transfer_systems(self)
            .into_iter()
            .map(|raw| TransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    pub fn containment_lattice(self: &Arc<Self>) -> Result<TransferLattice<A>, TransferError> {
        Ok(containment_lattice(
            Arc::clone(self),
            all_transfer_systems(self),
        )?)
    }

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
    pub fn new(raw: RawTransferSystem, universe: Arc<TransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    pub fn raw(&self) -> &RawTransferSystem {
        &self.raw
    }

    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

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

    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    pub fn raw_poset(&self) -> &Poset<RawTransferSystem> {
        &self.poset
    }

    pub fn size(&self) -> usize {
        self.poset.size()
    }

    pub fn cover_relations(&self) -> EdgeSet {
        self.poset.cover_relations()
    }

    pub fn system(&self, id: ElementId) -> Option<TransferSystem<A>> {
        self.poset
            .element(id)
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    pub fn systems(&self) -> impl Iterator<Item = TransferSystem<A>> + '_ {
        self.poset
            .elements()
            .iter()
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    pub fn to_system_poset(&self) -> Poset<TransferSystem<A>> {
        self.poset
            .relabelled(|raw| TransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

impl<A> TransferLattice<A> {
    fn new(universe: Arc<TransferUniverse<A>>, lattice: Lattice<RawTransferSystem>) -> Self {
        Self { universe, lattice }
    }

    pub fn universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.universe
    }

    pub fn raw_lattice(&self) -> &Lattice<RawTransferSystem> {
        &self.lattice
    }

    pub fn as_poset(&self) -> &Poset<RawTransferSystem> {
        self.lattice.as_poset()
    }

    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    pub fn system(&self, id: ElementId) -> Option<TransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    pub fn systems(&self) -> impl Iterator<Item = TransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
    }

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
fn bitvec_subset(left: &BitVec, right: &BitVec) -> bool {
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
