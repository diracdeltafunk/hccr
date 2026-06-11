use crate::lattice::{Lattice, LatticeError};
use crate::morphism::{LatticeMap, LatticeMapError};
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub type TransferSystemId = usize;

pub type TransferContext = FormalContext<Edge, Edge>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTransferSystem {
    /// Bitmask of the non-identity arrows in the transfer system.
    arrows: BitVec,
}

/// Shared ambient data that gives raw transfer-system bitsets their meaning.
#[derive(Debug)]
pub struct TransferSystemUniverse<A> {
    lattice: Arc<Lattice<A>>,
    proper_edges: Vec<Edge>,
    proper_edge_ids: HashMap<Edge, usize>,
    context: TransferContext,
}

/// An owned transfer system together with its ambient lattice data.
#[derive(Debug, Clone)]
pub struct TransferSystem<A> {
    raw: RawTransferSystem,
    universe: Arc<TransferSystemUniverse<A>>,
}

pub type TransferPoset<A> = Poset<TransferSystem<A>>;
pub type TransferLattice<A> = Lattice<TransferSystem<A>>;

/// A borrowed transfer-system view from a [`TransferSystems`] collection.
#[derive(Debug, Clone, Copy)]
pub struct TransferSystemRef<'a, A> {
    raw: &'a RawTransferSystem,
    universe: &'a Arc<TransferSystemUniverse<A>>,
}

/// Stores a collection of transfer systems on a fixed lattice.
#[derive(Debug, Clone)]
pub struct TransferSystems<A> {
    universe: Arc<TransferSystemUniverse<A>>,
    systems: Vec<RawTransferSystem>,
}

#[derive(Debug, Clone)]
struct SystemRelation {
    arrows: Vec<Edge>,
    columns: Vec<BitVec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    Poset(PosetError),
    Lattice(LatticeError),
    LatticeMap(LatticeMapError),
    PulledBackSystemNotFound,
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferError::Poset(error) => write!(f, "{error}"),
            TransferError::Lattice(error) => write!(f, "{error}"),
            TransferError::LatticeMap(error) => write!(f, "{error}"),
            TransferError::PulledBackSystemNotFound => {
                write!(f, "pulled-back transfer system was not found")
            }
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

impl From<LatticeMapError> for TransferError {
    fn from(error: LatticeMapError) -> Self {
        Self::LatticeMap(error)
    }
}

impl RawTransferSystem {
    pub fn new(arrows: BitVec) -> Self {
        Self { arrows }
    }

    pub fn empty(num_proper_edges: usize) -> Self {
        Self {
            arrows: BitVec::repeat(false, num_proper_edges),
        }
    }

    pub fn arrows(&self) -> &BitVec {
        &self.arrows
    }

    pub fn contains_proper_edge_id(&self, edge: usize) -> bool {
        self.arrows.get(edge).is_some_and(|bit| *bit)
    }
}

impl<A> TransferSystemUniverse<A> {
    pub fn new(
        lattice: Arc<Lattice<A>>,
        proper_edges: Vec<Edge>,
        context: TransferContext,
    ) -> Self {
        let proper_edge_ids = proper_edges
            .iter()
            .copied()
            .enumerate()
            .map(|(id, edge)| (edge, id))
            .collect();
        Self {
            lattice,
            proper_edges,
            proper_edge_ids,
            context,
        }
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.lattice
    }

    pub fn context(&self) -> &TransferContext {
        &self.context
    }

    pub fn proper_edges(&self) -> &[Edge] {
        &self.proper_edges
    }

    pub fn proper_edge_id(&self, edge: Edge) -> Option<usize> {
        self.proper_edge_ids.get(&edge).copied()
    }
}

impl<A> TransferSystem<A> {
    pub fn new(raw: RawTransferSystem, universe: Arc<TransferSystemUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    pub fn as_view(&self) -> TransferSystemRef<'_, A> {
        TransferSystemRef {
            raw: &self.raw,
            universe: &self.universe,
        }
    }

    pub fn raw(&self) -> &RawTransferSystem {
        &self.raw
    }

    pub fn into_raw(self) -> RawTransferSystem {
        self.raw
    }

    pub fn universe(&self) -> &Arc<TransferSystemUniverse<A>> {
        &self.universe
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    pub fn context(&self) -> &TransferContext {
        self.universe.context()
    }

    pub fn proper_edges(&self) -> &[Edge] {
        self.universe.proper_edges()
    }

    pub fn contains_proper_edge_id(&self, edge: usize) -> bool {
        self.raw.contains_proper_edge_id(edge)
    }

    pub fn contains_edge(&self, edge: Edge) -> bool {
        contains_edge(self.universe.as_ref(), &self.raw, edge)
    }

    pub fn selected_proper_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        selected_proper_edges(self.universe.as_ref(), &self.raw)
    }

    pub fn edge_set(&self, include_identities: bool) -> EdgeSet {
        edge_set(self.universe.as_ref(), &self.raw, include_identities)
    }
}

impl<A> PartialEq for TransferSystem<A> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && Arc::ptr_eq(&self.universe, &other.universe)
    }
}

impl<A> Eq for TransferSystem<A> {}

impl<'a, A> TransferSystemRef<'a, A> {
    pub fn raw(&self) -> &'a RawTransferSystem {
        self.raw
    }

    pub fn universe(&self) -> &'a TransferSystemUniverse<A> {
        self.universe.as_ref()
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.as_ref().lattice()
    }

    pub fn context(&self) -> &TransferContext {
        self.universe.as_ref().context()
    }

    pub fn proper_edges(&self) -> &[Edge] {
        self.universe.as_ref().proper_edges()
    }

    pub fn contains_proper_edge_id(&self, edge: usize) -> bool {
        self.raw.contains_proper_edge_id(edge)
    }

    pub fn contains_edge(&self, edge: Edge) -> bool {
        contains_edge(self.universe.as_ref(), self.raw, edge)
    }

    pub fn selected_proper_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        selected_proper_edges(self.universe.as_ref(), self.raw)
    }

    pub fn edge_set(&self, include_identities: bool) -> EdgeSet {
        edge_set(self.universe.as_ref(), self.raw, include_identities)
    }

    pub fn to_owned(&self) -> TransferSystem<A> {
        TransferSystem::new(self.raw.clone(), Arc::clone(self.universe))
    }
}

impl<A> TransferSystems<A> {
    pub fn on(lattice: Arc<Lattice<A>>) -> Result<Self, TransferError> {
        let proper_edges: Vec<_> = lattice.as_poset().proper_relations_iter().collect();
        let context = transfer_context(lattice.as_poset(), &proper_edges);
        let systems = context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| RawTransferSystem::new(concept.extent))
            .collect::<Vec<_>>();
        let universe = Arc::new(TransferSystemUniverse::new(lattice, proper_edges, context));

        Ok(Self { universe, systems })
    }

    pub fn universe(&self) -> &Arc<TransferSystemUniverse<A>> {
        &self.universe
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    pub fn context(&self) -> &TransferContext {
        self.universe.context()
    }

    pub fn proper_edges(&self) -> &[Edge] {
        self.universe.proper_edges()
    }

    pub fn len(&self) -> usize {
        self.systems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    pub fn raw_systems(&self) -> &[RawTransferSystem] {
        &self.systems
    }

    pub fn raw_system(&self, id: TransferSystemId) -> Option<&RawTransferSystem> {
        self.systems.get(id)
    }

    pub fn system(&self, id: TransferSystemId) -> Option<TransferSystemRef<'_, A>> {
        self.systems.get(id).map(|raw| TransferSystemRef {
            raw,
            universe: &self.universe,
        })
    }

    pub fn owned_system(&self, id: TransferSystemId) -> Option<TransferSystem<A>> {
        self.system(id).map(|system| system.to_owned())
    }

    pub fn iter(&self) -> impl Iterator<Item = TransferSystemRef<'_, A>> {
        let universe = &self.universe;
        self.systems
            .iter()
            .map(move |raw| TransferSystemRef { raw, universe })
    }

    pub fn owned_systems(&self) -> Vec<TransferSystem<A>> {
        self.systems
            .iter()
            .cloned()
            .map(|raw| TransferSystem::new(raw, Arc::clone(&self.universe)))
            .collect()
    }

    pub fn containment_lattice(&self) -> Result<TransferLattice<A>, LatticeError> {
        containment_lattice(self.owned_systems())
    }

    pub fn ordered_by<F>(&self, predicate: F) -> Result<TransferPoset<A>, PosetError>
    where
        F: Fn(&RawTransferSystem, &RawTransferSystem) -> bool,
    {
        poset_on_transfer_systems(self.owned_systems(), predicate)
    }

    pub fn pullback<B>(
        &self,
        f: &LatticeMap<A, B>,
        target: &TransferSystems<B>,
    ) -> Result<LatticeMap<TransferSystem<B>, TransferSystem<A>>, TransferError> {
        let mut map = Vec::with_capacity(target.systems.len());
        for target_system in &target.systems {
            let mut pulled_back = BitVec::repeat(false, self.proper_edges().len());
            for (edge_id, edge) in self.proper_edges().iter().copied().enumerate() {
                if target.contains_mapped_edge(target_system, f.apply(edge.from), f.apply(edge.to))
                {
                    pulled_back.set(edge_id, true);
                }
            }
            let Some(system_id) = self.find_system_by_arrows(&pulled_back) else {
                return Err(TransferError::PulledBackSystemNotFound);
            };
            map.push(system_id);
        }
        let domain = Arc::new(target.containment_lattice()?);
        let codomain = Arc::new(self.containment_lattice()?);

        Ok(LatticeMap::new(domain, codomain, map)?)
    }

    pub fn edge_set(&self, system: &RawTransferSystem, include_identities: bool) -> EdgeSet {
        edge_set(self.universe.as_ref(), system, include_identities)
    }

    pub fn composition_closed_order(&self) -> Result<TransferPoset<A>, PosetError> {
        let system_relations = self.system_relations();
        let order = self.lattice().as_poset().relation_matrix().to_vec();

        poset_on_transfer_systems_by_id(self.owned_systems(), |left, right| {
            bitvec_subset(self.systems[left].arrows(), self.systems[right].arrows())
                && factorization_condition(
                    &order,
                    &system_relations[left],
                    &system_relations[right],
                )
        })
    }

    fn contains_mapped_edge(
        &self,
        system: &RawTransferSystem,
        from: ElementId,
        to: ElementId,
    ) -> bool {
        from == to
            || self
                .universe
                .proper_edge_id(Edge::new(from, to))
                .is_some_and(|edge_id| system.contains_proper_edge_id(edge_id))
    }

    fn find_system_by_arrows(&self, arrows: &BitVec) -> Option<TransferSystemId> {
        self.systems
            .iter()
            .position(|system| system.arrows() == arrows)
    }

    fn system_relations(&self) -> Vec<SystemRelation> {
        self.systems
            .iter()
            .map(|system| system_relation(self.universe.as_ref(), system))
            .collect()
    }
}

fn contains_edge<A>(
    universe: &TransferSystemUniverse<A>,
    system: &RawTransferSystem,
    edge: Edge,
) -> bool {
    if edge.from == edge.to {
        edge.from < universe.lattice.size()
    } else {
        universe
            .proper_edge_id(edge)
            .is_some_and(|edge_id| system.contains_proper_edge_id(edge_id))
    }
}

fn selected_proper_edges<'a, A>(
    universe: &'a TransferSystemUniverse<A>,
    system: &'a RawTransferSystem,
) -> impl Iterator<Item = Edge> + 'a {
    system
        .arrows
        .iter_ones()
        .map(|edge_id| universe.proper_edges[edge_id])
}

fn edge_set<A>(
    universe: &TransferSystemUniverse<A>,
    system: &RawTransferSystem,
    include_identities: bool,
) -> EdgeSet {
    let mut result = EdgeSet::new();
    if include_identities {
        for id in 0..universe.lattice.size() {
            result.insert(Edge::new(id, id));
        }
    }
    result.extend(selected_proper_edges(universe, system));
    result
}

fn system_relation<A>(
    universe: &TransferSystemUniverse<A>,
    system: &RawTransferSystem,
) -> SystemRelation {
    let n = universe.lattice.size();
    let mut arrows = Vec::with_capacity(n + system.arrows().count_ones());
    let mut columns = vec![BitVec::repeat(false, n); n];

    for (id, column) in columns.iter_mut().enumerate() {
        arrows.push(Edge::new(id, id));
        column.set(id, true);
    }
    for edge in selected_proper_edges(universe, system) {
        arrows.push(edge);
        columns[edge.to].set(edge.from, true);
    }

    SystemRelation { arrows, columns }
}

fn factorization_condition(
    order: &[BitVec],
    left: &SystemRelation,
    right: &SystemRelation,
) -> bool {
    left.arrows.iter().all(|&first| {
        right.arrows.iter().all(|&second| {
            !order[first.from][second.from]
                || !order[first.to][second.to]
                || has_factorization_witness(order, left, right, first, second)
        })
    })
}

fn has_factorization_witness(
    order: &[BitVec],
    left: &SystemRelation,
    right: &SystemRelation,
    first: Edge,
    second: Edge,
) -> bool {
    for w_prime in right.columns[second.to].iter_ones() {
        if !order[first.to][w_prime] {
            continue;
        }
        for z_prime in left.columns[w_prime].iter_ones() {
            if order[first.from][z_prime] && order[z_prime][second.from] {
                return true;
            }
        }
    }
    false
}

pub fn transfer_context<A>(poset: &Poset<A>, proper_edges: &[Edge]) -> TransferContext {
    let matrix = proper_edges
        .iter()
        .map(|edge1| {
            proper_edges
                .iter()
                .map(|edge2| {
                    poset.leq(edge2.to, edge1.from)
                        || !poset.leq(edge2.to, edge1.to)
                        || !poset.leq(edge2.from, edge1.from)
                })
                .collect()
        })
        .collect();
    FormalContext::new(proper_edges.to_vec(), proper_edges.to_vec(), matrix)
}

fn containment_lattice<A>(
    systems: Vec<TransferSystem<A>>,
) -> Result<TransferLattice<A>, LatticeError> {
    let poset = poset_on_transfer_systems(systems, |left, right| {
        bitvec_subset(left.arrows(), right.arrows())
    })?;
    Lattice::new(poset)
}

fn poset_on_transfer_systems<A, F>(
    systems: Vec<TransferSystem<A>>,
    predicate: F,
) -> Result<TransferPoset<A>, PosetError>
where
    F: Fn(&RawTransferSystem, &RawTransferSystem) -> bool,
{
    let relation = systems
        .iter()
        .map(|left| {
            systems
                .iter()
                .map(|right| predicate(left.raw(), right.raw()))
                .collect()
        })
        .collect();
    Poset::from_relation(systems, relation)
}

fn poset_on_transfer_systems_by_id<A, F>(
    systems: Vec<TransferSystem<A>>,
    predicate: F,
) -> Result<TransferPoset<A>, PosetError>
where
    F: Fn(TransferSystemId, TransferSystemId) -> bool,
{
    let relation = (0..systems.len())
        .map(|left| {
            (0..systems.len())
                .map(|right| predicate(left, right))
                .collect()
        })
        .collect();
    Poset::from_relation(systems, relation)
}

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
