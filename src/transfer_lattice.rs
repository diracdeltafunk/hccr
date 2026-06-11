use crate::lattice::{Lattice, LatticeError};
use crate::morphism::{LatticeMap, MapError};
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use fcars::FormalContext;
use std::fmt;
use std::sync::Arc;

pub type TransferSystemId = usize;

pub type TransferContext = FormalContext<Edge, Edge>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferSystem {
    /// Proper arrows in the transfer system. Identity arrows are implicit.
    pub arrows: BitVec,
}

#[derive(Debug, Clone)]
pub struct TransferSystems<A> {
    lattice: Arc<Lattice<A>>,
    proper_edges: Vec<Edge>,
    context: TransferContext,
    systems: Vec<TransferSystem>,
    containment_lattice: Arc<Lattice<TransferSystemId>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    Poset(PosetError),
    Lattice(LatticeError),
    Map(MapError),
    PulledBackSystemNotFound,
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferError::Poset(error) => write!(f, "{error}"),
            TransferError::Lattice(error) => write!(f, "{error}"),
            TransferError::Map(error) => write!(f, "{error}"),
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

impl From<MapError> for TransferError {
    fn from(error: MapError) -> Self {
        Self::Map(error)
    }
}

impl TransferSystem {
    pub fn empty(num_proper_edges: usize) -> Self {
        Self {
            arrows: BitVec::repeat(false, num_proper_edges),
        }
    }

    pub fn arrows(&self) -> &BitVec {
        &self.arrows
    }

    pub fn contains_proper_edge_id(&self, edge: usize) -> bool {
        self.arrows[edge]
    }
}

impl<A> TransferSystems<A> {
    pub fn on(lattice: Arc<Lattice<A>>) -> Result<Self, TransferError> {
        let proper_edges: Vec<_> = lattice.as_poset().proper_relations_iter().collect();
        let context = transfer_context(lattice.as_poset(), &proper_edges);
        let systems = context
            .all_concepts_raw()
            .into_iter()
            .map(|concept| TransferSystem {
                arrows: concept.extent,
            })
            .collect::<Vec<_>>();
        let containment_lattice = Arc::new(containment_lattice(&systems)?);

        Ok(Self {
            lattice,
            proper_edges,
            context,
            systems,
            containment_lattice,
        })
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

    pub fn systems(&self) -> &[TransferSystem] {
        &self.systems
    }

    pub fn system(&self, id: TransferSystemId) -> Option<&TransferSystem> {
        self.systems.get(id)
    }

    pub fn containment_lattice(&self) -> Arc<Lattice<TransferSystemId>> {
        Arc::clone(&self.containment_lattice)
    }

    pub fn ordered_by<F>(&self, predicate: F) -> Result<Poset<TransferSystemId>, PosetError>
    where
        F: Fn(&TransferSystem, &TransferSystem) -> bool,
    {
        poset_on_system_ids(self.systems.len(), |left, right| {
            predicate(&self.systems[left], &self.systems[right])
        })
    }

    pub fn pullback<B>(
        &self,
        f: &LatticeMap<A, B>,
        target: &TransferSystems<B>,
    ) -> Result<LatticeMap<TransferSystemId, TransferSystemId>, TransferError> {
        let mut map = Vec::with_capacity(target.systems.len());
        for target_system in &target.systems {
            let mut pulled_back = BitVec::repeat(false, self.proper_edges.len());
            for (edge_id, edge) in self.proper_edges.iter().copied().enumerate() {
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

        Ok(LatticeMap::new(
            target.containment_lattice(),
            self.containment_lattice(),
            map,
        )?)
    }

    pub fn edge_set(&self, system: &TransferSystem, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            for id in 0..self.lattice.size() {
                result.insert(Edge::new(id, id));
            }
        }
        for edge_id in system.arrows.iter_ones() {
            result.insert(self.proper_edges[edge_id]);
        }
        result
    }

    pub fn composition_closed_order(&self) -> Result<Poset<TransferSystemId>, PosetError> {
        self.ordered_by(|left, right| {
            if !bitvec_subset(left.arrows(), right.arrows()) {
                return false;
            }
            let r1 = self.edge_set(left, true);
            let r2 = self.edge_set(right, false);
            let l2 = self.lattice.as_poset().llc(&r2);
            let composite = self.lattice.as_poset().compose(&r1, &l2);
            self.lattice.as_poset().composition_closed(&composite)
        })
    }

    fn contains_mapped_edge(
        &self,
        system: &TransferSystem,
        from: ElementId,
        to: ElementId,
    ) -> bool {
        from == to
            || self
                .proper_edges
                .iter()
                .position(|edge| edge.from == from && edge.to == to)
                .is_some_and(|edge_id| system.contains_proper_edge_id(edge_id))
    }

    fn find_system_by_arrows(&self, arrows: &BitVec) -> Option<TransferSystemId> {
        self.systems
            .iter()
            .position(|system| system.arrows() == arrows)
    }
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

fn containment_lattice(
    systems: &[TransferSystem],
) -> Result<Lattice<TransferSystemId>, LatticeError> {
    let poset = poset_on_system_ids(systems.len(), |left, right| {
        bitvec_subset(systems[left].arrows(), systems[right].arrows())
    })?;
    Lattice::new(poset)
}

fn poset_on_system_ids<F>(len: usize, predicate: F) -> Result<Poset<TransferSystemId>, PosetError>
where
    F: Fn(usize, usize) -> bool,
{
    let relation = (0..len)
        .map(|left| (0..len).map(|right| predicate(left, right)).collect())
        .collect();
    Poset::from_relation((0..len).collect(), relation)
}

fn bitvec_subset(left: &BitVec, right: &BitVec) -> bool {
    left.len() == right.len() && left.iter_ones().all(|i| right[i])
}
