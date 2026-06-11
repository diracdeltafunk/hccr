use crate::morphism::LatticeMap;
use crate::poset::{Edge, Either, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use std::convert::TryFrom;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice<A> {
    poset: Poset<A>,
    meet: Vec<Vec<ElementId>>,
    join: Vec<Vec<ElementId>>,
    bottom: ElementId,
    top: ElementId,
}

#[derive(Debug, Clone)]
pub struct LatticeFusion<A, B> {
    pub lattice: Arc<Lattice<Either<A, B>>>,
    pub left: LatticeMap<A, Either<A, B>>,
    pub right: LatticeMap<B, Either<A, B>>,
}

impl<A> Lattice<A> {
    pub fn new(poset: Poset<A>) -> Result<Self, PosetError> {
        Self::try_from(poset)
    }

    pub fn as_poset(&self) -> &Poset<A> {
        &self.poset
    }

    pub fn into_poset(self) -> Poset<A> {
        self.poset
    }

    pub fn len(&self) -> usize {
        self.poset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poset.is_empty()
    }

    pub fn elements(&self) -> &[A] {
        self.poset.elements()
    }

    pub fn element(&self, id: ElementId) -> Option<&A> {
        self.poset.element(id)
    }

    pub fn leq(&self, left: ElementId, right: ElementId) -> bool {
        self.poset.leq(left, right)
    }

    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.meet[left][right]
    }

    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.join[left][right]
    }

    pub fn bottom(&self) -> ElementId {
        self.bottom
    }

    pub fn top(&self) -> ElementId {
        self.top
    }
}

impl<A> TryFrom<Poset<A>> for Lattice<A> {
    type Error = PosetError;

    fn try_from(poset: Poset<A>) -> Result<Self, Self::Error> {
        if poset.is_empty() {
            return Err(PosetError::EmptyLattice);
        }

        let mut meet = vec![vec![0; poset.len()]; poset.len()];
        let mut join = vec![vec![0; poset.len()]; poset.len()];

        for i in 0..poset.len() {
            for j in 0..poset.len() {
                let Some(m) = poset.meet(i, j) else {
                    return Err(PosetError::NotALattice { left: i, right: j });
                };
                let Some(k) = poset.join(i, j) else {
                    return Err(PosetError::NotALattice { left: i, right: j });
                };
                meet[i][j] = m;
                join[i][j] = k;
            }
        }

        let bottom = poset.bottom().ok_or(PosetError::MissingBottom)?;
        let top = poset.top().ok_or(PosetError::MissingTop)?;

        Ok(Self {
            poset,
            meet,
            join,
            bottom,
            top,
        })
    }
}

impl<A: Clone, B: Clone> Lattice<Either<A, B>> {
    /// Fuses two nontrivial lattices by identifying their bottoms and tops.
    pub fn horizontal_join(
        left: Arc<Lattice<A>>,
        right: Arc<Lattice<B>>,
    ) -> Result<LatticeFusion<A, B>, PosetError> {
        if left.bottom() == left.top() || right.bottom() == right.top() {
            return Err(PosetError::TrivialFusionInput);
        }

        let left_len = left.len();
        let right_len = right.len();
        let mut elements = left
            .elements()
            .iter()
            .cloned()
            .map(Either::Left)
            .collect::<Vec<_>>();
        let mut right_map = vec![0; right_len];
        right_map[right.bottom()] = left.bottom();
        right_map[right.top()] = left.top();

        for (id, label) in right.elements().iter().cloned().enumerate() {
            if id != right.bottom() && id != right.top() {
                right_map[id] = elements.len();
                elements.push(Either::Right(label));
            }
        }

        let mut relation = vec![BitVec::repeat(false, elements.len()); elements.len()];
        for edge in left.as_poset().all_relations_iter() {
            relation[edge.from].set(edge.to, true);
        }
        for edge in right.as_poset().all_relations_iter() {
            relation[right_map[edge.from]].set(right_map[edge.to], true);
        }
        for (i, row) in relation.iter_mut().enumerate() {
            row.set(i, true);
        }

        let poset = Poset::from_relation(elements, relation)?;
        let lattice = Arc::new(Lattice::new(poset)?);
        let left_map = LatticeMap::new(
            Arc::clone(&left),
            Arc::clone(&lattice),
            (0..left_len).collect(),
        )
        .expect("fusion left embedding should be a lattice map");
        let right_map = LatticeMap::new(Arc::clone(&right), Arc::clone(&lattice), right_map)
            .expect("fusion right embedding should be a lattice map");

        Ok(LatticeFusion {
            lattice,
            left: left_map,
            right: right_map,
        })
    }
}

impl<A> From<&Lattice<A>> for Vec<Edge> {
    fn from(lattice: &Lattice<A>) -> Self {
        lattice.as_poset().proper_relations_iter().collect()
    }
}
