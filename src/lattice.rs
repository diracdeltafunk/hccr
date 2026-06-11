use crate::morphism::LatticeMap;
use crate::poset::{Edge, Either, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use std::convert::TryFrom;
use std::fmt;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeError {
    Poset(PosetError),
    Empty,
    MissingBottom,
    MissingTop,
    NotALattice { left: ElementId, right: ElementId },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::Poset(error) => write!(f, "{error}"),
            LatticeError::Empty => write!(f, "a finite lattice must be nonempty"),
            LatticeError::MissingBottom => write!(f, "poset has no bottom element"),
            LatticeError::MissingTop => write!(f, "poset has no top element"),
            LatticeError::NotALattice { left, right } => write!(
                f,
                "elements {left} and {right} do not have both meet and join"
            ),
        }
    }
}

impl std::error::Error for LatticeError {}

impl From<PosetError> for LatticeError {
    fn from(error: PosetError) -> Self {
        Self::Poset(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HorizontalJoinError {
    TrivialInput,
    Lattice(LatticeError),
}

impl fmt::Display for HorizontalJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HorizontalJoinError::TrivialInput => {
                write!(f, "cannot horizontally join a trivial lattice")
            }
            HorizontalJoinError::Lattice(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for HorizontalJoinError {}

impl From<LatticeError> for HorizontalJoinError {
    fn from(error: LatticeError) -> Self {
        Self::Lattice(error)
    }
}

impl From<PosetError> for HorizontalJoinError {
    fn from(error: PosetError) -> Self {
        Self::Lattice(LatticeError::from(error))
    }
}

impl<A> Lattice<A> {
    pub fn new(poset: Poset<A>) -> Result<Self, LatticeError> {
        Self::try_from(poset)
    }

    pub fn as_poset(&self) -> &Poset<A> {
        &self.poset
    }

    pub fn into_poset(self) -> Poset<A> {
        self.poset
    }

    pub fn relabel<B, F>(&self, f: F) -> Lattice<B>
    where
        F: FnMut(&A) -> B,
    {
        Lattice {
            poset: self.poset.relabel(f),
            meet: self.meet.clone(),
            join: self.join.clone(),
            bottom: self.bottom,
            top: self.top,
        }
    }

    pub fn size(&self) -> usize {
        self.poset.size()
    }

    pub fn is_trivial(&self) -> bool {
        self.bottom == self.top
    }

    pub fn is_fusion_of_total_orders(&self) -> bool {
        (0..self.size()).all(|i| {
            (i + 1..self.size()).all(|j| {
                let meet = self.meet_id(i, j);
                meet == i || meet == j || meet == self.bottom
            })
        })
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
    type Error = LatticeError;

    fn try_from(poset: Poset<A>) -> Result<Self, Self::Error> {
        if poset.is_empty() {
            return Err(LatticeError::Empty);
        }

        let mut meet = vec![vec![0; poset.size()]; poset.size()];
        let mut join = vec![vec![0; poset.size()]; poset.size()];

        for i in 0..poset.size() {
            for j in 0..poset.size() {
                let Some(m) = poset.meet(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                let Some(k) = poset.join(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                meet[i][j] = m;
                join[i][j] = k;
            }
        }

        let bottom = poset.bottom().ok_or(LatticeError::MissingBottom)?;
        let top = poset.top().ok_or(LatticeError::MissingTop)?;

        Ok(Self {
            poset,
            meet,
            join,
            bottom,
            top,
        })
    }
}

/// Fuses two nontrivial lattices by identifying their bottoms and tops.
pub fn horizontal_join<A: Clone, B: Clone>(
    left: Arc<Lattice<A>>,
    right: Arc<Lattice<B>>,
) -> Result<LatticeFusion<A, B>, HorizontalJoinError> {
    if left.bottom() == left.top() || right.bottom() == right.top() {
        return Err(HorizontalJoinError::TrivialInput);
    }

    let left_len = left.size();
    let right_len = right.size();
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

impl<A> From<&Lattice<A>> for Vec<Edge> {
    fn from(lattice: &Lattice<A>) -> Self {
        lattice.as_poset().proper_relations_iter().collect()
    }
}
