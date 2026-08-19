//! Finite lattices.
//!
//! A [`Lattice`] is a nonempty finite poset in which
//! every pair `x, y` has a **meet** `x /\ y` and a **join** `x \/ y`. The meet
//! is the greatest element below both inputs; the join is the least element
//! above both. In a finite lattice these operations also determine a unique
//! bottom element, below everything, and a unique top element, above
//! everything.
//!
//! This type stores the underlying [`Poset`] together with complete meet and
//! join tables. Construction does the quadratic validation and precomputation;
//! subsequent operations on element ids take constant time.

use crate::morphism::LatticeMap;
use crate::poset::{ElementId, Poset, PosetError};
use bitvec::prelude::*;
use either::Either;
use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

/// A finite lattice.
///
/// The underlying order is stored as a [`Poset`].  Meets, joins, bottom, and top
/// are computed once when the lattice is constructed and then accessed by
/// [`ElementId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lattice<A> {
    poset: Poset<A>,
    meet: Vec<Vec<ElementId>>,
    join: Vec<Vec<ElementId>>,
    bottom: ElementId,
    top: ElementId,
}

/// The horizontal join, or fusion, of two nontrivial lattices.
///
/// The construction identifies the two bottom elements and identifies the two
/// top elements, leaving all other elements in the two factors disjoint.
#[derive(Debug, Clone)]
pub struct LatticeFusion<A, B> {
    /// The fused lattice, with labels tagged according to their original side.
    pub lattice: Arc<Lattice<Either<A, B>>>,
    /// The canonical lattice embedding of the left factor.
    pub left: LatticeMap<A, Either<A, B>>,
    /// The canonical lattice embedding of the right factor.
    pub right: LatticeMap<B, Either<A, B>>,
}

/// The categorical product of two finite lattices and its projections.
///
/// The order, meet, and join are all computed componentwise.
#[derive(Debug, Clone)]
pub struct LatticeProduct<A, B> {
    /// The product lattice.
    pub lattice: Arc<Lattice<(A, B)>>,
    /// The first projection `(a, b) |-> a`.
    pub left_projection: LatticeMap<(A, B), A>,
    /// The second projection `(a, b) |-> b`.
    pub right_projection: LatticeMap<(A, B), B>,
}

/// Errors that can occur while constructing a finite lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeError {
    /// The underlying poset failed validation.
    Poset(PosetError),
    /// A finite lattice must have at least one element.
    Empty,
    /// The requested Boolean lattice cannot be encoded by `usize` bitmasks.
    BooleanRankTooLarge {
        /// The requested rank.
        rank: usize,
    },
    /// The poset has no element below every other element.
    MissingBottom,
    /// The poset has no element above every other element.
    MissingTop,
    /// A pair of elements lacks a meet or a join.
    NotALattice {
        /// The first element in the failing pair.
        left: ElementId,
        /// The second element in the failing pair.
        right: ElementId,
    },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::Poset(error) => write!(f, "{error}"),
            LatticeError::Empty => write!(f, "a finite lattice must be nonempty"),
            LatticeError::BooleanRankTooLarge { rank } => write!(
                f,
                "cannot encode Boolean lattice of rank {rank} as usize bitmasks"
            ),
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

/// Errors that can occur while forming a horizontal join of lattices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HorizontalJoinError {
    /// At least one input has bottom equal to top.
    TrivialInput,
    /// The fused order failed to form a lattice.
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
    /// Constructs a lattice from a finite poset.
    ///
    /// This checks every unordered pair for a meet and a join, stores the
    /// results in symmetric lookup tables, and locates bottom and top. The
    /// original element ids and labels are preserved.
    pub fn new(poset: Poset<A>) -> Result<Self, LatticeError> {
        Self::try_from(poset)
    }

    /// Returns the underlying poset.
    pub fn as_poset(&self) -> &Poset<A> {
        &self.poset
    }

    /// Consumes the lattice and returns its underlying poset.
    pub fn into_poset(self) -> Poset<A> {
        self.poset
    }

    /// Returns the same lattice with labels transformed by `f`.
    ///
    /// The order, meet table, join table, bottom, and top are unchanged.
    pub fn relabelled<B, F>(&self, f: F) -> Lattice<B>
    where
        F: FnMut(&A) -> B,
    {
        Lattice {
            poset: self.poset.relabelled(f),
            meet: self.meet.clone(),
            join: self.join.clone(),
            bottom: self.bottom,
            top: self.top,
        }
    }

    /// Returns the number of elements.
    pub fn size(&self) -> usize {
        self.poset.size()
    }

    /// Returns whether the lattice has exactly one element.
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.bottom == self.top
    }

    /// Returns whether the lattice is a fusion of total orders.
    ///
    /// Equivalently, any two incomparable non-bottom elements have meet equal
    /// to bottom.  This is a useful recognition criterion for examples built
    /// from chains by horizontal joins.
    #[must_use]
    pub fn is_fusion_of_total_orders(&self) -> bool {
        (0..self.size()).all(|i| {
            (i + 1..self.size()).all(|j| {
                let meet = self.meet_id(i, j);
                meet == i || meet == j || meet == self.bottom
            })
        })
    }

    /// Returns all element labels in `ElementId` order.
    pub fn elements(&self) -> &[A] {
        self.poset.elements()
    }

    /// Returns the label of an element by id.
    pub fn element(&self, id: ElementId) -> Option<&A> {
        self.poset.element(id)
    }

    /// Tests the order relation `left <= right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn leq(&self, left: ElementId, right: ElementId) -> bool {
        self.poset.leq(left, right)
    }

    /// Returns the meet `left /\ right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.meet[left][right]
    }

    /// Returns the join `left \/ right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.join[left][right]
    }

    /// Returns the bottom element.
    pub fn bottom(&self) -> ElementId {
        self.bottom
    }

    /// Returns the top element.
    pub fn top(&self) -> ElementId {
        self.top
    }
}

impl Lattice<usize> {
    /// Constructs the chain `[top] = {0, ..., top}` with its usual total order.
    ///
    /// The argument is the largest element, so the lattice has `top + 1`
    /// elements. Meet is minimum and join is maximum.
    pub fn chain(top: usize) -> Result<Self, LatticeError> {
        Lattice::new(Poset::chain(top)?)
    }

    /// Constructs the Boolean lattice of subsets of `{0, ..., rank - 1}`.
    ///
    /// Elements are `usize` bitmasks ordered by subset inclusion: bit `i` is
    /// set exactly when the subset contains `i`. Consequently meet is bitwise
    /// intersection and join is bitwise union. For example, rank `3` produces
    /// eight elements numbered `0` through `7`.
    pub fn boolean(rank: usize) -> Result<Self, LatticeError> {
        if rank >= usize::BITS as usize {
            return Err(LatticeError::BooleanRankTooLarge { rank });
        }

        let size = 1usize << rank;
        Lattice::new(Poset::from_vec_by((0..size).collect(), |left, right| {
            left & !right == 0
        })?)
    }
}

impl<A: Clone, B: Clone> Lattice<(A, B)> {
    /// Constructs the direct product of two lattices and its canonical projections.
    ///
    /// The product lattice has elements `(a, b)` and componentwise order:
    /// `(a, b) <= (a', b')` if and only if `a <= a'` and `b <= b'`.
    pub fn product(
        left: Arc<Lattice<A>>,
        right: Arc<Lattice<B>>,
    ) -> Result<LatticeProduct<A, B>, LatticeError> {
        let product = crate::poset::product(
            Arc::new(left.as_poset().clone()),
            Arc::new(right.as_poset().clone()),
        )
        .expect("product projections should be poset maps");
        let lattice = Arc::new(Lattice::new(product.poset.as_ref().clone())?);
        let left_projection = LatticeMap::new(
            Arc::clone(&lattice),
            left,
            product.left_projection.map().to_vec(),
        )
        .expect("left product projection should be a lattice map");
        let right_projection = LatticeMap::new(
            Arc::clone(&lattice),
            right,
            product.right_projection.map().to_vec(),
        )
        .expect("right product projection should be a lattice map");

        Ok(LatticeProduct {
            lattice,
            left_projection,
            right_projection,
        })
    }
}

impl<A> TryFrom<Poset<A>> for Lattice<A> {
    type Error = LatticeError;

    fn try_from(poset: Poset<A>) -> Result<Self, Self::Error> {
        if poset.is_empty() {
            return Err(LatticeError::Empty);
        }

        let n = poset.size();
        let mut meet = vec![vec![0usize; n]; n];
        let mut join = vec![vec![0usize; n]; n];

        // Only compute the upper triangle; meet and join are symmetric.
        for i in 0..n {
            meet[i][i] = i;
            join[i][i] = i;
            for j in (i + 1)..n {
                let Some(m) = poset.meet(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                meet[i][j] = m;
                meet[j][i] = m;

                let Some(k) = poset.join(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                join[i][j] = k;
                join[j][i] = k;
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
///
/// The resulting lattice contains a copy of each input lattice, except that the
/// two bottom elements are identified and the two top elements are identified.
/// No new comparabilities are added between the two factors beyond those forced
/// by the common bottom and common top. The result is sometimes called the
/// horizontal sum of the lattices. The returned embeddings record which
/// element ids of the fused lattice represent the two inputs.
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
