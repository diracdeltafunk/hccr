//! Finite lattices.
//!
//! A [`Lattice`] is a nonempty finite poset in which
//! every pair `x, y` has a **meet** `x /\ y` and a **join** `x \/ y`. The meet
//! is the greatest element below both inputs; the join is the least element
//! above both. In a finite lattice these operations also determine a unique
//! bottom element, below everything, and a unique top element, above
//! everything.
//!
//! A `Lattice` is a cheap-to-clone handle around a [`Poset`] whose meet and
//! join tables have been computed. Every poset method is available on a
//! lattice directly, because `Lattice` dereferences to `Poset`.

use crate::label::Label;
use crate::morphism::LatticeMap;
use crate::poset::{ElementId, Poset, PosetError, product_coordinates};
use bitvec::prelude::*;
use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;

/// A finite lattice.
///
/// This is a cheap-to-clone handle; clones share their data. Meets, joins,
/// bottom, and top are computed once, when the lattice is constructed.
#[derive(Clone, PartialEq, Eq)]
pub struct Lattice {
    poset: Poset,
}

/// Precomputed lattice operations, cached on the underlying poset.
#[derive(Debug, Clone)]
pub(crate) struct LatticeTables {
    meet: Vec<Vec<ElementId>>,
    join: Vec<Vec<ElementId>>,
    bottom: ElementId,
    top: ElementId,
}

/// Errors that can occur while constructing a finite lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeError {
    /// The underlying poset failed validation.
    Poset(PosetError),
    /// A finite lattice must have at least one element.
    Empty,
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
    /// A factor has bottom equal to top.
    TrivialFactor {
        /// The position of the offending factor.
        factor: usize,
    },
}

impl fmt::Display for HorizontalJoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HorizontalJoinError::TrivialFactor { factor } => write!(
                f,
                "cannot horizontally join a trivial lattice (factor {factor})"
            ),
        }
    }
}

impl std::error::Error for HorizontalJoinError {}

impl Lattice {
    /// Constructs a lattice from a finite poset.
    ///
    /// This checks every pair of elements for a meet and a join, stores the
    /// results in lookup tables, and locates bottom and top. The result shares
    /// the poset's data, so its element ids and labels are those of the poset.
    pub fn new(poset: Poset) -> Result<Self, LatticeError> {
        poset
            .data()
            .lattice
            .get_or_init(|| LatticeTables::compute(&poset))
            .as_ref()
            .map_err(Clone::clone)?;
        Ok(Self { poset })
    }

    /// Constructs a lattice from labels and generating relations named by
    /// label. See [`Poset::from_covers`].
    pub fn from_covers<I, C, L, M>(labels: I, covers: C) -> Result<Self, LatticeError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
        C: IntoIterator<Item = (L, M)>,
        L: Into<Label>,
        M: Into<Label>,
    {
        Self::new(Poset::from_covers(labels, covers)?)
    }

    /// Constructs a lattice from your own values and an order predicate on
    /// them. See [`Poset::from_elements_by`].
    pub fn from_elements_by<I, F>(elements: I, leq: F) -> Result<Self, LatticeError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
        F: Fn(&I::Item, &I::Item) -> bool,
    {
        Self::new(Poset::from_elements_by(elements, leq)?)
    }

    fn from_poset_with_tables(poset: Poset, tables: LatticeTables) -> Self {
        let stored = poset.data().lattice.set(Ok(tables));
        debug_assert!(stored.is_ok(), "fresh posets have no lattice tables yet");
        Self { poset }
    }

    fn tables(&self) -> &LatticeTables {
        self.poset
            .data()
            .lattice
            .get()
            .and_then(|tables| tables.as_ref().ok())
            .expect("a Lattice always has computed tables")
    }

    /// Returns the underlying poset.
    pub fn as_poset(&self) -> &Poset {
        &self.poset
    }

    /// Consumes the lattice and returns its underlying poset.
    pub fn into_poset(self) -> Poset {
        self.poset
    }

    /// Returns the meet `left /\ right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn meet(&self, left: ElementId, right: ElementId) -> ElementId {
        self.tables().meet[left][right]
    }

    /// Returns the join `left \/ right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn join(&self, left: ElementId, right: ElementId) -> ElementId {
        self.tables().join[left][right]
    }

    /// Returns the bottom element.
    pub fn bottom(&self) -> ElementId {
        self.tables().bottom
    }

    /// Returns the top element.
    pub fn top(&self) -> ElementId {
        self.tables().top
    }

    /// Returns whether the lattice has exactly one element.
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.bottom() == self.top()
    }

    /// Returns whether the lattice is a horizontal join of chains.
    ///
    /// Equivalently, any two incomparable non-bottom elements have meet equal
    /// to bottom.
    #[must_use]
    pub fn is_fusion_of_total_orders(&self) -> bool {
        (0..self.size()).all(|i| {
            (i + 1..self.size()).all(|j| {
                let meet = self.meet(i, j);
                meet == i || meet == j || meet == self.bottom()
            })
        })
    }

    /// Returns the opposite lattice, with the same labels and element ids,
    /// every relation reversed, and meet and join exchanged.
    pub fn opposite(&self) -> Self {
        let tables = self.tables();
        Self::from_poset_with_tables(
            self.poset.opposite(),
            LatticeTables {
                meet: tables.join.clone(),
                join: tables.meet.clone(),
                bottom: tables.top,
                top: tables.bottom,
            },
        )
    }

    /// Returns the same lattice with new labels. See [`Poset::relabelled`].
    pub fn relabelled<F, L>(&self, f: F) -> Result<Self, PosetError>
    where
        F: FnMut(ElementId, Label) -> L,
        L: Into<Label>,
    {
        Ok(Self::from_poset_with_tables(
            self.poset.relabelled(f)?,
            self.tables().clone(),
        ))
    }

    /// Constructs the chain `[top] = {0 < 1 < ... < top}`.
    ///
    /// The argument is the largest element, so the lattice has `top + 1`
    /// elements, labelled by the integers `0` through `top`. Meet is minimum
    /// and join is maximum.
    pub fn chain(top: usize) -> Self {
        let size = top + 1;
        Self::from_poset_with_tables(
            Poset::chain(top),
            LatticeTables {
                meet: (0..size)
                    .map(|i| (0..size).map(|j| i.min(j)).collect())
                    .collect(),
                join: (0..size)
                    .map(|i| (0..size).map(|j| i.max(j)).collect())
                    .collect(),
                bottom: 0,
                top,
            },
        )
    }

    /// Constructs the Boolean lattice of subsets of `{0, ..., rank - 1}`.
    ///
    /// Elements are labelled by the subsets themselves, such as `{0, 2}`, and
    /// ordered by inclusion. The element with id `i` is the subset whose
    /// members are the positions of the set bits of `i`.
    ///
    /// Panics if `rank` is at least the number of bits in a `usize`.
    pub fn boolean(rank: usize) -> Self {
        assert!(
            rank < usize::BITS as usize,
            "cannot construct the Boolean lattice of rank {rank}"
        );
        let size = 1usize << rank;
        let labels = (0..size)
            .map(|mask| Label::set((0..rank).filter(|bit| mask & (1 << bit) != 0)))
            .collect();
        let relation = (0..size)
            .map(|left| (0..size).map(|right| left & !right == 0).collect())
            .collect();
        Self::from_poset_with_tables(
            Poset::from_validated(labels, relation),
            LatticeTables {
                meet: (0..size)
                    .map(|i| (0..size).map(|j| i & j).collect())
                    .collect(),
                join: (0..size)
                    .map(|i| (0..size).map(|j| i | j).collect())
                    .collect(),
                bottom: 0,
                top: size - 1,
            },
        )
    }

    /// Constructs the direct product of finitely many lattices.
    ///
    /// Elements are tuples `(x_0, ..., x_{k-1})`; the order, meet, and join
    /// are computed componentwise.
    pub fn product<I>(factors: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Lattice>,
    {
        Self::product_with_projections(factors).0
    }

    /// Constructs the direct product of finitely many lattices together with
    /// its projections, one for each factor.
    pub fn product_with_projections<I>(factors: I) -> (Self, Vec<LatticeMap>)
    where
        I: IntoIterator,
        I::Item: Borrow<Lattice>,
    {
        let factors = factors
            .into_iter()
            .map(|factor| factor.borrow().clone())
            .collect::<Vec<_>>();
        let (poset, _) = Poset::product_with_projections(factors.iter().map(Lattice::as_poset));
        let (coordinates, _) = product_coordinates(
            &factors
                .iter()
                .map(|factor| factor.relation_matrix())
                .collect::<Vec<_>>(),
        );
        let id_of = |coordinate: &[ElementId]| {
            coordinate
                .iter()
                .zip(&factors)
                .fold(0, |id, (&x, factor)| id * factor.size() + x)
        };
        let componentwise = |op: fn(&Lattice, ElementId, ElementId) -> ElementId| {
            coordinates
                .iter()
                .map(|left| {
                    coordinates
                        .iter()
                        .map(|right| {
                            let combined = left
                                .iter()
                                .zip(right)
                                .zip(&factors)
                                .map(|((&x, &y), factor)| op(factor, x, y))
                                .collect::<Vec<_>>();
                            id_of(&combined)
                        })
                        .collect()
                })
                .collect()
        };
        let tables = LatticeTables {
            meet: componentwise(Lattice::meet),
            join: componentwise(Lattice::join),
            bottom: id_of(&factors.iter().map(Lattice::bottom).collect::<Vec<_>>()),
            top: id_of(&factors.iter().map(Lattice::top).collect::<Vec<_>>()),
        };
        let product = Self::from_poset_with_tables(poset, tables);
        let projections = factors
            .iter()
            .enumerate()
            .map(|(index, factor)| {
                LatticeMap::from_validated(
                    product.clone(),
                    factor.clone(),
                    coordinates
                        .iter()
                        .map(|coordinate| coordinate[index])
                        .collect(),
                )
            })
            .collect();
        (product, projections)
    }

    /// Constructs the horizontal join (fusion) of finitely many nontrivial
    /// lattices.
    ///
    /// The factors are placed side by side, with all of their bottoms
    /// identified and all of their tops identified; no other relations are
    /// added between different factors. The shared bottom and top are
    /// labelled `"bot"` and `"top"`, and every other element `x` of the `i`th
    /// factor is labelled `(i, x)`.
    ///
    /// ```
    /// use hccr::lattice::Lattice;
    ///
    /// let c3 = Lattice::chain(3);
    /// let l = Lattice::horizontal_join([&c3, &c3, &c3])?;
    /// assert_eq!(l.size(), 8);
    /// assert!(l.leq(l.id((1, 2))?, l.id("top")?));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Joining no lattices gives the two-element chain `bot < top`, the
    /// unit for this operation.
    pub fn horizontal_join<I>(factors: I) -> Result<Self, HorizontalJoinError>
    where
        I: IntoIterator,
        I::Item: Borrow<Lattice>,
    {
        Ok(Self::horizontal_join_with_inclusions(factors)?.0)
    }

    /// Constructs the horizontal join of finitely many nontrivial lattices
    /// together with its inclusions, one for each factor.
    pub fn horizontal_join_with_inclusions<I>(
        factors: I,
    ) -> Result<(Self, Vec<LatticeMap>), HorizontalJoinError>
    where
        I: IntoIterator,
        I::Item: Borrow<Lattice>,
    {
        let factors = factors
            .into_iter()
            .map(|factor| factor.borrow().clone())
            .collect::<Vec<_>>();
        if let Some(factor) = factors.iter().position(Lattice::is_trivial) {
            return Err(HorizontalJoinError::TrivialFactor { factor });
        }

        let interior_count = factors
            .iter()
            .map(|factor| factor.size() - 2)
            .sum::<usize>();
        let size = interior_count + 2;
        let bottom = 0;
        let top = size - 1;
        let mut labels = Vec::with_capacity(size);
        labels.push(Label::from("bot"));
        let mut maps = Vec::with_capacity(factors.len());
        for (index, factor) in factors.iter().enumerate() {
            let mut map = vec![0; factor.size()];
            for id in factor.ids() {
                map[id] = if id == factor.bottom() {
                    bottom
                } else if id == factor.top() {
                    top
                } else {
                    labels.push(Label::from((index, factor.label(id))));
                    labels.len() - 1
                };
            }
            maps.push(map);
        }
        labels.push(Label::from("top"));

        let mut relation = vec![BitVec::repeat(false, size); size];
        relation[bottom].fill(true);
        for row in &mut relation {
            row.set(top, true);
        }
        for (factor, map) in factors.iter().zip(&maps) {
            for edge in factor.all_relations_iter() {
                relation[map[edge.from]].set(map[edge.to], true);
            }
        }
        let fusion = Lattice::new(Poset::from_validated(labels, relation))
            .expect("a horizontal join of nontrivial lattices is a lattice");
        let inclusions = factors
            .iter()
            .zip(maps)
            .map(|(factor, map)| LatticeMap::from_validated(factor.clone(), fusion.clone(), map))
            .collect();
        Ok((fusion, inclusions))
    }
}

impl LatticeTables {
    fn compute(poset: &Poset) -> Result<Self, LatticeError> {
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
                let Some(m) = poset.try_meet(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                meet[i][j] = m;
                meet[j][i] = m;

                let Some(k) = poset.try_join(i, j) else {
                    return Err(LatticeError::NotALattice { left: i, right: j });
                };
                join[i][j] = k;
                join[j][i] = k;
            }
        }

        let bottom = poset.try_bottom().ok_or(LatticeError::MissingBottom)?;
        let top = poset.try_top().ok_or(LatticeError::MissingTop)?;

        Ok(Self {
            meet,
            join,
            bottom,
            top,
        })
    }
}

impl Deref for Lattice {
    type Target = Poset;

    fn deref(&self) -> &Poset {
        &self.poset
    }
}

impl AsRef<Poset> for Lattice {
    fn as_ref(&self) -> &Poset {
        &self.poset
    }
}

impl From<Lattice> for Poset {
    fn from(lattice: Lattice) -> Self {
        lattice.poset
    }
}

impl TryFrom<Poset> for Lattice {
    type Error = LatticeError;

    fn try_from(poset: Poset) -> Result<Self, Self::Error> {
        Self::new(poset)
    }
}

impl fmt::Display for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.poset.fmt_named("Lattice", f)
    }
}

impl fmt::Debug for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The chain `[top] = {0 < 1 < ... < top}`; see [`Lattice::chain`].
pub fn chain(top: usize) -> Lattice {
    Lattice::chain(top)
}

/// The Boolean lattice of subsets of `{0, ..., rank - 1}`; see
/// [`Lattice::boolean`].
pub fn boolean(rank: usize) -> Lattice {
    Lattice::boolean(rank)
}

/// The direct product of finitely many lattices; see [`Lattice::product`].
pub fn product<I>(factors: I) -> Lattice
where
    I: IntoIterator,
    I::Item: Borrow<Lattice>,
{
    Lattice::product(factors)
}

/// The horizontal join of finitely many nontrivial lattices; see
/// [`Lattice::horizontal_join`].
pub fn horizontal_join<I>(factors: I) -> Result<Lattice, HorizontalJoinError>
where
    I: IntoIterator,
    I::Item: Borrow<Lattice>,
{
    Lattice::horizontal_join(factors)
}
