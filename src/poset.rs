//! Finite partially ordered sets.
//!
//! A partial order is a relation `<=` that is reflexive (`x <= x`),
//! antisymmetric (`x <= y` and `y <= x` imply `x = y`), and transitive. A
//! [`Poset`] stores the whole relation and its transpose
//! as dense Boolean matrices: `relation[i][j]` means `i <= j`. Storing all
//! comparable pairs, rather than only the edges of a Hasse diagram, makes tests
//! such as comparability and the computation of upper and lower sets
//! inexpensive for small finite posets.
//!
//! Every element has a [`Label`], unique within its poset, and a stable
//! position called its [`ElementId`]. Algorithms work with ids; labels are for
//! naming elements when building posets, looking elements up with
//! [`Poset::id`], and printing. An [`Edge`] is an ordered pair of ids,
//! oriented from the smaller element to the larger one.
//!
//! A `Poset` is a cheap-to-clone handle: cloning it shares the underlying data
//! rather than copying it.

use crate::bitvec_utils::{difference_assign, intersection, is_subset, transpose, union_assign};
use crate::cotransfer_lattice::CotransferUniverse;
use crate::label::Label;
use crate::lattice::{LatticeError, LatticeTables};
use crate::morphism::PosetMap;
use crate::transfer_lattice::TransferUniverse;
use bitvec::prelude::*;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

/// The stable index of an element in a finite poset.
///
/// Ids run from `0` through `poset.size() - 1` and index the slice returned by
/// [`Poset::labels`]. They describe a particular stored presentation of a
/// poset, not an isomorphism-invariant mathematical name.
pub type ElementId = usize;

/// An ordered relation `from <= to` between elements of a poset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Edge {
    /// The lower element of the relation.
    pub from: ElementId,
    /// The upper element of the relation.
    pub to: ElementId,
}

impl Edge {
    /// Constructs the relation `from <= to`.
    pub fn new(from: ElementId, to: ElementId) -> Self {
        Self { from, to }
    }

    /// Returns whether this relation is an identity relation `x <= x`.
    pub fn is_identity(self) -> bool {
        self.from == self.to
    }

    /// Returns the reversed relation `to <= from`.
    pub fn reversed(self) -> Self {
        Self::new(self.to, self.from)
    }
}

impl From<(ElementId, ElementId)> for Edge {
    fn from((from, to): (ElementId, ElementId)) -> Self {
        Self { from, to }
    }
}

impl From<Edge> for (ElementId, ElementId) {
    fn from(edge: Edge) -> Self {
        (edge.from, edge.to)
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} <= {}", self.from, self.to)
    }
}

/// A set of ordered relations in a finite poset.
pub type EdgeSet = HashSet<Edge>;

/// Errors that can occur while validating or constructing a finite poset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosetError {
    /// The relation matrix has the wrong number of rows.
    RelationHeight {
        /// The number of rows required by the element list.
        expected: usize,
        /// The number of rows actually supplied.
        actual: usize,
    },
    /// A row of the relation matrix has the wrong number of columns.
    RelationWidth {
        /// The row whose width is invalid.
        row: usize,
        /// The number of columns required by the element list.
        expected: usize,
        /// The number of columns actually supplied.
        actual: usize,
    },
    /// The relation is not reflexive.
    MissingReflexiveEdge {
        /// The element `x` for which `x <= x` is absent.
        element: ElementId,
    },
    /// The relation has a nontrivial two-cycle.
    NotAntisymmetric {
        /// One element in a pair with both `left <= right` and `right <= left`.
        left: ElementId,
        /// The other element in the antisymmetry violation.
        right: ElementId,
    },
    /// The relation is not transitive.
    NotTransitive {
        /// The lower element in `lower <= middle <= upper`.
        lower: ElementId,
        /// The middle element in `lower <= middle <= upper`.
        middle: ElementId,
        /// The upper element that should be above `lower`.
        upper: ElementId,
    },
    /// An edge references an element outside the poset.
    EdgeOutOfBounds {
        /// The invalid edge.
        edge: Edge,
        /// The number of elements in the poset.
        len: usize,
    },
    /// Two elements were given the same label.
    DuplicateLabel {
        /// The repeated label.
        label: Label,
    },
    /// No element has the requested label.
    UnknownLabel {
        /// The label that was looked up.
        label: Label,
    },
}

impl fmt::Display for PosetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PosetError::RelationHeight { expected, actual } => {
                write!(f, "relation has {actual} rows, expected {expected}")
            }
            PosetError::RelationWidth {
                row,
                expected,
                actual,
            } => write!(
                f,
                "relation row {row} has width {actual}, expected {expected}"
            ),
            PosetError::MissingReflexiveEdge { element } => {
                write!(
                    f,
                    "relation is missing reflexive edge {element} <= {element}"
                )
            }
            PosetError::NotAntisymmetric { left, right } => {
                write!(
                    f,
                    "relation contains both {left} <= {right} and {right} <= {left}"
                )
            }
            PosetError::NotTransitive {
                lower,
                middle,
                upper,
            } => write!(
                f,
                "relation contains {lower} <= {middle} and {middle} <= {upper}, but not {lower} <= {upper}"
            ),
            PosetError::EdgeOutOfBounds { edge, len } => write!(
                f,
                "edge {} <= {} is out of bounds for a poset with {len} elements",
                edge.from, edge.to
            ),
            PosetError::DuplicateLabel { label } => {
                write!(f, "two elements have the same label `{label}`")
            }
            PosetError::UnknownLabel { label } => write!(f, "no element has the label `{label}`"),
        }
    }
}

impl std::error::Error for PosetError {}

/// A finite poset.
///
/// This is a cheap-to-clone handle; clones share their data. Two posets
/// compare equal when they have the same labels in the same order and the same
/// order relation.
#[derive(Clone)]
pub struct Poset {
    data: Arc<PosetData>,
}

/// The shared data behind a [`Poset`] handle, together with lazily computed
/// structure that is cached for the lifetime of the poset.
pub(crate) struct PosetData {
    labels: Vec<Label>,
    index: HashMap<Label, ElementId>,
    relation: Arc<Vec<BitVec>>,
    relation_transpose: Arc<Vec<BitVec>>,
    pub(crate) lattice: OnceLock<Result<LatticeTables, LatticeError>>,
    pub(crate) transfer_universe: OnceLock<TransferUniverse>,
    pub(crate) cotransfer_universe: OnceLock<CotransferUniverse>,
}

impl Poset {
    /// Constructs a poset from labels and an already-computed order relation.
    ///
    /// The matrix must be square of size equal to the number of labels,
    /// reflexive, antisymmetric, and transitive. Entry `[i][j]` is
    /// interpreted as `i <= j`. Labels must be distinct.
    pub fn from_relation<I>(labels: I, relation: Vec<BitVec>) -> Result<Self, PosetError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
    {
        let labels = labels.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_relation(labels.len(), &relation)?;
        Self::from_parts(labels, Arc::new(relation), None)
    }

    /// Constructs the poset whose order relation is given by `leq`.
    ///
    /// The predicate is evaluated on your own values, which are converted to
    /// labels afterwards. For example, the divisors of 12 under divisibility:
    ///
    /// ```
    /// use hccr::poset::Poset;
    ///
    /// let divisors = Poset::from_elements_by([1, 2, 3, 4, 6, 12], |a, b| b % a == 0)?;
    /// assert!(divisors.leq(divisors.id(2)?, divisors.id(12)?));
    /// # Ok::<(), hccr::poset::PosetError>(())
    /// ```
    ///
    /// An arbitrary predicate is allowed, but one that is not a partial order
    /// produces a [`PosetError`].
    pub fn from_elements_by<I, F>(elements: I, leq: F) -> Result<Self, PosetError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
        F: Fn(&I::Item, &I::Item) -> bool,
    {
        let elements = elements.into_iter().collect::<Vec<_>>();
        let relation = elements
            .iter()
            .map(|a| elements.iter().map(|b| leq(a, b)).collect())
            .collect();
        Self::from_relation(elements, relation)
    }

    /// Constructs the poset defined by Rust's `PartialOrd` relation.
    pub fn from_partial_ord<I>(elements: I) -> Result<Self, PosetError>
    where
        I: IntoIterator,
        I::Item: Into<Label> + PartialOrd,
    {
        Self::from_elements_by(elements, |a, b| a <= b)
    }

    /// Produces the smallest partial order containing the given relations,
    /// which are named by element id.
    ///
    /// The supplied edges may be a Hasse diagram, a redundant collection of
    /// comparable pairs, or any other generating relation. Identity edges are
    /// added and transitive consequences are computed. Generators that create
    /// a directed cycle return [`PosetError::NotAntisymmetric`].
    pub fn from_edges<I, E>(labels: I, edges: E) -> Result<Self, PosetError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
        E: IntoIterator,
        E::Item: Into<Edge>,
    {
        let labels = labels.into_iter().map(Into::into).collect::<Vec<_>>();
        let n = labels.len();
        let mut relation = vec![BitVec::repeat(false, n); n];
        for (i, row) in relation.iter_mut().enumerate() {
            row.set(i, true);
        }
        for edge in edges {
            let edge = edge.into();
            if edge.from >= n || edge.to >= n {
                return Err(PosetError::EdgeOutOfBounds { edge, len: n });
            }
            relation[edge.from].set(edge.to, true);
        }
        transitive_closure(&mut relation);
        Self::from_relation(labels, relation)
    }

    /// Produces the smallest partial order containing the given relations,
    /// which are named by label.
    ///
    /// Typically the relations are the cover relations of a Hasse diagram,
    /// but any generating set of relations is allowed.
    ///
    /// ```
    /// use hccr::poset::Poset;
    ///
    /// let pentagon = Poset::from_covers(
    ///     ["0", "a", "b", "c", "1"],
    ///     [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    /// )?;
    /// assert!(pentagon.leq(pentagon.id("a")?, pentagon.id("1")?));
    /// # Ok::<(), hccr::poset::PosetError>(())
    /// ```
    pub fn from_covers<I, C, L, M>(labels: I, covers: C) -> Result<Self, PosetError>
    where
        I: IntoIterator,
        I::Item: Into<Label>,
        C: IntoIterator<Item = (L, M)>,
        L: Into<Label>,
        M: Into<Label>,
    {
        let labels = labels.into_iter().map(Into::into).collect::<Vec<_>>();
        let index = label_index(&labels)?;
        let lookup = |label: Label| {
            index
                .get(&label)
                .copied()
                .ok_or(PosetError::UnknownLabel { label })
        };
        let edges = covers
            .into_iter()
            .map(|(lower, upper)| Ok(Edge::new(lookup(lower.into())?, lookup(upper.into())?)))
            .collect::<Result<Vec<_>, PosetError>>()?;
        Self::from_edges(labels, edges)
    }

    /// Constructs the chain `[top] = {0 < 1 < ... < top}`.
    ///
    /// The argument is the largest element, so the poset has `top + 1`
    /// elements, labelled by the integers `0` through `top`.
    pub fn chain(top: usize) -> Self {
        let relation = (0..=top)
            .map(|i| (0..=top).map(|j| i <= j).collect())
            .collect();
        Self::from_validated((0..=top).map(Label::from).collect(), relation)
    }

    /// Constructs the antichain on `size` elements, labelled `0` through
    /// `size - 1`, with no relations other than identities.
    pub fn antichain(size: usize) -> Self {
        let relation = (0..size)
            .map(|i| (0..size).map(|j| i == j).collect())
            .collect();
        Self::from_validated((0..size).map(Label::from).collect(), relation)
    }

    /// Constructs the direct product of finitely many posets.
    ///
    /// Elements are tuples `(x_0, ..., x_{k-1})` with componentwise order,
    /// listed in lexicographic order of their ids. The product of no posets
    /// is the one-element poset labelled `()`.
    pub fn product<I>(factors: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Poset>,
    {
        Self::product_with_projections(factors).0
    }

    /// Constructs the direct product of finitely many posets together with
    /// its projections, one for each factor.
    pub fn product_with_projections<I>(factors: I) -> (Self, Vec<PosetMap>)
    where
        I: IntoIterator,
        I::Item: Borrow<Poset>,
    {
        let factors = factors
            .into_iter()
            .map(|factor| factor.borrow().clone())
            .collect::<Vec<_>>();
        let (coordinates, relation) = product_coordinates(
            &factors
                .iter()
                .map(|factor| factor.relation_matrix())
                .collect::<Vec<_>>(),
        );
        let labels = coordinates
            .iter()
            .map(|coordinate| {
                Label::tuple(
                    coordinate
                        .iter()
                        .zip(&factors)
                        .map(|(&id, factor)| factor.label(id).clone()),
                )
            })
            .collect();
        let product = Self::from_validated(labels, relation);
        let projections = factors
            .iter()
            .enumerate()
            .map(|(index, factor)| {
                PosetMap::from_validated(
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

    /// Constructs the disjoint union (coproduct) of finitely many posets.
    ///
    /// The element `x` of the `i`th summand is labelled `(i, x)`. Elements of
    /// different summands are incomparable.
    pub fn disjoint_union<I>(summands: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Poset>,
    {
        Self::disjoint_union_with_inclusions(summands).0
    }

    /// Constructs the disjoint union of finitely many posets together with
    /// its inclusions, one for each summand.
    pub fn disjoint_union_with_inclusions<I>(summands: I) -> (Self, Vec<PosetMap>)
    where
        I: IntoIterator,
        I::Item: Borrow<Poset>,
    {
        let summands = summands
            .into_iter()
            .map(|summand| summand.borrow().clone())
            .collect::<Vec<_>>();
        let total = summands.iter().map(Poset::size).sum();
        let mut labels = Vec::with_capacity(total);
        let mut relation = vec![BitVec::repeat(false, total); total];
        let mut offsets = Vec::with_capacity(summands.len());
        for (index, summand) in summands.iter().enumerate() {
            let offset = labels.len();
            offsets.push(offset);
            labels.extend(
                summand
                    .labels()
                    .iter()
                    .map(|label| Label::from((index, label))),
            );
            for edge in summand.all_relations_iter() {
                relation[offset + edge.from].set(offset + edge.to, true);
            }
        }
        let union = Self::from_validated(labels, relation);
        let inclusions = summands
            .iter()
            .zip(offsets)
            .map(|(summand, offset)| {
                PosetMap::from_validated(
                    summand.clone(),
                    union.clone(),
                    (offset..offset + summand.size()).collect(),
                )
            })
            .collect();
        (union, inclusions)
    }

    /// Returns the opposite poset, with the same labels and element ids and
    /// every relation reversed.
    pub fn opposite(&self) -> Self {
        Self {
            data: Arc::new(PosetData::new(
                self.data.labels.clone(),
                self.data.index.clone(),
                Arc::clone(&self.data.relation_transpose),
                Arc::clone(&self.data.relation),
            )),
        }
    }

    /// Returns the same ordered set with new labels.
    ///
    /// The function receives each element's id and current label. The new
    /// labels must again be distinct. For example, `|_, label| ("x", label)`
    /// tags every label.
    pub fn relabelled<F, L>(&self, mut f: F) -> Result<Self, PosetError>
    where
        F: FnMut(ElementId, Label) -> L,
        L: Into<Label>,
    {
        let labels = self
            .data
            .labels
            .iter()
            .enumerate()
            .map(|(id, label)| f(id, label.clone()).into())
            .collect();
        Self::from_parts(
            labels,
            Arc::clone(&self.data.relation),
            Some(Arc::clone(&self.data.relation_transpose)),
        )
    }

    /// Returns the subposet on the given elements, which keep their labels.
    ///
    /// Ids in the result follow the order in which the elements are given.
    pub fn subposet<I>(&self, elements: I) -> Result<Self, PosetError>
    where
        I: IntoIterator<Item = ElementId>,
    {
        let elements = elements.into_iter().collect::<Vec<_>>();
        if let Some(&bad) = elements.iter().find(|&&id| id >= self.size()) {
            return Err(PosetError::EdgeOutOfBounds {
                edge: Edge::new(bad, bad),
                len: self.size(),
            });
        }
        let relation = elements
            .iter()
            .map(|&i| elements.iter().map(|&j| self.leq(i, j)).collect())
            .collect();
        Self::from_relation(elements.iter().map(|&id| self.label(id).clone()), relation)
    }

    pub(crate) fn from_validated(labels: Vec<Label>, relation: Vec<BitVec>) -> Self {
        debug_assert!(validate_relation(labels.len(), &relation).is_ok());
        Self::from_parts(labels, Arc::new(relation), None)
            .expect("internally constructed posets have distinct labels")
    }

    fn from_parts(
        labels: Vec<Label>,
        relation: Arc<Vec<BitVec>>,
        relation_transpose: Option<Arc<Vec<BitVec>>>,
    ) -> Result<Self, PosetError> {
        let index = label_index(&labels)?;
        let relation_transpose =
            relation_transpose.unwrap_or_else(|| Arc::new(transpose(&relation)));
        Ok(Self {
            data: Arc::new(PosetData::new(labels, index, relation, relation_transpose)),
        })
    }

    pub(crate) fn data(&self) -> &PosetData {
        &self.data
    }

    /// Returns whether two handles refer to the very same stored poset.
    ///
    /// This is stronger than `==`, which compares labels and relations.
    pub fn ptr_eq(&self, other: &Poset) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// Returns the number of elements.
    pub fn size(&self) -> usize {
        self.data.labels.len()
    }

    /// Returns whether the poset has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.labels.is_empty()
    }

    /// Returns the element ids `0..size`, convenient for loops.
    pub fn ids(&self) -> Range<ElementId> {
        0..self.size()
    }

    /// Returns all labels in `ElementId` order.
    pub fn labels(&self) -> &[Label] {
        &self.data.labels
    }

    /// Returns the label of an element.
    ///
    /// Panics if `id` is out of bounds.
    pub fn label(&self, id: ElementId) -> &Label {
        &self.data.labels[id]
    }

    /// Returns the id of the element with the given label.
    ///
    /// ```
    /// use hccr::poset::Poset;
    ///
    /// let p = Poset::from_covers([(0, "a"), (1, "b")], [((0, "a"), (1, "b"))])?;
    /// assert_eq!(p.id((1, "b"))?, 1);
    /// assert!(p.id("missing").is_err());
    /// # Ok::<(), hccr::poset::PosetError>(())
    /// ```
    pub fn id(&self, label: impl Into<Label>) -> Result<ElementId, PosetError> {
        let label = label.into();
        self.try_id(&label)
            .ok_or(PosetError::UnknownLabel { label })
    }

    /// Returns the id of the element with the given label, if there is one.
    pub fn try_id(&self, label: &Label) -> Option<ElementId> {
        self.data.index.get(label).copied()
    }

    /// Returns the relation `lower <= upper` between the elements with the
    /// given labels. This does not check that the relation holds.
    pub fn edge(
        &self,
        lower: impl Into<Label>,
        upper: impl Into<Label>,
    ) -> Result<Edge, PosetError> {
        Ok(Edge::new(self.id(lower)?, self.id(upper)?))
    }

    /// Returns the dense order matrix.
    ///
    /// The returned rows satisfy `relation_matrix()[i][j] == true` exactly
    /// when `i <= j`.
    pub fn relation_matrix(&self) -> &[BitVec] {
        &self.data.relation
    }

    /// Returns the transposed dense order matrix.
    ///
    /// The returned rows satisfy `relation_matrix_transpose()[j][i] == true`
    /// exactly when `i <= j`. Thus row `j` is the principal lower set of `j`.
    pub fn relation_matrix_transpose(&self) -> &[BitVec] {
        &self.data.relation_transpose
    }

    /// Tests the order relation `left <= right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn leq(&self, left: ElementId, right: ElementId) -> bool {
        self.data.relation[left][right]
    }

    /// Returns a bottom element, if one exists.
    pub fn try_bottom(&self) -> Option<ElementId> {
        self.data
            .relation
            .iter()
            .position(|row| row.count_ones() == self.size())
    }

    /// Returns a top element, if one exists.
    pub fn try_top(&self) -> Option<ElementId> {
        self.data
            .relation_transpose
            .iter()
            .position(|column| column.count_ones() == self.size())
    }

    /// Returns the greatest lower bound of two elements, if it exists.
    ///
    /// Returns `None` when either input id is out of bounds or when the two
    /// elements have no meet.
    pub fn try_meet(&self, left: ElementId, right: ElementId) -> Option<ElementId> {
        if left >= self.size() || right >= self.size() {
            return None;
        }
        let lower_bounds = intersection(
            &self.data.relation_transpose[left],
            &self.data.relation_transpose[right],
        );
        lower_bounds
            .iter_ones()
            .find(|&candidate| is_subset(&lower_bounds, &self.data.relation_transpose[candidate]))
    }

    /// Returns the least upper bound of two elements, if it exists.
    ///
    /// Returns `None` when either input id is out of bounds or when the two
    /// elements have no join.
    pub fn try_join(&self, left: ElementId, right: ElementId) -> Option<ElementId> {
        if left >= self.size() || right >= self.size() {
            return None;
        }
        let upper_bounds = intersection(&self.data.relation[left], &self.data.relation[right]);
        upper_bounds
            .iter_ones()
            .find(|&candidate| is_subset(&upper_bounds, &self.data.relation[candidate]))
    }

    /// Returns whether every pair of elements has both meet and join.
    #[must_use]
    pub fn is_lattice(&self) -> bool {
        crate::lattice::Lattice::new(self.clone()).is_ok()
    }

    /// Returns whether every pair of elements is comparable.
    #[must_use]
    pub fn is_total_order(&self) -> bool {
        (0..self.size()).all(|i| (i + 1..self.size()).all(|j| self.leq(i, j) || self.leq(j, i)))
    }

    /// Iterates over all ordered pairs `x <= y`, including identities.
    ///
    /// Relations are yielded in row-major order with respect to the internal
    /// matrix: increasing lower element, then increasing upper element.
    pub fn all_relations_iter(&self) -> impl Iterator<Item = Edge> + '_ {
        self.data
            .relation
            .iter()
            .enumerate()
            .flat_map(|(from, row)| row.iter_ones().map(move |to| Edge { from, to }))
    }

    /// Iterates over all non-identity ordered pairs `x < y`.
    pub fn proper_relations_iter(&self) -> impl Iterator<Item = Edge> + '_ {
        self.all_relations_iter().filter(|edge| !edge.is_identity())
    }

    /// Returns the cover relations in the Hasse diagram.
    ///
    /// A relation `x < y` is a cover when no `z` satisfies `x < z < y`.
    pub fn cover_relations(&self) -> EdgeSet {
        let mut result = EdgeSet::new();
        for (from, upper_set) in self.data.relation.iter().enumerate() {
            let mut covers = upper_set.clone();
            covers.set(from, false);
            for middle in upper_set.iter_ones().filter(|&middle| middle != from) {
                if !covers[middle] {
                    continue;
                }
                difference_assign(&mut covers, &self.data.relation[middle]);
                covers.set(middle, true);
            }
            result.extend(covers.iter_ones().map(|to| Edge::new(from, to)));
        }
        result
    }

    /// Returns the cover relations in the Hasse diagram, sorted.
    pub fn sorted_cover_relations(&self) -> Vec<Edge> {
        let mut covers = self.cover_relations().into_iter().collect::<Vec<_>>();
        covers.sort_unstable();
        covers
    }

    /// Returns all minimal elements.
    pub fn minimal_elements(&self) -> Vec<ElementId> {
        self.data
            .relation_transpose
            .iter()
            .enumerate()
            .filter_map(|(id, lower_set)| (lower_set.count_ones() == 1).then_some(id))
            .collect()
    }

    /// Returns all maximal elements.
    pub fn maximal_elements(&self) -> Vec<ElementId> {
        self.data
            .relation
            .iter()
            .enumerate()
            .filter_map(|(id, upper_set)| (upper_set.count_ones() == 1).then_some(id))
            .collect()
    }

    /// Computes the left lifting class of a set of arrows.
    ///
    /// Regard a poset as a category with one arrow `x -> y` exactly when
    /// `x <= y`. An arrow is in the left lifting class when it has the left
    /// lifting property against every arrow in `arrows`: every commutative
    /// square admits a diagonal filler. Because a poset has at most one arrow
    /// between two objects, this reduces to the Boolean condition
    ///
    /// `!(edge1.from <= edge2.from) || !(edge1.to <= edge2.to) || edge1.to <= edge2.from`.
    ///
    /// Identity relations are always included.
    pub fn llc(&self, arrows: &EdgeSet) -> EdgeSet {
        let mut excluded = vec![BitVec::repeat(false, self.size()); self.size()];
        for edge in arrows {
            if edge.is_identity() {
                continue;
            }
            // The lifting condition fails for x <= y against a <= b exactly
            // when x <= a, y <= b, and y !<= a.
            let mut possible_tos = self.data.relation_transpose[edge.to].clone();
            difference_assign(&mut possible_tos, &self.data.relation_transpose[edge.from]);
            for from in self.data.relation_transpose[edge.from].iter_ones() {
                union_assign(&mut excluded[from], &possible_tos);
            }
        }

        let mut result = EdgeSet::new();
        for (from, upper_set) in self.data.relation.iter().enumerate() {
            let mut included = upper_set.clone();
            difference_assign(&mut included, &excluded[from]);
            result.extend(included.iter_ones().map(|to| Edge::new(from, to)));
        }
        result
    }

    /// Computes the right lifting class of a set of arrows.
    ///
    /// This is dual to [`Poset::llc`]: it returns all relations `edge2` that
    /// satisfy the lifting condition against every `edge1` in `arrows`.
    /// Identity relations are always included.
    pub fn rlc(&self, arrows: &EdgeSet) -> EdgeSet {
        let mut excluded = vec![BitVec::repeat(false, self.size()); self.size()];
        for edge in arrows {
            if edge.is_identity() {
                continue;
            }
            // Dually, x <= y fails to lift against a <= b exactly when
            // x <= a, y <= b, and y !<= a.
            let mut possible_froms = self.data.relation[edge.from].clone();
            difference_assign(&mut possible_froms, &self.data.relation[edge.to]);
            for from in possible_froms.iter_ones() {
                union_assign(&mut excluded[from], &self.data.relation[edge.to]);
            }
        }

        let mut result = EdgeSet::new();
        for (from, upper_set) in self.data.relation.iter().enumerate() {
            let mut included = upper_set.clone();
            difference_assign(&mut included, &excluded[from]);
            result.extend(included.iter_ones().map(|to| Edge::new(from, to)));
        }
        result
    }

    /// Returns whether a class of relations has the 2-out-of-3 property.
    ///
    /// For every composable pair `x <= y <= z`, if any two of `x <= y`,
    /// `y <= z`, and their composite `x <= z` belong to `class`, then all
    /// three must belong to it. Returns `false` if `class` contains an edge
    /// that is not a relation of this poset.
    #[must_use]
    pub fn two_out_of_three(&self, class: &EdgeSet) -> bool {
        if class.iter().any(|edge| {
            edge.from >= self.size() || edge.to >= self.size() || !self.leq(edge.from, edge.to)
        }) {
            return false;
        }

        for middle in 0..self.size() {
            for from in self.data.relation_transpose[middle].iter_ones() {
                let first = class.contains(&Edge::new(from, middle));
                for to in self.data.relation[middle].iter_ones() {
                    let second = class.contains(&Edge::new(middle, to));
                    let composite = class.contains(&Edge::new(from, to));
                    if usize::from(first) + usize::from(second) + usize::from(composite) == 2 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn fmt_named(&self, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{name} on {{")?;
        for (index, label) in self.labels().iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{label}")?;
        }
        f.write_str("} with covers [")?;
        for (index, edge) in self.sorted_cover_relations().into_iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} < {}", self.label(edge.from), self.label(edge.to))?;
        }
        f.write_str("]")
    }
}

impl PosetData {
    fn new(
        labels: Vec<Label>,
        index: HashMap<Label, ElementId>,
        relation: Arc<Vec<BitVec>>,
        relation_transpose: Arc<Vec<BitVec>>,
    ) -> Self {
        Self {
            labels,
            index,
            relation,
            relation_transpose,
            lattice: OnceLock::new(),
            transfer_universe: OnceLock::new(),
            cotransfer_universe: OnceLock::new(),
        }
    }
}

impl PartialEq for Poset {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
            || (self.data.labels == other.data.labels && self.data.relation == other.data.relation)
    }
}

impl Eq for Poset {}

impl fmt::Display for Poset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_named("Poset", f)
    }
}

impl fmt::Debug for Poset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Computes the composites of arrows in `class2` followed by arrows in `class1`.
///
/// Thus `x <= y` from `class2` and `y <= z` from `class1` contribute
/// `x <= z`. Arrows that cannot be joined at their middle endpoint contribute
/// nothing.
pub fn compose(class1: &EdgeSet, class2: &EdgeSet) -> EdgeSet {
    let class2_by_target = sources_by_target(class2);
    let mut result = EdgeSet::new();
    for edge1 in class1 {
        let Some(sources) = class2_by_target.get(&edge1.from) else {
            continue;
        };
        for &source in sources {
            result.insert(Edge::new(source, edge1.to));
        }
    }
    result
}

/// Returns whether a class of relations is closed under composition.
///
/// Composition is computed in the thin category associated to a poset: given
/// `x <= y` and `y <= z`, closure requires the composite relation `x <= z`.
#[must_use]
pub fn composition_closed(class: &EdgeSet) -> bool {
    let class_by_target = sources_by_target(class);
    class.iter().all(|edge1| {
        class_by_target.get(&edge1.from).is_none_or(|sources| {
            sources
                .iter()
                .all(|&source| class.contains(&Edge::new(source, edge1.to)))
        })
    })
}

fn sources_by_target(class: &EdgeSet) -> HashMap<ElementId, Vec<ElementId>> {
    let mut result = HashMap::new();
    for edge in class {
        result
            .entry(edge.to)
            .or_insert_with(Vec::new)
            .push(edge.from);
    }
    result
}

fn label_index(labels: &[Label]) -> Result<HashMap<Label, ElementId>, PosetError> {
    let mut index = HashMap::with_capacity(labels.len());
    for (id, label) in labels.iter().enumerate() {
        if index.insert(label.clone(), id).is_some() {
            return Err(PosetError::DuplicateLabel {
                label: label.clone(),
            });
        }
    }
    Ok(index)
}

/// Returns the coordinate tuples of a product, in lexicographic order, and
/// the componentwise order relation on them.
pub(crate) fn product_coordinates(factors: &[&[BitVec]]) -> (Vec<Vec<ElementId>>, Vec<BitVec>) {
    let mut coordinates = vec![Vec::new()];
    let mut relation = vec![BitVec::repeat(true, 1)];
    for factor in factors {
        let n = coordinates.len();
        let m = factor.len();
        let mut next_coordinates = Vec::with_capacity(n * m);
        for coordinate in &coordinates {
            for id in 0..m {
                let mut next = coordinate.clone();
                next.push(id);
                next_coordinates.push(next);
            }
        }
        let mut next_relation = vec![BitVec::repeat(false, n * m); n * m];
        for (i1, left_upper_set) in relation.iter().enumerate() {
            for (j1, right_upper_set) in factor.iter().enumerate() {
                let upper_set = &mut next_relation[i1 * m + j1];
                for i2 in left_upper_set.iter_ones() {
                    for j2 in right_upper_set.iter_ones() {
                        upper_set.set(i2 * m + j2, true);
                    }
                }
            }
        }
        coordinates = next_coordinates;
        relation = next_relation;
    }
    (coordinates, relation)
}

pub(crate) fn validate_relation(len: usize, relation: &[BitVec]) -> Result<(), PosetError> {
    if relation.len() != len {
        return Err(PosetError::RelationHeight {
            expected: len,
            actual: relation.len(),
        });
    }
    for (row, bits) in relation.iter().enumerate() {
        if bits.len() != len {
            return Err(PosetError::RelationWidth {
                row,
                expected: len,
                actual: bits.len(),
            });
        }
    }
    for (i, row) in relation.iter().enumerate() {
        if !row[i] {
            return Err(PosetError::MissingReflexiveEdge { element: i });
        }
    }
    for (i, row) in relation.iter().enumerate() {
        for (j, other_row) in relation.iter().enumerate().skip(i + 1) {
            if row[j] && other_row[i] {
                return Err(PosetError::NotAntisymmetric { left: i, right: j });
            }
        }
    }
    for (lower, lower_upper_set) in relation.iter().enumerate() {
        for middle in lower_upper_set.iter_ones() {
            if !is_subset(&relation[middle], lower_upper_set) {
                let upper = relation[middle]
                    .iter_ones()
                    .find(|&upper| !lower_upper_set[upper])
                    .expect("a failed subset check should have a witness");
                return Err(PosetError::NotTransitive {
                    lower,
                    middle,
                    upper,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn transitive_closure(relation: &mut [BitVec]) {
    for middle in 0..relation.len() {
        let middle_upper_set = relation[middle].clone();
        for lower_upper_set in relation.iter_mut() {
            if lower_upper_set[middle] {
                union_assign(lower_upper_set, &middle_upper_set);
            }
        }
    }
}
