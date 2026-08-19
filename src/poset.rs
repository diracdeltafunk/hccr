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
//! Labels are not required to be unique. Public APIs therefore refer to an
//! element by its [`ElementId`], a stable position in the label vector. An
//! [`Edge`] is an ordered pair of ids, oriented from the smaller element to the
//! larger one.

use crate::bitvec_utils::{difference_assign, intersection, is_subset, transpose, union_assign};
use crate::morphism::{PosetMap, PosetMapError};
use bitvec::prelude::*;
use either::Either;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// The stable index of an element in a finite poset.
///
/// Ids run from `0` through `poset.size() - 1` and index the slice returned by
/// [`Poset::elements`]. They describe a particular stored presentation of a
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
        }
    }
}

impl std::error::Error for PosetError {}

/// A finite poset stored as dense upper- and lower-set matrices.
///
/// `relation[i][j]` is true if and only if `i <= j`. Element labels are not
/// required to be unique.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poset<A> {
    elements: Vec<A>,
    relation: Arc<Vec<BitVec>>,
    relation_transpose: Arc<Vec<BitVec>>,
}

/// The coproduct, or disjoint union, of two finite posets.
///
/// No order relations are added between the two summands.  Thus an element
/// from the left summand is incomparable with every element from the right
/// summand, and vice versa.
#[derive(Debug, Clone)]
pub struct PosetCoproduct<A, B> {
    /// The disjoint union poset, with labels tagged by [`either::Either`].
    pub poset: Arc<Poset<Either<A, B>>>,
    /// The canonical order embedding of the left summand.
    pub left: PosetMap<A, Either<A, B>>,
    /// The canonical order embedding of the right summand.
    pub right: PosetMap<B, Either<A, B>>,
}

/// The categorical product of two finite posets and its projections.
#[derive(Debug, Clone)]
pub struct PosetProduct<A, B> {
    /// The product poset with componentwise order.
    pub poset: Arc<Poset<(A, B)>>,
    /// The first projection `(a, b) |-> a`.
    pub left_projection: PosetMap<(A, B), A>,
    /// The second projection `(a, b) |-> b`.
    pub right_projection: PosetMap<(A, B), B>,
}

impl<A> Poset<A> {
    /// Constructs the poset whose order relation is given by `pred`.
    ///
    /// For labels `a` and `b`, `pred(a, b)` is interpreted as `a <= b`. The
    /// complete relation is evaluated and then checked for reflexivity,
    /// antisymmetry, and transitivity; an arbitrary predicate is therefore
    /// allowed as input but may produce a [`PosetError`].
    pub fn from_vec_by<F>(elements: Vec<A>, pred: F) -> Result<Self, PosetError>
    where
        F: Fn(&A, &A) -> bool,
    {
        let relation = elements
            .iter()
            .map(|a| elements.iter().map(|b| pred(a, b)).collect())
            .collect();
        Self::from_relation(elements, relation)
    }
}

impl<A: PartialOrd> Poset<A> {
    /// Constructs the poset defined by Rust's `PartialOrd` relation.
    pub fn from_vec(elements: Vec<A>) -> Result<Self, PosetError> {
        Poset::from_vec_by(elements, |a, b| a <= b)
    }
}

impl Poset<usize> {
    /// Constructs the chain `[top] = {0, ..., top}` with its usual total order.
    ///
    /// The argument is the largest element, so the resulting poset has
    /// `top + 1` elements.
    pub fn chain(top: usize) -> Result<Self, PosetError> {
        Poset::from_vec((0..=top).collect())
    }
}

impl<A> Poset<A> {
    /// Constructs a poset from an already-computed order relation.
    ///
    /// The matrix must be square of size `elements.len()`, reflexive,
    /// antisymmetric, and transitive.  Entry `[i][j]` is interpreted as
    /// `i <= j`.
    pub fn from_relation(elements: Vec<A>, relation: Vec<BitVec>) -> Result<Self, PosetError> {
        validate_relation(elements.len(), &relation)?;
        let relation_transpose = transpose(&relation);
        Ok(Self {
            elements,
            relation: Arc::new(relation),
            relation_transpose: Arc::new(relation_transpose),
        })
    }

    /// Returns the same ordered set with labels transformed by `f`.
    ///
    /// The order relation is copied exactly; only the element labels change.
    pub fn relabelled<B, F>(&self, f: F) -> Poset<B>
    where
        F: FnMut(&A) -> B,
    {
        Poset {
            elements: self.elements.iter().map(f).collect(),
            relation: Arc::clone(&self.relation),
            relation_transpose: Arc::clone(&self.relation_transpose),
        }
    }

    /// Tests whether two posets use the same ordered element coordinates.
    ///
    /// Sharing both immutable relation matrices means that the element ids in
    /// the two posets refer to the same ordered presentation, even when their
    /// element labels have different types.
    pub(crate) fn shares_order_coordinates_with<B>(&self, other: &Poset<B>) -> bool {
        Arc::ptr_eq(&self.relation, &other.relation)
            && Arc::ptr_eq(&self.relation_transpose, &other.relation_transpose)
    }

    /// Produces the smallest partial order containing the given edges.
    ///
    /// The supplied edges may be a Hasse diagram, a redundant collection of
    /// comparable pairs, or any other generating relation. Identity edges are
    /// added and transitive consequences are computed with the finite
    /// Warshall closure algorithm. The result is then checked for
    /// antisymmetry; in particular, generators that create a directed cycle
    /// return [`PosetError::NotAntisymmetric`].
    pub fn from_edges(
        elements: Vec<A>,
        edges: impl IntoIterator<Item = Edge>,
    ) -> Result<Self, PosetError> {
        let n = elements.len();
        let mut relation = vec![BitVec::repeat(false, n); n];
        for (i, row) in relation.iter_mut().enumerate() {
            row.set(i, true);
        }
        for edge in edges {
            if edge.from >= n || edge.to >= n {
                return Err(PosetError::EdgeOutOfBounds { edge, len: n });
            }
            relation[edge.from].set(edge.to, true);
        }
        transitive_closure(&mut relation);
        Self::from_relation(elements, relation)
    }

    pub(crate) fn from_validated(elements: Vec<A>, relation: Vec<BitVec>) -> Self {
        debug_assert!(validate_relation(elements.len(), &relation).is_ok());
        let relation_transpose = transpose(&relation);
        Self {
            elements,
            relation: Arc::new(relation),
            relation_transpose: Arc::new(relation_transpose),
        }
    }

    /// Returns the number of elements.
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Returns whether the poset has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Returns all element labels in `ElementId` order.
    pub fn elements(&self) -> &[A] {
        &self.elements
    }

    /// Returns the label of an element by id.
    pub fn element(&self, id: ElementId) -> Option<&A> {
        self.elements.get(id)
    }

    /// Returns the dense order matrix.
    ///
    /// The returned rows satisfy `relation_matrix()[i][j] == true` exactly
    /// when `i <= j`.
    pub fn relation_matrix(&self) -> &[BitVec] {
        &self.relation
    }

    /// Returns the transposed dense order matrix.
    ///
    /// The returned rows satisfy `relation_matrix_transpose()[j][i] == true`
    /// exactly when `i <= j`. Thus row `j` is the principal lower set of `j`.
    pub fn relation_matrix_transpose(&self) -> &[BitVec] {
        &self.relation_transpose
    }

    /// Tests the order relation `left <= right`.
    ///
    /// Panics if either id is out of bounds.
    pub fn leq(&self, left: ElementId, right: ElementId) -> bool {
        self.relation[left][right]
    }

    /// Returns a bottom element, if one exists.
    ///
    /// A bottom element is an element `bot` such that `bot <= x` for every
    /// element `x`.
    pub fn bottom(&self) -> Option<ElementId> {
        self.relation
            .iter()
            .position(|row| row.count_ones() == self.elements.len())
    }

    /// Returns a top element, if one exists.
    ///
    /// A top element is an element `top` such that `x <= top` for every
    /// element `x`.
    pub fn top(&self) -> Option<ElementId> {
        self.relation_transpose
            .iter()
            .position(|column| column.count_ones() == self.size())
    }

    /// Returns the greatest lower bound (meet) of two elements, if it exists.
    ///
    /// Returns `None` when either input id is out of bounds or when the two
    /// elements have no meet. The algorithm intersects their principal lower
    /// sets and looks for a member lying above every element of that
    /// intersection.
    pub fn meet(&self, left: ElementId, right: ElementId) -> Option<ElementId> {
        if left >= self.size() || right >= self.size() {
            return None;
        }
        let lower_bounds = intersection(
            &self.relation_transpose[left],
            &self.relation_transpose[right],
        );
        lower_bounds
            .iter_ones()
            .find(|&candidate| is_subset(&lower_bounds, &self.relation_transpose[candidate]))
    }

    /// Returns the least upper bound (join) of two elements, if it exists.
    ///
    /// Returns `None` when either input id is out of bounds or when the two
    /// elements have no join. The algorithm intersects their principal upper
    /// sets and looks for a member lying below every element of that
    /// intersection.
    pub fn join(&self, left: ElementId, right: ElementId) -> Option<ElementId> {
        if left >= self.size() || right >= self.size() {
            return None;
        }
        let upper_bounds = intersection(&self.relation[left], &self.relation[right]);
        upper_bounds
            .iter_ones()
            .find(|&candidate| is_subset(&upper_bounds, &self.relation[candidate]))
    }

    /// Returns whether every pair of elements has both meet and join.
    #[must_use]
    pub fn is_lattice(&self) -> bool {
        !self.is_empty()
            && (0..self.size()).all(|i| {
                (i + 1..self.size()).all(|j| self.meet(i, j).is_some() && self.join(i, j).is_some())
            })
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
    pub fn all_relations_iter(&self) -> impl Iterator<Item = Edge> {
        self.relation
            .iter()
            .enumerate()
            .flat_map(|(from, row)| row.iter_ones().map(move |to| Edge { from, to }))
    }

    /// Iterates over all non-identity ordered pairs `x < y`.
    pub fn proper_relations_iter(&self) -> impl Iterator<Item = Edge> {
        self.all_relations_iter().filter(|edge| !edge.is_identity())
    }

    /// Returns the cover relations in the Hasse diagram.
    ///
    /// A relation `x < y` is a cover when no `z` satisfies `x < z < y`. The
    /// implementation starts with the whole strict upper set of each `x` and
    /// removes elements known to lie above a nearer candidate.
    pub fn cover_relations(&self) -> EdgeSet {
        let mut result = EdgeSet::new();
        for (from, upper_set) in self.relation.iter().enumerate() {
            let mut covers = upper_set.clone();
            covers.set(from, false);
            for middle in upper_set.iter_ones().filter(|&middle| middle != from) {
                if !covers[middle] {
                    continue;
                }
                difference_assign(&mut covers, &self.relation[middle]);
                covers.set(middle, true);
            }
            result.extend(covers.iter_ones().map(|to| Edge::new(from, to)));
        }
        result
    }

    /// Returns all minimal elements.
    ///
    /// A minimal element has no strictly smaller element below it.
    pub fn minimal_elements(&self) -> Vec<ElementId> {
        self.relation_transpose
            .iter()
            .enumerate()
            .filter_map(|(id, lower_set)| (lower_set.count_ones() == 1).then_some(id))
            .collect()
    }

    /// Returns all maximal elements.
    ///
    /// A maximal element has no strictly larger element above it.
    pub fn maximal_elements(&self) -> Vec<ElementId> {
        self.relation
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
            let mut possible_tos = self.relation_transpose[edge.to].clone();
            difference_assign(&mut possible_tos, &self.relation_transpose[edge.from]);
            for from in self.relation_transpose[edge.from].iter_ones() {
                union_assign(&mut excluded[from], &possible_tos);
            }
        }

        let mut result = EdgeSet::new();
        for (from, upper_set) in self.relation.iter().enumerate() {
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
            let mut possible_froms = self.relation[edge.from].clone();
            difference_assign(&mut possible_froms, &self.relation[edge.to]);
            for from in possible_froms.iter_ones() {
                union_assign(&mut excluded[from], &self.relation[edge.to]);
            }
        }

        let mut result = EdgeSet::new();
        for (from, upper_set) in self.relation.iter().enumerate() {
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
            for from in self.relation_transpose[middle].iter_ones() {
                let first = class.contains(&Edge::new(from, middle));
                for to in self.relation[middle].iter_ones() {
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

/// Returns whether a class of relations in `poset` has the 2-out-of-3 property.
#[must_use]
pub fn two_out_of_three<A>(poset: &Poset<A>, class: &EdgeSet) -> bool {
    poset.two_out_of_three(class)
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

/// Constructs the disjoint union of two posets and its canonical inclusions.
pub fn disjoint_union<A: Clone, B: Clone>(
    left: Arc<Poset<A>>,
    right: Arc<Poset<B>>,
) -> Result<PosetCoproduct<A, B>, PosetMapError> {
    Poset::<Either<A, B>>::disjoint_union(left, right)
}

/// Constructs the direct product of two posets and its canonical projections.
pub fn product<A: Clone, B: Clone>(
    left: Arc<Poset<A>>,
    right: Arc<Poset<B>>,
) -> Result<PosetProduct<A, B>, PosetMapError> {
    Poset::<(A, B)>::product(left, right)
}

impl<A: Clone, B: Clone> Poset<Either<A, B>> {
    /// Constructs the disjoint union of two posets and its canonical inclusions.
    ///
    /// This is the same construction as [`disjoint_union`], exposed as an
    /// associated function on the tagged label type.
    pub fn disjoint_union(
        left: Arc<Poset<A>>,
        right: Arc<Poset<B>>,
    ) -> Result<PosetCoproduct<A, B>, PosetMapError> {
        let n = left.size();
        let m = right.size();
        let mut elements = Vec::with_capacity(n + m);
        elements.extend(left.elements().iter().cloned().map(Either::Left));
        elements.extend(right.elements().iter().cloned().map(Either::Right));

        let mut relation = vec![BitVec::repeat(false, n + m); n + m];
        for edge in left.all_relations_iter() {
            relation[edge.from].set(edge.to, true);
        }
        for edge in right.all_relations_iter() {
            relation[n + edge.from].set(n + edge.to, true);
        }

        let poset = Arc::new(Poset::from_validated(elements, relation));
        let left_map = PosetMap::new(left, Arc::clone(&poset), (0..n).collect())?;
        let right_map = PosetMap::new(right, Arc::clone(&poset), (n..n + m).collect())?;
        Ok(PosetCoproduct {
            poset,
            left: left_map,
            right: right_map,
        })
    }
}

impl<A: Clone, B: Clone> Poset<(A, B)> {
    /// Constructs the direct product of two posets and its canonical projections.
    ///
    /// The order is componentwise: `(a, b) <= (a', b')` if and only if
    /// `a <= a'` in the left factor and `b <= b'` in the right factor.
    pub fn product(
        left: Arc<Poset<A>>,
        right: Arc<Poset<B>>,
    ) -> Result<PosetProduct<A, B>, PosetMapError> {
        let n = left.size();
        let m = right.size();
        let mut elements = Vec::with_capacity(n * m);
        for a in left.elements() {
            for b in right.elements() {
                elements.push((a.clone(), b.clone()));
            }
        }

        let mut relation = vec![BitVec::repeat(false, n * m); n * m];
        for (i1, left_upper_set) in left.relation_matrix().iter().enumerate() {
            for (j1, right_upper_set) in right.relation_matrix().iter().enumerate() {
                let product_upper_set = &mut relation[i1 * m + j1];
                for i2 in left_upper_set.iter_ones() {
                    for j2 in right_upper_set.iter_ones() {
                        product_upper_set.set(i2 * m + j2, true);
                    }
                }
            }
        }

        let poset = Arc::new(Poset::from_validated(elements, relation));
        let left_projection = PosetMap::new(
            Arc::clone(&poset),
            left,
            (0..n).flat_map(|i| std::iter::repeat_n(i, m)).collect(),
        )?;
        let right_projection = PosetMap::new(
            Arc::clone(&poset),
            right,
            (0..n).flat_map(|_| 0..m).collect(),
        )?;
        Ok(PosetProduct {
            poset,
            left_projection,
            right_projection,
        })
    }
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
