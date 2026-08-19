//! Model structures on finite lattices.
//!
//! A model structure is encoded by an interval `R <= R'` of transfer
//! systems satisfying one additional condition.  Its five classes of maps
//! are
//!
//! - acyclic fibrations `R`;
//! - fibrations `R'`;
//! - cofibrations `llc(R)`;
//! - acyclic cofibrations `llc(R')`; and
//! - weak equivalences `R ∘ llc(R')`.
//!
//! The final class must have the 2-out-of-3 property.  This module validates
//! that condition at construction time, so every [`ModelStructure`] value is
//! a genuine model structure rather than merely a pair of weak factorization
//! systems.

use crate::cotransfer_lattice::{CotransferSystem, CotransferSystemError, CotransferUniverse};
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, compose};
use crate::transfer_lattice::{TransferSystem, TransferUniverse};
use std::fmt;
use std::sync::Arc;

/// A composable triangle witnessing failure of the 2-out-of-3 property.
///
/// Exactly two of `first`, `second`, and `composite` belong to the class in
/// question.  The remaining one is recorded as `missing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoOutOfThreeFailure {
    /// The first arrow `lower -> middle`.
    pub first: Edge,
    /// The second arrow `middle -> upper`.
    pub second: Edge,
    /// The composite arrow `lower -> upper`.
    pub composite: Edge,
    /// The unique arrow in the triangle that is absent from the class.
    pub missing: Edge,
}

/// Two composable arrows exhibiting a factorization through an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorizationWitness {
    /// The first factor, from the source to the intermediate object.
    pub first: Edge,
    /// The second factor, from the intermediate object to the target.
    pub second: Edge,
}

impl FactorizationWitness {
    /// Returns the intermediate object of the factorization.
    pub fn middle(self) -> ElementId {
        self.first.to
    }
}

/// Errors that can occur while constructing a model structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStructureError {
    /// The two transfer systems do not belong to the same universe.
    DifferentUniverses,
    /// An acyclic fibration is not a fibration.
    AcyclicFibrationsNotContained {
        /// A relation belonging to the acyclic fibrations but not the
        /// fibrations.
        relation: Edge,
    },
    /// The proposed weak equivalences fail the 2-out-of-3 property.
    WeakEquivalencesFailTwoOutOfThree {
        /// A composable triangle witnessing the failure.
        witness: TwoOutOfThreeFailure,
    },
    /// Construction of one of the cotransfer systems failed.
    Cotransfer(CotransferSystemError),
}

impl fmt::Display for ModelStructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentUniverses => {
                write!(f, "the two transfer systems belong to different universes")
            }
            Self::AcyclicFibrationsNotContained { relation } => write!(
                f,
                "acyclic fibration {} <= {} is not a fibration",
                relation.from, relation.to
            ),
            Self::WeakEquivalencesFailTwoOutOfThree { witness } => write!(
                f,
                "weak equivalences fail 2-out-of-3 on {} <= {} <= {}; the relation {} <= {} is missing",
                witness.first.from,
                witness.first.to,
                witness.second.to,
                witness.missing.from,
                witness.missing.to
            ),
            Self::Cotransfer(error) => write!(f, "could not construct cofibrations: {error}"),
        }
    }
}

impl std::error::Error for ModelStructureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cotransfer(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CotransferSystemError> for ModelStructureError {
    fn from(error: CotransferSystemError) -> Self {
        Self::Cotransfer(error)
    }
}

/// A model structure on a finite lattice.
///
/// The structure stores the transfer systems of acyclic fibrations and
/// fibrations, the cotransfer systems of cofibrations and acyclic
/// cofibrations, and the resulting class of weak equivalences.  It can only be
/// constructed after containment and 2-out-of-3 have been checked.
#[derive(Debug, Clone)]
pub struct ModelStructure<A> {
    acyclic_fibrations: TransferSystem<A>,
    fibrations: TransferSystem<A>,
    cofibrations: CotransferSystem<A>,
    acyclic_cofibrations: CotransferSystem<A>,
    weak_equivalences: EdgeSet,
}

impl<A> ModelStructure<A> {
    /// Constructs the model structure represented by `R <= R'`.
    ///
    /// Here `R` is `acyclic_fibrations` and `R'` is `fibrations`.  The two
    /// systems must share one transfer universe, `R` must be contained in
    /// `R'`, and `R ∘ llc(R')` must satisfy 2-out-of-3.
    pub fn new(
        acyclic_fibrations: TransferSystem<A>,
        fibrations: TransferSystem<A>,
    ) -> Result<Self, ModelStructureError> {
        if !Arc::ptr_eq(acyclic_fibrations.universe(), fibrations.universe()) {
            return Err(ModelStructureError::DifferentUniverses);
        }

        if let Some(relation) = acyclic_fibrations
            .universe()
            .proper_edges()
            .iter()
            .copied()
            .find(|&relation| {
                acyclic_fibrations.contains_relation(relation)
                    && !fibrations.contains_relation(relation)
            })
        {
            return Err(ModelStructureError::AcyclicFibrationsNotContained { relation });
        }

        let ambient = acyclic_fibrations.lattice().as_poset();
        let acyclic_fibration_edges = acyclic_fibrations.edges(true);
        let acyclic_cofibration_edges = ambient.llc(&fibrations.edges(true));
        let weak_equivalences = compose(&acyclic_fibration_edges, &acyclic_cofibration_edges);

        if let Some(witness) = two_out_of_three_failure(ambient, &weak_equivalences) {
            return Err(ModelStructureError::WeakEquivalencesFailTwoOutOfThree { witness });
        }

        let cotransfer_universe = Arc::new(CotransferUniverse::from_transfer_universe(Arc::clone(
            acyclic_fibrations.universe(),
        )));
        let cofibrations = cotransfer_universe.left_lifting_of(&acyclic_fibrations)?;
        let acyclic_cofibrations = cotransfer_universe.left_lifting_of(&fibrations)?;

        Ok(Self {
            acyclic_fibrations,
            fibrations,
            cofibrations,
            acyclic_cofibrations,
            weak_equivalences,
        })
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.fibrations.lattice()
    }

    /// Returns the shared universe of the two transfer systems.
    pub fn transfer_universe(&self) -> &Arc<TransferUniverse<A>> {
        self.fibrations.universe()
    }

    /// Returns the shared universe of the two cotransfer systems.
    pub fn cotransfer_universe(&self) -> &Arc<CotransferUniverse<A>> {
        self.cofibrations.universe()
    }

    /// Returns the transfer system of fibrations.
    pub fn fibrations(&self) -> &TransferSystem<A> {
        &self.fibrations
    }

    /// Returns the transfer system of acyclic fibrations.
    pub fn acyclic_fibrations(&self) -> &TransferSystem<A> {
        &self.acyclic_fibrations
    }

    /// Returns the cotransfer system of cofibrations.
    pub fn cofibrations(&self) -> &CotransferSystem<A> {
        &self.cofibrations
    }

    /// Returns the cotransfer system of acyclic cofibrations.
    pub fn acyclic_cofibrations(&self) -> &CotransferSystem<A> {
        &self.acyclic_cofibrations
    }

    /// Returns the class of weak equivalences.
    pub fn weak_equivalences(&self) -> &EdgeSet {
        &self.weak_equivalences
    }

    /// Returns whether `relation` is a fibration.
    pub fn contains_fibration(&self, relation: Edge) -> bool {
        self.fibrations.contains_relation(relation)
    }

    /// Returns whether `relation` is an acyclic fibration.
    pub fn contains_acyclic_fibration(&self, relation: Edge) -> bool {
        self.acyclic_fibrations.contains_relation(relation)
    }

    /// Returns whether `relation` is a cofibration.
    pub fn contains_cofibration(&self, relation: Edge) -> bool {
        self.cofibrations.contains_relation(relation)
    }

    /// Returns whether `relation` is an acyclic cofibration.
    pub fn contains_acyclic_cofibration(&self, relation: Edge) -> bool {
        self.acyclic_cofibrations.contains_relation(relation)
    }

    /// Returns whether `relation` is a weak equivalence.
    pub fn contains_weak_equivalence(&self, relation: Edge) -> bool {
        self.weak_equivalences.contains(&relation)
    }

    /// Returns whether an object is fibrant.
    ///
    /// An object `x` is fibrant when `x -> top` is a fibration.  Out-of-range
    /// object ids return `false`.
    pub fn is_fibrant(&self, object: ElementId) -> bool {
        object < self.lattice().size()
            && self.contains_fibration(Edge::new(object, self.lattice().top()))
    }

    /// Returns whether an object is cofibrant.
    ///
    /// An object `x` is cofibrant when `bottom -> x` is a cofibration.
    /// Out-of-range object ids return `false`.
    pub fn is_cofibrant(&self, object: ElementId) -> bool {
        object < self.lattice().size()
            && self.contains_cofibration(Edge::new(self.lattice().bottom(), object))
    }

    /// Returns whether an object is both fibrant and cofibrant.
    pub fn is_bifibrant(&self, object: ElementId) -> bool {
        self.is_fibrant(object) && self.is_cofibrant(object)
    }

    /// Returns all fibrant objects in element-id order.
    pub fn fibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_fibrant(object))
            .collect()
    }

    /// Returns all cofibrant objects in element-id order.
    pub fn cofibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_cofibrant(object))
            .collect()
    }

    /// Returns all bifibrant objects in element-id order.
    pub fn bifibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_bifibrant(object))
            .collect()
    }

    /// Factors a relation as a cofibration followed by an acyclic fibration.
    ///
    /// Returns `None` if `relation` is not an in-range relation of the
    /// underlying lattice.  Every valid relation has a witness.
    pub fn factor_as_cofibration_then_acyclic_fibration(
        &self,
        relation: Edge,
    ) -> Option<FactorizationWitness> {
        factorization_witness(
            self.lattice(),
            relation,
            |edge| self.contains_cofibration(edge),
            |edge| self.contains_acyclic_fibration(edge),
        )
    }

    /// Factors a relation as an acyclic cofibration followed by a fibration.
    ///
    /// Returns `None` if `relation` is not an in-range relation of the
    /// underlying lattice.  Every valid relation has a witness.
    pub fn factor_as_acyclic_cofibration_then_fibration(
        &self,
        relation: Edge,
    ) -> Option<FactorizationWitness> {
        factorization_witness(
            self.lattice(),
            relation,
            |edge| self.contains_acyclic_cofibration(edge),
            |edge| self.contains_fibration(edge),
        )
    }
}

impl<A> TryFrom<(TransferSystem<A>, TransferSystem<A>)> for ModelStructure<A> {
    type Error = ModelStructureError;

    fn try_from(
        (acyclic_fibrations, fibrations): (TransferSystem<A>, TransferSystem<A>),
    ) -> Result<Self, Self::Error> {
        Self::new(acyclic_fibrations, fibrations)
    }
}

fn two_out_of_three_failure<A>(
    ambient: &crate::poset::Poset<A>,
    class: &EdgeSet,
) -> Option<TwoOutOfThreeFailure> {
    for middle in 0..ambient.size() {
        for lower in ambient.relation_matrix_transpose()[middle].iter_ones() {
            let first = Edge::new(lower, middle);
            for upper in ambient.relation_matrix()[middle].iter_ones() {
                let second = Edge::new(middle, upper);
                let composite = Edge::new(lower, upper);
                let membership = [
                    class.contains(&first),
                    class.contains(&second),
                    class.contains(&composite),
                ];
                if membership.iter().filter(|&&present| present).count() == 2 {
                    let missing = [first, second, composite]
                        .into_iter()
                        .zip(membership)
                        .find_map(|(edge, present)| (!present).then_some(edge))
                        .expect("exactly one edge is absent when exactly two are present");
                    return Some(TwoOutOfThreeFailure {
                        first,
                        second,
                        composite,
                        missing,
                    });
                }
            }
        }
    }
    None
}

fn factorization_witness<A, L, R>(
    lattice: &Lattice<A>,
    relation: Edge,
    in_left: L,
    in_right: R,
) -> Option<FactorizationWitness>
where
    L: Fn(Edge) -> bool,
    R: Fn(Edge) -> bool,
{
    if relation.from >= lattice.size()
        || relation.to >= lattice.size()
        || !lattice.leq(relation.from, relation.to)
    {
        return None;
    }

    (0..lattice.size()).find_map(|middle| {
        if lattice.leq(relation.from, middle) && lattice.leq(middle, relation.to) {
            let first = Edge::new(relation.from, middle);
            let second = Edge::new(middle, relation.to);
            if in_left(first) && in_right(second) {
                return Some(FactorizationWitness { first, second });
            }
        }
        None
    })
}
