//! Model structures on finite lattices.
//!
//! A model structure on a finite lattice, viewed as a category, is determined
//! by an interval `R <= R'` of transfer systems satisfying one extra condition.
//! Write `llc(S)` for the left lifting class of a class of relations `S` (see
//! [`Poset::left_lifting_class`](crate::poset::Poset::left_lifting_class)). The
//! five classes of maps are
//!
//! - acyclic fibrations `R`;
//! - fibrations `R'`;
//! - cofibrations `llc(R)`;
//! - acyclic cofibrations `llc(R')`; and
//! - weak equivalences `R ∘ llc(R')`, the composites of an acyclic cofibration
//!   followed by an acyclic fibration.
//!
//! The extra condition is that the weak equivalences have the 2-out-of-3
//! property. [`ModelStructure::new`] checks it, so every [`ModelStructure`] is a
//! genuine model structure rather than merely a pair of weak factorization
//! systems. [`Lattice::model_structures`](crate::lattice::Lattice::model_structures)
//! lists them all.

use crate::cotransfer_lattice::CotransferSystem;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, compose};
use crate::transfer_lattice::TransferSystem;
use std::fmt;

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
    /// The two transfer systems are on different lattices.
    DifferentLattices,
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
}

impl fmt::Display for ModelStructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentLattices => {
                write!(f, "the two transfer systems are on different lattices")
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
        }
    }
}

impl std::error::Error for ModelStructureError {}

/// A model structure on a finite lattice.
///
/// The structure stores the transfer systems of acyclic fibrations and
/// fibrations, the cotransfer systems of cofibrations and acyclic
/// cofibrations, and the resulting class of weak equivalences.  It can only be
/// constructed after containment and 2-out-of-3 have been checked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelStructure {
    acyclic_fibrations: TransferSystem,
    fibrations: TransferSystem,
    cofibrations: CotransferSystem,
    acyclic_cofibrations: CotransferSystem,
    weak_equivalences: EdgeSet,
}

impl ModelStructure {
    /// Constructs the model structure with acyclic fibrations
    /// `R = acyclic_fibrations` and fibrations `R' = fibrations`.
    ///
    /// # Errors
    ///
    /// - [`ModelStructureError::DifferentLattices`] if the two systems are on
    ///   different lattices;
    /// - [`ModelStructureError::AcyclicFibrationsNotContained`] if `R` is not
    ///   contained in `R'`;
    /// - [`ModelStructureError::WeakEquivalencesFailTwoOutOfThree`] if the
    ///   weak equivalences fail 2-out-of-3. The weak equivalences are the
    ///   composites of an acyclic cofibration (a relation in the
    ///   [left lifting class](Poset::left_lifting_class) of `R'`) followed by
    ///   an acyclic fibration.
    pub fn new(
        acyclic_fibrations: TransferSystem,
        fibrations: TransferSystem,
    ) -> Result<Self, ModelStructureError> {
        if acyclic_fibrations.lattice() != fibrations.lattice() {
            return Err(ModelStructureError::DifferentLattices);
        }

        if let Some(relation) = acyclic_fibrations
            .proper_relations()
            .into_iter()
            .find(|&relation| !fibrations.contains_relation(relation))
        {
            return Err(ModelStructureError::AcyclicFibrationsNotContained { relation });
        }

        let ambient = acyclic_fibrations.lattice().as_poset();
        let acyclic_fibration_edges = acyclic_fibrations.relations();
        let acyclic_cofibration_edges = ambient.left_lifting_class(&fibrations.relations());
        let weak_equivalences = compose(&acyclic_fibration_edges, &acyclic_cofibration_edges);

        if let Some(witness) = two_out_of_three_failure(ambient, &weak_equivalences) {
            return Err(ModelStructureError::WeakEquivalencesFailTwoOutOfThree { witness });
        }

        let cofibrations = acyclic_fibrations.left_lifting_cotransfer();
        let acyclic_cofibrations = fibrations.left_lifting_cotransfer();

        Ok(Self {
            acyclic_fibrations,
            fibrations,
            cofibrations,
            acyclic_cofibrations,
            weak_equivalences,
        })
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Lattice {
        self.fibrations.lattice()
    }

    /// Returns the transfer system of fibrations.
    pub fn fibrations(&self) -> &TransferSystem {
        &self.fibrations
    }

    /// Returns the transfer system of acyclic fibrations.
    pub fn acyclic_fibrations(&self) -> &TransferSystem {
        &self.acyclic_fibrations
    }

    /// Returns the cotransfer system of cofibrations.
    pub fn cofibrations(&self) -> &CotransferSystem {
        &self.cofibrations
    }

    /// Returns the cotransfer system of acyclic cofibrations.
    pub fn acyclic_cofibrations(&self) -> &CotransferSystem {
        &self.acyclic_cofibrations
    }

    /// Returns the class of weak equivalences, identities included.
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

    /// Returns all fibrant objects, in increasing order of id.
    pub fn fibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_fibrant(object))
            .collect()
    }

    /// Returns all cofibrant objects, in increasing order of id.
    pub fn cofibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_cofibrant(object))
            .collect()
    }

    /// Returns all bifibrant objects, in increasing order of id.
    pub fn bifibrant_objects(&self) -> Vec<ElementId> {
        (0..self.lattice().size())
            .filter(|&object| self.is_bifibrant(object))
            .collect()
    }

    /// Factors a relation as a cofibration followed by an acyclic fibration.
    ///
    /// Factorizations need not be unique; this returns the one whose middle
    /// object has the smallest id. Returns `None` if `relation` is not a
    /// relation `x <= y` of the lattice (including when an id is out of
    /// bounds).
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
    /// Factorizations need not be unique; this returns the one whose middle
    /// object has the smallest id. Returns `None` if `relation` is not a
    /// relation `x <= y` of the lattice (including when an id is out of
    /// bounds).
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

impl TryFrom<(TransferSystem, TransferSystem)> for ModelStructure {
    type Error = ModelStructureError;

    fn try_from(
        (acyclic_fibrations, fibrations): (TransferSystem, TransferSystem),
    ) -> Result<Self, Self::Error> {
        Self::new(acyclic_fibrations, fibrations)
    }
}

impl Lattice {
    /// Returns all model structures on this lattice.
    ///
    /// They correspond to the intervals `R <= R'` of the model-structure order
    /// on transfer systems (see
    /// [`Lattice::transfer_systems_model_structure_order`]), and are listed in
    /// lexicographic order of the ids of `(R, R')` in that poset.
    pub fn model_structures(&self) -> Vec<ModelStructure> {
        let order = self.transfer_systems_model_structure_order();
        order
            .relations()
            .into_iter()
            .map(|edge| {
                ModelStructure::new(
                    order.system(edge.from).clone(),
                    order.system(edge.to).clone(),
                )
                .expect("intervals of the model-structure order are model structures")
            })
            .collect()
    }
}

impl fmt::Display for ModelStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model structure with acyclic fibrations {} and fibrations {}",
            self.acyclic_fibrations, self.fibrations
        )
    }
}

fn two_out_of_three_failure(ambient: &Poset, class: &EdgeSet) -> Option<TwoOutOfThreeFailure> {
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

fn factorization_witness<L, R>(
    lattice: &Lattice,
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
