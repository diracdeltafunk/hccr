//! GAP-backed finite group actions on finite lattices.
//!
//! A `G`-lattice is a finite lattice `L` equipped with an action of a finite
//! GAP group `G` by lattice automorphisms.  The action is stored both as GAP
//! homomorphisms to permutation groups and as validated Rust permutations of
//! lattice element ids.  From the element action, this module induces an action
//! on all lattice relations `x <= y`, including identity relations, and
//! precomputes relation orbits, stabilizers, and transporter elements.
//! GAP values stored in these structures are [`gap_sys::GapValue`]s, so they
//! retain their GAP GC roots for as long as the Rust structures remain alive.
//!
//! G-transfer systems are computed after passing from individual non-identity
//! relations to their `G`-orbits.  Thus a G-transfer system is stored as a set
//! of non-identity relation orbits.

use crate::bitvec_utils::{is_subset, set_partial_cmp};
use crate::group_theory::{self, GapAction, GapSubgroup, GroupTheoryError, PointOrbitError};
use crate::lattice::{Lattice, LatticeError};
use crate::morphism::LatticeMapError;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use crate::transfer_lattice::{RawTransferSystem, TransferSystem, TransferUniverse};
use bitvec::prelude::*;
use fcars::FormalContext;
use gap_sys::{Gap, GapValue};
use std::fmt;
use std::sync::Arc;

/// The formal context whose concepts are transfer systems on a G-lattice.
///
/// Objects and attributes are both non-identity relation orbits, represented by
/// [`RelationOrbitLabel`] values.
pub type GTransferContext = FormalContext<RelationOrbitLabel, RelationOrbitLabel>;

/// A transfer system on a G-lattice stored as a bitvector of relation orbits.
///
/// Identity relations are implicit.  Each set bit corresponds to one
/// non-identity relation orbit in the ambient [`GTransferUniverse`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawGTransferSystem {
    /// Bitmask of the non-identity relation orbits in the transfer system.
    orbit_arrows: BitVec,
}

/// A finite lattice equipped with an action of a finite GAP group.
///
/// The action of `G` on lattice elements is required to be by lattice
/// automorphisms.  The induced action on all relations of the lattice is
/// precomputed, together with relation orbits and GAP stabilizer/transporter
/// data for those orbits.
pub struct GLattice<A> {
    action_coordinates: Arc<()>,
    lattice: Arc<Lattice<A>>,
    group: GapValue,
    element_action_homomorphism: GapValue,
    element_image_group: GapValue,
    relation_action_homomorphism: GapValue,
    relation_image_group: GapValue,
    element_generator_permutations: Vec<Vec<ElementId>>,
    relation_generator_permutations: Vec<Vec<usize>>,
    relations: Vec<Edge>,
    relation_ids: Vec<Vec<Option<usize>>>,
    relation_to_orbit: Vec<usize>,
    relation_orbits: Vec<RelationOrbit>,
}

/// One orbit of the `G`-action on lattice relations.
///
/// Relation orbits include identity relations.  The canonical representative is
/// the relation with smallest deterministic relation id in the orbit.
pub struct RelationOrbit {
    canonical_relation_id: usize,
    canonical_representative: Edge,
    relation_ids: Vec<usize>,
    relations: Vec<Edge>,
    stabilizer: GapValue,
    transporters: Vec<RelationTransporter>,
}

/// A chosen group element carrying an orbit representative to a relation.
///
/// For each relation in a [`RelationOrbit`], one transporter is stored.  The
/// stored GAP element sends the canonical representative of that orbit to the
/// transporter's relation under the induced relation action.
pub struct RelationTransporter {
    relation_id: usize,
    relation: Edge,
    group_element: GapValue,
}

/// A compact label for a relation orbit in a G-transfer-system context.
///
/// Labels are ordered by orbit id and remember the canonical representative so
/// generated diagrams and diagnostics can refer back to an actual relation
/// `x <= y` in the underlying lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationOrbitLabel {
    orbit_id: usize,
    canonical_relation_id: usize,
    canonical_representative: Edge,
}

/// The subgroup lattice of a finite GAP group with the conjugation action.
///
/// This wrapper keeps both the Rust [`GLattice`] and the GAP objects used to
/// construct it alive.
pub struct SubgroupGLattice {
    g_lattice: GLattice<GapSubgroup>,
    gap_lattice: GapValue,
    conjugacy_classes: GapValue,
    subgroup_list: GapValue,
    subgroups: Vec<GapValue>,
    subgroup_structure_descriptions: Vec<String>,
    subgroup_structure_descriptions_tex: Vec<String>,
}

/// Shared ambient data that gives raw G-transfer-system bitsets their meaning.
///
/// A universe fixes the underlying lattice, the ordered list of non-identity
/// relation orbits, and the formal context whose concepts enumerate
/// G-transfer systems.
///
/// It also retains an ordinary [`TransferUniverse`] for the underlying
/// lattice.  Expanding a G-transfer system walks its selected relation orbits
/// to realize the identification of G-transfer systems with G-fixed ordinary
/// transfer systems.
#[derive(Debug)]
pub struct GTransferUniverse<A> {
    action_coordinates: Arc<()>,
    underlying_transfer_universe: Arc<TransferUniverse<A>>,
    context: GTransferContext,
    relation_orbits: Vec<Vec<Edge>>,
    relation_to_orbit_label: Vec<Vec<Option<usize>>>,
}

/// An owned transfer system on a G-lattice together with its ambient data.
///
/// This is the user-facing form of a G-transfer system.  The raw bitvector
/// records selected relation orbits, while the universe interprets those bits
/// as actual orbits in the underlying lattice.
#[derive(Debug)]
pub struct GTransferSystem<A> {
    raw: RawGTransferSystem,
    universe: Arc<GTransferUniverse<A>>,
}

/// A lattice of transfer systems on a fixed G-lattice, ordered by containment.
///
/// The order is inclusion of selected non-identity relation orbits.
#[derive(Debug, Clone)]
pub struct GTransferLattice<A> {
    universe: Arc<GTransferUniverse<A>>,
    lattice: Lattice<RawGTransferSystem>,
}

/// Errors that can occur while constructing or using a G-lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GLatticeError {
    /// An error reported by the GAP interface.
    Gap(String),
    /// Construction of an underlying poset failed.
    Poset(PosetError),
    /// Construction of an underlying lattice failed.
    Lattice(LatticeError),
    /// The supplied GAP group is not finite.
    GroupIsNotFinite,
    /// The supplied GAP object is not a group homomorphism.
    NotAGroupHomomorphism,
    /// The number of supplied generator images does not match GAP's generators.
    GeneratorCountMismatch {
        /// The number of generators of the GAP group.
        expected: usize,
        /// The number of image permutations supplied.
        actual: usize,
    },
    /// A generator image has the wrong degree.
    WrongPermutationLength {
        /// The zero-based index of the generator.
        generator: usize,
        /// The required permutation length.
        expected: usize,
        /// The supplied permutation length.
        actual: usize,
    },
    /// A generator image sends an element outside the lattice.
    PermutationImageOutOfBounds {
        /// The zero-based index of the generator.
        generator: usize,
        /// The source element.
        element: ElementId,
        /// The invalid target element.
        image: ElementId,
        /// The number of lattice elements.
        len: usize,
    },
    /// A generator image is not injective.
    DuplicatePermutationImage {
        /// The zero-based index of the generator.
        generator: usize,
        /// The first source element with this image.
        first_element: ElementId,
        /// The second source element with this image.
        second_element: ElementId,
        /// The repeated target element.
        image: ElementId,
    },
    /// A generator image is a permutation but not a lattice automorphism.
    NotALatticeAutomorphism {
        /// The zero-based index of the generator.
        generator: usize,
        /// The lattice-map validation error.
        source: LatticeMapError,
    },
    /// GAP rejected the proposed generator images as a group homomorphism.
    HomomorphismByImagesFailed,
    /// The induced image of a relation was not a relation in the lattice.
    RelationImageMissing {
        /// The zero-based index of the generator.
        generator: usize,
        /// The source relation.
        relation: Edge,
        /// The attempted image relation.
        image: Edge,
    },
    /// GAP could not find a transporter in the relation image group.
    MissingTransporter {
        /// The canonical representative of the orbit.
        canonical: Edge,
        /// The target relation in the same orbit.
        target: Edge,
    },
    /// GAP could not lift a relation-image transporter back to `G`.
    MissingPreimage {
        /// The canonical representative of the orbit.
        canonical: Edge,
        /// The target relation in the same orbit.
        target: Edge,
    },
    /// GAP could not locate a conjugate subgroup in the stored subgroup list.
    SubgroupConjugateNotFound {
        /// The zero-based index of the generator.
        generator: usize,
        /// The subgroup whose conjugate was not found.
        subgroup: GapSubgroup,
    },
}

/// Errors that can occur while constructing an individual G-transfer system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GTransferSystemError {
    /// A generating relation references an element outside the lattice.
    EdgeOutOfBounds {
        /// The invalid generating relation.
        edge: Edge,
        /// The number of elements in the lattice.
        lattice_size: usize,
    },
    /// A non-identity generating relation is not present in the lattice order.
    NotLatticeRelation {
        /// The invalid generating relation.
        edge: Edge,
    },
    /// A raw bitvector does not have one bit for every proper relation orbit.
    WrongOrbitCount {
        /// The number of non-identity relation orbits in the universe.
        expected: usize,
        /// The number of bits in the raw representation.
        actual: usize,
    },
    /// A raw bitvector is not closed under the G-transfer-system axioms.
    RawNotClosed,
}

struct GLatticeParts<A> {
    lattice: Arc<Lattice<A>>,
    group: GapValue,
    element_action_homomorphism: GapValue,
    element_image_group: GapValue,
    relation_action: GapAction,
    element_generator_permutations: Vec<Vec<ElementId>>,
    relation_generator_permutations: Vec<Vec<usize>>,
    relations: Vec<Edge>,
    relation_ids: Vec<Vec<Option<usize>>>,
}

impl<A: fmt::Debug> fmt::Debug for GLattice<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLattice")
            .field("lattice", &self.lattice)
            .field(
                "element_generator_permutations",
                &self.element_generator_permutations,
            )
            .field(
                "relation_generator_permutations",
                &self.relation_generator_permutations,
            )
            .field("relations", &self.relations)
            .field("relation_orbits", &self.relation_orbits)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for RelationOrbit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationOrbit")
            .field("canonical_relation_id", &self.canonical_relation_id)
            .field("canonical_representative", &self.canonical_representative)
            .field("relation_ids", &self.relation_ids)
            .field("relations", &self.relations)
            .field("transporters", &self.transporters)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for RelationTransporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationTransporter")
            .field("relation_id", &self.relation_id)
            .field("relation", &self.relation)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for SubgroupGLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubgroupGLattice")
            .field("g_lattice", &self.g_lattice)
            .field("subgroup_count", &self.subgroups.len())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for GLatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GLatticeError::Gap(error) => write!(f, "{error}"),
            GLatticeError::Poset(error) => write!(f, "{error}"),
            GLatticeError::Lattice(error) => write!(f, "{error}"),
            GLatticeError::GroupIsNotFinite => write!(f, "G must be finite"),
            GLatticeError::NotAGroupHomomorphism => {
                write!(f, "GAP object is not a group homomorphism")
            }
            GLatticeError::GeneratorCountMismatch { expected, actual } => {
                write!(f, "received {actual} generator images, expected {expected}")
            }
            GLatticeError::WrongPermutationLength {
                generator,
                expected,
                actual,
            } => write!(
                f,
                "generator image {generator} has length {actual}, expected {expected}"
            ),
            GLatticeError::PermutationImageOutOfBounds {
                generator,
                element,
                image,
                len,
            } => write!(
                f,
                "generator image {generator} sends element {element} to {image}, out of bounds for {len} elements"
            ),
            GLatticeError::DuplicatePermutationImage {
                generator,
                first_element,
                second_element,
                image,
            } => write!(
                f,
                "generator image {generator} sends both {first_element} and {second_element} to {image}"
            ),
            GLatticeError::NotALatticeAutomorphism { generator, source } => {
                write!(
                    f,
                    "generator image {generator} is not a lattice automorphism: {source}"
                )
            }
            GLatticeError::HomomorphismByImagesFailed => {
                write!(
                    f,
                    "GAP could not construct a homomorphism from the generator images"
                )
            }
            GLatticeError::RelationImageMissing {
                generator,
                relation,
                image,
            } => write!(
                f,
                "generator image {generator} sends relation {} <= {} to non-relation {} <= {}",
                relation.from, relation.to, image.from, image.to
            ),
            GLatticeError::MissingTransporter { canonical, target } => write!(
                f,
                "GAP found no transporter from {} <= {} to {} <= {}",
                canonical.from, canonical.to, target.from, target.to
            ),
            GLatticeError::MissingPreimage { canonical, target } => write!(
                f,
                "GAP found no preimage for a transporter from {} <= {} to {} <= {}",
                canonical.from, canonical.to, target.from, target.to
            ),
            GLatticeError::SubgroupConjugateNotFound {
                generator,
                subgroup,
            } => write!(
                f,
                "GAP did not find the conjugate of subgroup class {}, element {} under generator {generator}",
                subgroup.conjugacy_class(),
                subgroup.class_element()
            ),
        }
    }
}

impl std::error::Error for GLatticeError {}

impl fmt::Display for GTransferSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GTransferSystemError::EdgeOutOfBounds { edge, lattice_size } => write!(
                f,
                "generator {} <= {} is out of bounds for a lattice with {lattice_size} elements",
                edge.from, edge.to
            ),
            GTransferSystemError::NotLatticeRelation { edge } => write!(
                f,
                "generator {} <= {} is not a relation in the lattice order",
                edge.from, edge.to
            ),
            GTransferSystemError::WrongOrbitCount { expected, actual } => write!(
                f,
                "raw G-transfer system has {actual} relation-orbit bits, expected {expected}"
            ),
            GTransferSystemError::RawNotClosed => {
                write!(
                    f,
                    "raw relation-orbit set is not closed under the G-transfer-system axioms"
                )
            }
        }
    }
}

impl std::error::Error for GTransferSystemError {}

impl From<PosetError> for GLatticeError {
    fn from(error: PosetError) -> Self {
        Self::Poset(error)
    }
}

impl From<LatticeError> for GLatticeError {
    fn from(error: LatticeError) -> Self {
        Self::Lattice(error)
    }
}

impl From<GroupTheoryError> for GLatticeError {
    fn from(error: GroupTheoryError) -> Self {
        match error {
            GroupTheoryError::Gap(error) => Self::Gap(error),
            GroupTheoryError::GroupIsNotFinite => Self::GroupIsNotFinite,
            GroupTheoryError::NotAGroupHomomorphism => Self::NotAGroupHomomorphism,
            GroupTheoryError::HomomorphismSourceMismatch => {
                Self::Gap("group homomorphism has the wrong source".to_owned())
            }
            GroupTheoryError::HomomorphismRangeMismatch => {
                Self::Gap("group homomorphism has the wrong range".to_owned())
            }
            GroupTheoryError::SubgroupImageNotFound { subgroup } => Self::Gap(format!(
                "GAP did not find the image of subgroup {subgroup} in the codomain subgroup list"
            )),
            GroupTheoryError::SubgroupPreimageNotFound { subgroup } => Self::Gap(format!(
                "GAP did not find the preimage of subgroup {subgroup} in the domain subgroup list"
            )),
            GroupTheoryError::GeneratorCountMismatch { expected, actual } => {
                Self::GeneratorCountMismatch { expected, actual }
            }
            GroupTheoryError::WrongPermutationLength {
                generator,
                expected,
                actual,
            } => Self::WrongPermutationLength {
                generator,
                expected,
                actual,
            },
            GroupTheoryError::PermutationImageOutOfBounds {
                generator,
                element,
                image,
                len,
            } => Self::PermutationImageOutOfBounds {
                generator,
                element,
                image,
                len,
            },
            GroupTheoryError::DuplicatePermutationImage {
                generator,
                first_element,
                second_element,
                image,
            } => Self::DuplicatePermutationImage {
                generator,
                first_element,
                second_element,
                image,
            },
            GroupTheoryError::HomomorphismByImagesFailed => Self::HomomorphismByImagesFailed,
            GroupTheoryError::SubgroupConjugateNotFound {
                generator,
                subgroup,
            } => Self::SubgroupConjugateNotFound {
                generator,
                subgroup,
            },
        }
    }
}

impl PartialOrd for RawGTransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        set_partial_cmp(&self.orbit_arrows, &other.orbit_arrows)
    }
}

impl RawGTransferSystem {
    pub(crate) fn new(orbit_arrows: BitVec) -> Self {
        Self { orbit_arrows }
    }

    /// Returns the bitvector of selected non-identity relation orbits.
    ///
    /// Bit `i` corresponds to `universe.relation_orbit_labels()[i]` for the
    /// ambient [`GTransferUniverse`].
    pub fn orbit_arrows(&self) -> &BitVec {
        &self.orbit_arrows
    }
}

impl RelationOrbitLabel {
    /// Constructs a relation-orbit label.
    pub fn new(
        orbit_id: usize,
        canonical_relation_id: usize,
        canonical_representative: Edge,
    ) -> Self {
        Self {
            orbit_id,
            canonical_relation_id,
            canonical_representative,
        }
    }

    /// Returns the orbit id in the parent [`GLattice`].
    pub fn orbit_id(&self) -> usize {
        self.orbit_id
    }

    /// Returns the id of the canonical representative relation.
    pub fn canonical_relation_id(&self) -> usize {
        self.canonical_relation_id
    }

    /// Returns the canonical representative relation.
    pub fn canonical_representative(&self) -> Edge {
        self.canonical_representative
    }
}

impl<A> GLattice<A> {
    /// Constructs a G-lattice from a GAP homomorphism to a permutation group.
    ///
    /// The homomorphism is interpreted as an action of `group` on the element
    /// ids `[1..n]` of the lattice, using GAP's one-based point convention.
    /// Each generator image is extracted, converted to zero-based Rust element
    /// ids, and validated as a lattice automorphism.  The induced action on
    /// lattice relations is then built automatically.
    pub fn from_gap_homomorphism(
        lattice: Arc<Lattice<A>>,
        group: &GapValue,
        homomorphism: &GapValue,
    ) -> Result<Self, GLatticeError> {
        let mut gap = group_theory::global_gap()?;
        group_theory::validate_finite_group(&mut gap, group)?;
        group_theory::validate_group_homomorphism(&mut gap, homomorphism)?;

        let group = group.clone();
        let element_action_homomorphism = homomorphism.clone();
        let element_image_group =
            group_theory::homomorphism_image(&mut gap, &element_action_homomorphism)?;
        let source_generators = group_theory::generators_of_group(&mut gap, &group)?;
        let element_generator_permutations =
            group_theory::generator_permutations_from_homomorphism(
                &mut gap,
                &source_generators,
                homomorphism,
                lattice.size(),
            )?;
        for (generator, permutation) in element_generator_permutations.iter().enumerate() {
            validate_lattice_automorphism(generator, &lattice, permutation)?;
        }
        let relations = lattice.as_poset().all_relations_iter().collect::<Vec<_>>();
        let relation_ids = relation_id_matrix(lattice.size(), &relations);
        let relation_generator_permutations = relation_generator_permutations(
            &relations,
            &relation_ids,
            &element_generator_permutations,
        )?;
        let relation_action = group_theory::action_from_generator_permutations(
            &mut gap,
            &group,
            &source_generators,
            &relation_generator_permutations,
            relations.len(),
        )?;

        Self::from_parts(
            &mut gap,
            GLatticeParts {
                lattice,
                group,
                element_action_homomorphism,
                element_image_group,
                relation_action,
                element_generator_permutations,
                relation_generator_permutations,
                relations,
                relation_ids,
            },
        )
    }

    /// Constructs a G-lattice from Rust permutations for GAP generators.
    ///
    /// The `generator_images` vector must list one zero-based permutation of
    /// lattice element ids for each generator returned by GAP's
    /// `GeneratorsOfGroup(group)`.  The constructor validates that each
    /// permutation is a lattice automorphism and that GAP accepts these images
    /// as defining a homomorphism from `group`.
    pub fn from_generator_images(
        lattice: Arc<Lattice<A>>,
        group: &GapValue,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        let mut gap = group_theory::global_gap()?;
        Self::from_generator_images_with_gap(&mut gap, lattice, group, generator_images)
    }

    fn from_generator_images_with_gap(
        gap: &mut Gap,
        lattice: Arc<Lattice<A>>,
        group: &GapValue,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        group_theory::validate_finite_group(gap, group)?;
        let group = group.clone();
        let source_generators = group_theory::generators_of_group(gap, &group)?;
        group_theory::validate_generator_count(gap, &source_generators, generator_images.len())?;

        for (generator, image) in generator_images.iter().enumerate() {
            validate_lattice_automorphism(generator, &lattice, image)?;
        }

        let element_action = group_theory::action_from_generator_permutations(
            gap,
            &group,
            &source_generators,
            &generator_images,
            lattice.size(),
        )?;
        let relations = lattice.as_poset().all_relations_iter().collect::<Vec<_>>();
        let relation_ids = relation_id_matrix(lattice.size(), &relations);
        let relation_generator_permutations =
            relation_generator_permutations(&relations, &relation_ids, &generator_images)?;
        let relation_action = group_theory::action_from_generator_permutations(
            gap,
            &group,
            &source_generators,
            &relation_generator_permutations,
            relations.len(),
        )?;

        Self::from_parts(
            gap,
            GLatticeParts {
                lattice,
                group,
                element_action_homomorphism: element_action.homomorphism,
                element_image_group: element_action.image_group,
                relation_action,
                element_generator_permutations: generator_images,
                relation_generator_permutations,
                relations,
                relation_ids,
            },
        )
    }

    /// Returns the underlying lattice.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.lattice
    }

    /// Returns the opaque identity of this concrete group-action presentation.
    pub(crate) fn action_coordinates(&self) -> &Arc<()> {
        &self.action_coordinates
    }

    /// Returns the GAP group acting on the lattice.
    pub fn group(&self) -> &GapValue {
        &self.group
    }

    /// Returns the GAP homomorphism describing the action on lattice elements.
    pub fn element_action_homomorphism(&self) -> &GapValue {
        &self.element_action_homomorphism
    }

    /// Returns the image group of the element action.
    pub fn element_image_group(&self) -> &GapValue {
        &self.element_image_group
    }

    /// Returns the induced GAP homomorphism on lattice relations.
    pub fn relation_action_homomorphism(&self) -> &GapValue {
        &self.relation_action_homomorphism
    }

    /// Returns the image group of the induced relation action.
    pub fn relation_image_group(&self) -> &GapValue {
        &self.relation_image_group
    }

    /// Returns the validated generator permutations on lattice elements.
    ///
    /// Each inner vector is a zero-based permutation of element ids.
    pub fn element_generator_permutations(&self) -> &[Vec<ElementId>] {
        &self.element_generator_permutations
    }

    /// Returns the induced generator permutations on relation ids.
    ///
    /// Each inner vector is a zero-based permutation of the entries returned by
    /// [`GLattice::relations`].
    pub fn relation_generator_permutations(&self) -> &[Vec<usize>] {
        &self.relation_generator_permutations
    }

    /// Returns all lattice relations, including identities.
    ///
    /// Relations are stored in deterministic row-major order inherited from the
    /// underlying poset.
    pub fn relations(&self) -> &[Edge] {
        &self.relations
    }

    /// Returns a relation by relation id.
    pub fn relation(&self, relation_id: usize) -> Option<Edge> {
        self.relations.get(relation_id).copied()
    }

    /// Returns the relation id of `relation`, if it is a relation in the lattice.
    pub fn relation_id(&self, relation: Edge) -> Option<usize> {
        self.relation_ids
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }

    /// Returns the precomputed orbits of the action on all relations.
    pub fn relation_orbits(&self) -> &[RelationOrbit] {
        &self.relation_orbits
    }

    /// Returns the orbit containing the relation with the given relation id.
    pub fn relation_orbit_by_id(&self, relation_id: usize) -> Option<&RelationOrbit> {
        self.relation_to_orbit
            .get(relation_id)
            .and_then(|&orbit| self.relation_orbits.get(orbit))
    }

    /// Returns the orbit containing a relation.
    pub fn relation_orbit(&self, relation: Edge) -> Option<&RelationOrbit> {
        self.relation_id(relation)
            .and_then(|relation_id| self.relation_orbit_by_id(relation_id))
    }

    /// Returns labels for the non-identity relation orbits.
    ///
    /// These labels are the objects and attributes of the G-transfer-system
    /// formal context.
    pub fn non_identity_relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.relation_orbits
            .iter()
            .enumerate()
            .filter(|(_, orbit)| !orbit.canonical_representative().is_identity())
            .map(|(orbit_id, orbit)| {
                RelationOrbitLabel::new(
                    orbit_id,
                    orbit.canonical_relation_id(),
                    orbit.canonical_representative(),
                )
            })
            .collect()
    }

    /// Constructs the formal context whose concepts are G-transfer systems.
    ///
    /// Objects and attributes are non-identity relation orbits.  For an object
    /// orbit represented by `r` and an attribute orbit `O`, incidence holds
    /// when the ordinary transfer-system lifting relation holds between `r`
    /// and every relation in `O`.
    pub fn transfer_context(&self) -> GTransferContext {
        let labels = self.non_identity_relation_orbit_labels();
        let matrix = labels
            .iter()
            .map(|object| {
                labels
                    .iter()
                    .map(|attribute| {
                        self.relation_orbits[attribute.orbit_id()]
                            .relations()
                            .iter()
                            .all(|&relation| {
                                transfer_context_relation(
                                    self.lattice.as_ref(),
                                    relation,
                                    object.canonical_representative(),
                                )
                            })
                    })
                    .collect()
            })
            .collect();
        FormalContext::new(labels.clone(), labels, matrix)
    }

    /// Builds the shared universe used to enumerate transfer systems on this G-lattice.
    pub fn transfer_universe(&self) -> Arc<GTransferUniverse<A>> {
        Arc::new(GTransferUniverse::new(self))
    }

    /// Constructs the lattice of G-transfer systems ordered by containment.
    pub fn transfer_systems_containment(&self) -> Result<GTransferLattice<A>, GLatticeError> {
        self.transfer_universe().containment_lattice()
    }

    fn from_parts(gap: &mut Gap, parts: GLatticeParts<A>) -> Result<Self, GLatticeError> {
        let point_orbits = group_theory::point_orbits(
            gap,
            parts.relations.len(),
            &parts.relation_generator_permutations,
            &parts.relation_action.image_group,
            &parts.relation_action.homomorphism,
        )
        .map_err(|error| relation_orbit_error(error, &parts.relations))?;
        let relation_to_orbit = point_orbits.point_to_orbit;
        let relation_orbits = point_orbits
            .orbits
            .into_iter()
            .map(|orbit| {
                let canonical_relation_id = orbit.canonical_point;
                let canonical_representative = parts.relations[canonical_relation_id];
                let relations = orbit
                    .points
                    .iter()
                    .map(|&relation_id| parts.relations[relation_id])
                    .collect();
                let transporters = orbit
                    .transporters
                    .into_iter()
                    .map(|transporter| RelationTransporter {
                        relation_id: transporter.point,
                        relation: parts.relations[transporter.point],
                        group_element: transporter.group_element,
                    })
                    .collect();
                RelationOrbit {
                    canonical_relation_id,
                    canonical_representative,
                    relation_ids: orbit.points,
                    relations,
                    stabilizer: orbit.stabilizer,
                    transporters,
                }
            })
            .collect();

        Ok(Self {
            action_coordinates: Arc::new(()),
            lattice: parts.lattice,
            group: parts.group,
            element_action_homomorphism: parts.element_action_homomorphism,
            element_image_group: parts.element_image_group,
            relation_action_homomorphism: parts.relation_action.homomorphism,
            relation_image_group: parts.relation_action.image_group,
            element_generator_permutations: parts.element_generator_permutations,
            relation_generator_permutations: parts.relation_generator_permutations,
            relations: parts.relations,
            relation_ids: parts.relation_ids,
            relation_to_orbit,
            relation_orbits,
        })
    }
}

impl GLattice<GapSubgroup> {
    /// Constructs the subgroup lattice of a finite GAP group with conjugation action.
    ///
    /// The underlying lattice has one element for each subgroup in GAP's
    /// `LatticeSubgroups(group)`, ordered by inclusion.  The group acts by
    /// conjugating subgroups.
    pub fn from_subgroup_lattice(group: &GapValue) -> Result<SubgroupGLattice, GLatticeError> {
        SubgroupGLattice::new(group)
    }
}

impl SubgroupGLattice {
    /// Constructs the subgroup lattice of a finite GAP group with conjugation action.
    pub fn new(group: &GapValue) -> Result<Self, GLatticeError> {
        let mut gap = group_theory::global_gap()?;
        let data = group_theory::subgroup_lattice_data(&mut gap, group)?;
        let subgroup_structure_descriptions_tex = data
            .structure_descriptions
            .iter()
            .map(|description| structure_description_to_tex(description))
            .collect();
        let lattice = Arc::new(Lattice::new(Poset::from_relation(
            data.labels.clone(),
            data.inclusion_relation,
        )?)?);
        let g_lattice = GLattice::from_generator_images_with_gap(
            &mut gap,
            lattice,
            group,
            data.conjugation_generator_images,
        )?;

        Ok(Self {
            g_lattice,
            gap_lattice: data.gap_lattice,
            conjugacy_classes: data.conjugacy_classes,
            subgroup_list: data.subgroup_list,
            subgroups: data.subgroups,
            subgroup_structure_descriptions: data.structure_descriptions,
            subgroup_structure_descriptions_tex,
        })
    }

    /// Returns the G-lattice of subgroups.
    pub fn g_lattice(&self) -> &GLattice<GapSubgroup> {
        &self.g_lattice
    }

    /// Returns the underlying subgroup lattice.
    pub fn lattice(&self) -> &Arc<Lattice<GapSubgroup>> {
        self.g_lattice.lattice()
    }

    /// Returns GAP's `LatticeSubgroups(group)` object.
    pub fn gap_lattice(&self) -> &GapValue {
        &self.gap_lattice
    }

    /// Returns GAP's conjugacy classes of subgroups for the subgroup lattice.
    pub fn conjugacy_classes(&self) -> &GapValue {
        &self.conjugacy_classes
    }

    /// Returns the rooted GAP list of subgroup objects in lattice element order.
    pub fn subgroup_list(&self) -> &GapValue {
        &self.subgroup_list
    }

    /// Returns the rooted GAP subgroup objects in lattice element order.
    pub fn subgroups(&self) -> &[GapValue] {
        &self.subgroups
    }

    /// Returns a GAP subgroup object by lattice element id.
    pub fn subgroup(&self, id: ElementId) -> Option<&GapValue> {
        self.subgroups.get(id)
    }

    /// Returns GAP's `StructureDescription` strings in lattice element order.
    ///
    /// The strings are computed once while constructing this object, so they
    /// can later be used as labels in transfer-system diagrams without making
    /// further GAP calls. Conjugate but distinct subgroups may have the same
    /// structure description.
    pub fn subgroup_structure_descriptions(&self) -> &[String] {
        &self.subgroup_structure_descriptions
    }

    /// Returns GAP's `StructureDescription` string for one subgroup.
    pub fn subgroup_structure_description(&self, id: ElementId) -> Option<&str> {
        self.subgroup_structure_descriptions
            .get(id)
            .map(String::as_str)
    }

    /// Returns TeX math-mode labels for GAP's structure descriptions.
    ///
    /// These labels are cached in lattice element order and are suitable for
    /// [`crate::tikz::GlyphNodeDisplay::raw`]. For example, GAP's `"C2"`
    /// becomes `"$C_{2}$"`, and `"C2 x C3"` becomes
    /// `"$C_{2} \\times C_{3}$"`.
    pub fn subgroup_structure_descriptions_tex(&self) -> &[String] {
        &self.subgroup_structure_descriptions_tex
    }

    /// Returns the TeX math-mode structure-description label for one subgroup.
    pub fn subgroup_structure_description_tex(&self, id: ElementId) -> Option<&str> {
        self.subgroup_structure_descriptions_tex
            .get(id)
            .map(String::as_str)
    }

    /// Constructs the containment lattice of transfer systems on this subgroup
    /// G-lattice.
    ///
    /// This is a convenience forwarding method for the common workflow of
    /// constructing a finite GAP group, taking its subgroup lattice with
    /// conjugation action, and then studying its G-transfer systems.
    ///
    /// ```no_run
    /// use hccr::g_lattice::SubgroupGLattice;
    /// use hccr::tikz::{GlyphNodeDisplay, ToTikz, TransferSystemTikzOptions};
    ///
    /// let group = gap_sys::eval("SymmetricGroup(3);")?;
    /// let subgroup_lattice = SubgroupGLattice::new(&group)?;
    /// let systems = subgroup_lattice.transfer_systems_containment()?;
    /// let mut options = TransferSystemTikzOptions::default();
    /// options.glyph.node_display =
    ///     GlyphNodeDisplay::raw(subgroup_lattice.subgroup_structure_descriptions_tex());
    /// let picture = systems.to_tikz_with(&options);
    /// # let _ = picture;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transfer_systems_containment(
        &self,
    ) -> Result<GTransferLattice<GapSubgroup>, GLatticeError> {
        self.g_lattice.transfer_systems_containment()
    }
}

impl<A> GTransferUniverse<A> {
    /// Constructs the transfer-system universe for a G-lattice.
    pub fn new(g_lattice: &GLattice<A>) -> Self {
        let context = g_lattice.transfer_context();
        let underlying_transfer_universe =
            Arc::new(TransferUniverse::new(Arc::clone(g_lattice.lattice())));
        let relation_orbits = context
            .objects
            .iter()
            .map(|label| {
                g_lattice.relation_orbits[label.orbit_id()]
                    .relations()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        let mut relation_to_orbit_label =
            vec![vec![None; g_lattice.lattice().size()]; g_lattice.lattice().size()];
        for (orbit_label_id, relations) in relation_orbits.iter().enumerate() {
            for &relation in relations {
                relation_to_orbit_label[relation.from][relation.to] = Some(orbit_label_id);
            }
        }

        Self {
            action_coordinates: Arc::clone(g_lattice.action_coordinates()),
            underlying_transfer_universe,
            context,
            relation_orbits,
            relation_to_orbit_label,
        }
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.underlying_transfer_universe.lattice()
    }

    /// Returns the opaque identity of the group action defining these orbit coordinates.
    pub(crate) fn action_coordinates(&self) -> &Arc<()> {
        &self.action_coordinates
    }

    /// Returns the ordinary transfer-system universe for the underlying lattice.
    ///
    /// This universe shares the underlying lattice with the G-transfer
    /// universe and is ready for ordinary formal-concept calculations.
    pub fn underlying_transfer_universe(&self) -> &Arc<TransferUniverse<A>> {
        &self.underlying_transfer_universe
    }

    /// Returns the formal context whose concepts are G-transfer systems.
    pub fn context(&self) -> &GTransferContext {
        &self.context
    }

    /// Returns the non-identity relation-orbit labels used as generators.
    pub fn relation_orbit_labels(&self) -> &[RelationOrbitLabel] {
        &self.context.objects
    }

    /// Returns all underlying lattice relations in a non-identity orbit.
    pub fn relation_orbit_relations(&self, orbit_label_id: usize) -> Option<&[Edge]> {
        self.relation_orbits.get(orbit_label_id).map(Vec::as_slice)
    }

    pub(crate) fn relation_orbit_label_id(&self, relation: Edge) -> Option<usize> {
        self.relation_to_orbit_label
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }

    /// Constructs the G-transfer system generated by the supplied relations.
    ///
    /// Identity relations may be supplied but need not be. A proper relation
    /// selects its entire G-orbit, and the returned system is then closed under
    /// all G-transfer-system axioms. Every non-identity generator must be a
    /// relation in the underlying lattice order.
    pub fn generated_by<I, E>(
        self: &Arc<Self>,
        generators: I,
    ) -> Result<GTransferSystem<A>, GTransferSystemError>
    where
        I: IntoIterator<Item = E>,
        E: Into<Edge>,
    {
        let lattice_size = self.lattice().size();
        let mut orbit_arrows = BitVec::repeat(false, self.relation_orbit_labels().len());

        for generator in generators {
            let edge = generator.into();
            if edge.from >= lattice_size || edge.to >= lattice_size {
                return Err(GTransferSystemError::EdgeOutOfBounds { edge, lattice_size });
            }
            if edge.is_identity() {
                continue;
            }

            let Some(orbit_label_id) = self.relation_orbit_label_id(edge) else {
                return Err(GTransferSystemError::NotLatticeRelation { edge });
            };
            orbit_arrows.set(orbit_label_id, true);
        }

        let raw = RawGTransferSystem::new(self.close_orbit_arrows(&orbit_arrows));
        Ok(GTransferSystem::new(raw, Arc::clone(self)))
    }

    /// Validates raw relation-orbit data and pairs it with this universe.
    ///
    /// The bitvector must have one bit for each non-identity relation orbit and
    /// must already be closed under the G-transfer-system axioms. Use
    /// [`GTransferUniverse::generated_by`] when closure should be added.
    pub fn try_from_raw(
        self: &Arc<Self>,
        raw: RawGTransferSystem,
    ) -> Result<GTransferSystem<A>, GTransferSystemError> {
        let expected = self.relation_orbit_labels().len();
        let actual = raw.orbit_arrows().len();
        if actual != expected {
            return Err(GTransferSystemError::WrongOrbitCount { expected, actual });
        }
        if self.close_orbit_arrows(raw.orbit_arrows()) != *raw.orbit_arrows() {
            return Err(GTransferSystemError::RawNotClosed);
        }

        Ok(GTransferSystem::new(raw, Arc::clone(self)))
    }

    /// Closes a correctly sized orbit bitvector under the G-transfer-system axioms.
    pub(crate) fn close_orbit_arrows(&self, orbit_arrows: &BitVec) -> BitVec {
        debug_assert_eq!(orbit_arrows.len(), self.relation_orbit_labels().len());
        self.context.induce_l(&self.context.induce_r(orbit_arrows))
    }

    fn expanded_raw_transfer_system(&self, raw: &RawGTransferSystem) -> RawTransferSystem {
        let mut arrows = BitVec::repeat(
            false,
            self.underlying_transfer_universe.proper_edges().len(),
        );
        for orbit_label_id in raw.orbit_arrows().iter_ones() {
            for &relation in &self.relation_orbits[orbit_label_id] {
                let edge_id = self
                    .underlying_transfer_universe
                    .relation_index()
                    .proper_edge_id(relation)
                    .expect("non-identity relation orbit should contain only proper edges");
                arrows.set(edge_id, true);
            }
        }
        RawTransferSystem::new(arrows)
    }

    /// Enumerates all G-transfer systems in this universe.
    pub fn transfer_systems(self: &Arc<Self>) -> Vec<GTransferSystem<A>> {
        all_g_transfer_systems(self)
            .into_iter()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    /// Constructs the containment lattice of G-transfer systems.
    pub fn containment_lattice(self: &Arc<Self>) -> Result<GTransferLattice<A>, GLatticeError> {
        Ok(g_containment_lattice(
            Arc::clone(self),
            all_g_transfer_systems(self),
        )?)
    }
}

impl<A> GTransferSystem<A> {
    /// Pairs raw G-transfer-system data with its ambient universe.
    pub(crate) fn new(raw: RawGTransferSystem, universe: Arc<GTransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    /// Returns the raw bitvector representation.
    pub fn raw(&self) -> &RawGTransferSystem {
        &self.raw
    }

    /// Returns the ambient universe.
    pub fn universe(&self) -> &Arc<GTransferUniverse<A>> {
        &self.universe
    }

    /// Returns the underlying lattice, forgetting the group action.
    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    /// Returns whether a relation belongs to this G-transfer system.
    ///
    /// Every in-range identity relation is present.  A proper relation belongs
    /// exactly when its entire G-orbit is one of the selected orbit generators.
    /// Non-relations and out-of-range edges return `false`.
    pub fn contains_relation(&self, relation: Edge) -> bool {
        if relation.is_identity() {
            return relation.from < self.lattice().size();
        }

        self.universe
            .relation_orbit_label_id(relation)
            .is_some_and(|orbit_label_id| self.raw.orbit_arrows()[orbit_label_id])
    }

    /// Returns the selected non-identity relation-orbit labels.
    pub fn relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.raw
            .orbit_arrows()
            .iter_ones()
            .map(|orbit_label_id| self.universe.relation_orbit_labels()[orbit_label_id])
            .collect()
    }

    /// Returns the underlying lattice relations belonging to this G-transfer system.
    ///
    /// Each selected orbit contributes all of its relations.  If
    /// `include_identities` is true, identity relations `x <= x` are included
    /// as well.
    pub fn relations(&self, include_identities: bool) -> EdgeSet {
        let mut result = EdgeSet::new();
        if include_identities {
            for id in 0..self.lattice().size() {
                result.insert(Edge::new(id, id));
            }
        }

        for orbit_label_id in self.raw.orbit_arrows().iter_ones() {
            result.extend(
                self.universe.relation_orbits[orbit_label_id]
                    .iter()
                    .copied(),
            );
        }
        result
    }

    /// Expands this G-transfer system to an ordinary transfer system.
    ///
    /// The resulting system is on the underlying lattice and contains the
    /// union of all selected non-identity relation orbits.  It is fixed by the
    /// given G-action.  Conversely, every G-fixed ordinary transfer system is
    /// obtained uniquely in this way, so this method realizes the usual
    /// fixed-point correspondence without constructing the full ordinary
    /// transfer-system lattice.
    pub fn underlying_transfer_system(&self) -> TransferSystem<A> {
        TransferSystem::new(
            self.universe.expanded_raw_transfer_system(&self.raw),
            Arc::clone(self.universe.underlying_transfer_universe()),
        )
    }
}

impl<A> Clone for GTransferSystem<A> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            universe: Arc::clone(&self.universe),
        }
    }
}

impl<A> PartialEq for GTransferSystem<A> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw && Arc::ptr_eq(&self.universe, &other.universe)
    }
}

impl<A> Eq for GTransferSystem<A> {}

impl<A> GTransferLattice<A> {
    fn new(universe: Arc<GTransferUniverse<A>>, lattice: Lattice<RawGTransferSystem>) -> Self {
        Self { universe, lattice }
    }

    /// Returns the universe shared by all systems in this lattice.
    pub fn universe(&self) -> &Arc<GTransferUniverse<A>> {
        &self.universe
    }

    /// Returns the raw lattice whose labels are bitvector G-transfer systems.
    pub fn raw_lattice(&self) -> &Lattice<RawGTransferSystem> {
        &self.lattice
    }

    /// Returns the underlying poset of the G-transfer-system lattice.
    pub fn as_poset(&self) -> &Poset<RawGTransferSystem> {
        self.lattice.as_poset()
    }

    /// Returns the number of G-transfer systems.
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Returns the meet of two G-transfer systems by element id.
    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    /// Returns the join of two G-transfer systems by element id.
    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    /// Returns the bottom G-transfer system.
    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    /// Returns the top G-transfer system.
    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    /// Returns a G-transfer system by element id.
    pub fn system(&self, id: ElementId) -> Option<GTransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Iterates over all G-transfer systems in element-id order.
    pub fn systems(&self) -> impl Iterator<Item = GTransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    /// Relabels the raw lattice by user-facing [`GTransferSystem`] values.
    pub fn to_system_lattice(&self) -> Lattice<GTransferSystem<A>> {
        self.lattice
            .relabelled(|raw| GTransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }

    /// Relabels the underlying containment poset by user-facing
    /// [`GTransferSystem`] values.
    ///
    /// This avoids copying the lattice's meet and join tables when only its
    /// order is needed.
    pub fn to_system_poset(&self) -> Poset<GTransferSystem<A>> {
        self.as_poset()
            .relabelled(|raw| GTransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

impl RelationOrbit {
    /// Returns the relation id of the canonical representative.
    pub fn canonical_relation_id(&self) -> usize {
        self.canonical_relation_id
    }

    /// Returns the canonical representative of the orbit.
    ///
    /// The representative is the relation with smallest relation id in the
    /// orbit.
    pub fn canonical_representative(&self) -> Edge {
        self.canonical_representative
    }

    /// Returns the relation ids in this orbit.
    pub fn relation_ids(&self) -> &[usize] {
        &self.relation_ids
    }

    /// Returns the relations in this orbit.
    pub fn relations(&self) -> &[Edge] {
        &self.relations
    }

    /// Returns the stabilizer in the original GAP group of the canonical representative.
    ///
    /// This subgroup is computed as the preimage of the stabilizer in the
    /// relation image group.
    pub fn stabilizer(&self) -> &GapValue {
        &self.stabilizer
    }

    /// Returns chosen transporters from the canonical representative to each relation.
    pub fn transporters(&self) -> &[RelationTransporter] {
        &self.transporters
    }

    /// Returns the transporter targeting a relation id.
    pub fn transporter_for_relation_id(&self, relation_id: usize) -> Option<&RelationTransporter> {
        self.transporters
            .iter()
            .find(|transporter| transporter.relation_id == relation_id)
    }

    /// Returns the transporter targeting a relation.
    pub fn transporter_for_relation(&self, relation: Edge) -> Option<&RelationTransporter> {
        self.transporters
            .iter()
            .find(|transporter| transporter.relation == relation)
    }
}

impl RelationTransporter {
    /// Returns the id of the target relation.
    pub fn relation_id(&self) -> usize {
        self.relation_id
    }

    /// Returns the target relation.
    pub fn relation(&self) -> Edge {
        self.relation
    }

    /// Returns the GAP group element carrying the orbit representative to the target relation.
    pub fn group_element(&self) -> &GapValue {
        &self.group_element
    }
}

/// Converts GAP's documented `StructureDescription` notation to TeX math mode.
///
/// GAP uses an intentionally small grammar: named groups have an adjacent
/// numeric parameter, `x` denotes a direct product, `:` a semidirect product,
/// and `.` an extension. This function preserves the raw GAP description
/// while typesetting its parameters as subscripts and its operations with
/// standard mathematical symbols.
pub fn structure_description_to_tex(description: &str) -> String {
    let mut result = String::from("$");
    let mut characters = description.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            'A'..='Z' | 'a'..='z' => {
                let mut name = String::from(character);
                while characters
                    .peek()
                    .is_some_and(|next| next.is_ascii_alphabetic())
                {
                    name.push(characters.next().expect("peeked character should exist"));
                }

                if name == "x" {
                    result.push_str("\\times");
                    continue;
                }

                result.push_str(&name);
                if matches!(name.as_str(), "O")
                    && matches!(characters.peek(), Some(&'+') | Some(&'-'))
                {
                    result.push_str("^{");
                    result.push(characters.next().expect("peeked sign should exist"));
                    result.push('}');
                }

                let mut parameter = String::new();
                while characters.peek().is_some_and(|next| next.is_ascii_digit()) {
                    parameter.push(characters.next().expect("peeked digit should exist"));
                }
                if !parameter.is_empty() {
                    result.push_str("_{");
                    result.push_str(&parameter);
                    result.push('}');
                }
            }
            '0'..='9' => {
                let mut number = String::from(character);
                while characters.peek().is_some_and(|next| next.is_ascii_digit()) {
                    number.push(characters.next().expect("peeked digit should exist"));
                }
                if characters
                    .peek()
                    .is_some_and(|next| next.is_ascii_alphabetic())
                {
                    result.push_str("{}^{");
                    result.push_str(&number);
                    result.push('}');
                } else {
                    result.push_str(&number);
                }
            }
            ':' => result.push_str("\\rtimes"),
            '.' => result.push_str("\\cdot"),
            '\'' => result.push_str("^{\\prime}"),
            '^' => {
                let mut exponent = String::new();
                while characters.peek().is_some_and(|next| next.is_ascii_digit()) {
                    exponent.push(characters.next().expect("peeked digit should exist"));
                }
                result.push_str("^{");
                result.push_str(&exponent);
                result.push('}');
            }
            ' ' => result.push(' '),
            '_' => result.push_str("\\_"),
            '&' => result.push_str("\\&"),
            '%' => result.push_str("\\%"),
            '#' => result.push_str("\\#"),
            '$' => result.push_str("\\$"),
            '{' => result.push_str("\\{"),
            '}' => result.push_str("\\}"),
            '\\' => result.push_str("\\textbackslash{}"),
            other => result.push(other),
        }
    }

    result.push('$');
    result
}

fn validate_lattice_automorphism<A>(
    generator: usize,
    lattice: &Arc<Lattice<A>>,
    image: &[ElementId],
) -> Result<(), GLatticeError> {
    group_theory::validate_permutation(generator, lattice.size(), image)?;
    crate::morphism::LatticeMap::new(Arc::clone(lattice), Arc::clone(lattice), image.to_vec())
        .map_err(|source| GLatticeError::NotALatticeAutomorphism { generator, source })?;
    Ok(())
}

fn relation_id_matrix(len: usize, relations: &[Edge]) -> Vec<Vec<Option<usize>>> {
    let mut result = vec![vec![None; len]; len];
    for (relation_id, relation) in relations.iter().copied().enumerate() {
        result[relation.from][relation.to] = Some(relation_id);
    }
    result
}

fn relation_generator_permutations(
    relations: &[Edge],
    relation_ids: &[Vec<Option<usize>>],
    element_generator_permutations: &[Vec<ElementId>],
) -> Result<Vec<Vec<usize>>, GLatticeError> {
    element_generator_permutations
        .iter()
        .enumerate()
        .map(|(generator, permutation)| {
            relations
                .iter()
                .map(|relation| {
                    let image = Edge::new(permutation[relation.from], permutation[relation.to]);
                    relation_ids[image.from][image.to].ok_or(GLatticeError::RelationImageMissing {
                        generator,
                        relation: *relation,
                        image,
                    })
                })
                .collect()
        })
        .collect()
}

fn transfer_context_relation<A>(
    lattice: &Lattice<A>,
    attribute_relation: Edge,
    object_relation: Edge,
) -> bool {
    lattice.leq(attribute_relation.to, object_relation.from)
        || !lattice.leq(attribute_relation.to, object_relation.to)
        || !lattice.leq(attribute_relation.from, object_relation.from)
}

fn all_g_transfer_systems<A>(universe: &GTransferUniverse<A>) -> Vec<RawGTransferSystem> {
    universe
        .context()
        .all_concepts_raw()
        .into_iter()
        .map(|concept| RawGTransferSystem::new(concept.extent))
        .collect()
}

fn g_containment_lattice<A>(
    universe: Arc<GTransferUniverse<A>>,
    systems: Vec<RawGTransferSystem>,
) -> Result<GTransferLattice<A>, LatticeError> {
    let poset = g_transfer_systems_ordered_by(systems, |left, right| {
        is_subset(left.orbit_arrows(), right.orbit_arrows())
    })?;
    Ok(GTransferLattice::new(universe, Lattice::new(poset)?))
}

fn g_transfer_systems_ordered_by<F>(
    systems: Vec<RawGTransferSystem>,
    predicate: F,
) -> Result<Poset<RawGTransferSystem>, PosetError>
where
    F: Fn(&RawGTransferSystem, &RawGTransferSystem) -> bool,
{
    let relation = systems
        .iter()
        .map(|left| systems.iter().map(|right| predicate(left, right)).collect())
        .collect();
    Poset::from_relation(systems, relation)
}

fn relation_orbit_error(error: PointOrbitError, relations: &[Edge]) -> GLatticeError {
    match error {
        PointOrbitError::Group(error) => error.into(),
        PointOrbitError::MissingTransporter { canonical, target } => {
            GLatticeError::MissingTransporter {
                canonical: relations[canonical],
                target: relations[target],
            }
        }
        PointOrbitError::MissingPreimage { canonical, target } => GLatticeError::MissingPreimage {
            canonical: relations[canonical],
            target: relations[target],
        },
    }
}
