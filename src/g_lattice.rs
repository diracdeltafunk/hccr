//! GAP-backed finite group actions on finite lattices.
//!
//! A `G`-lattice is a finite lattice `L` together with an action of a finite
//! group `G`: every `g` in `G` permutes the elements of `L`, preserving meets,
//! joins, bottom, and top. Such a permutation is a lattice automorphism. Acting
//! on both endpoints sends a relation `x <= y` to `g.x <= g.y`, so `G` also
//! acts on the set of lattice relations.
//!
//! The **orbit** of a relation is the set of all its translates under `G`. Its
//! **stabilizer** consists of the group elements that leave it fixed, and a
//! **transporter** from `r` to `s` is a chosen group element satisfying
//! `g.r = s`. This module precomputes those data. The action is stored both as
//! GAP homomorphisms to permutation groups and as validated Rust permutations
//! of lattice element ids.
//! GAP values stored in these structures are [`gap_sys::GapValue`]s, so they
//! retain their GAP GC roots for as long as the Rust structures remain alive.
//!
//! Transfer systems invariant under the action are in
//! [`crate::g_transfer_lattice`], and cotransfer systems in
//! [`crate::g_cotransfer_lattice`].
//!
//! The most common G-lattice is the lattice of subgroups of a finite group
//! with the conjugation action:
//!
//! ```no_run
//! use hccr::g_lattice::SubgroupGLattice;
//!
//! let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
//! assert_eq!(s3.transfer_system_count(), 9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`GLattice`] and [`SubgroupGLattice`] are cheap-to-clone handles. GAP may be
//! used from any thread, but these handles hold GAP values, which cannot be
//! sent between threads, so each handle stays on the thread that created it.

use crate::g_transfer_lattice::GTransferUniverse;
use crate::group_theory::{self, GapAction, GapSubgroup, GroupTheoryError, PointOrbitError};
use crate::lattice::{Lattice, LatticeError};
use crate::morphism::LatticeMapError;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use gap_sys::{Gap, GapValue};
use std::cell::OnceCell;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

/// A finite lattice equipped with an action of a finite GAP group.
///
/// The action of `G` on lattice elements is required to be by lattice
/// automorphisms, so it preserves all order-theoretic structure. The induced
/// action on all relations of the lattice is precomputed, together with
/// relation orbits and GAP stabilizer/transporter data for those orbits.
///
/// This is a cheap-to-clone handle. It dereferences to the underlying
/// [`Lattice`], so lattice methods such as `size` and `leq` apply directly.
#[derive(Clone)]
pub struct GLattice {
    pub(crate) data: Rc<GLatticeData>,
}

pub(crate) struct GLatticeData {
    action_coordinates: Rc<()>,
    pub(crate) lattice: Lattice,
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
    pub(crate) relation_orbits: Vec<RelationOrbit>,
    pub(crate) transfer_universe: OnceCell<Rc<GTransferUniverse>>,
    cotransfer_universe: OnceCell<Rc<crate::g_cotransfer_lattice::GCotransferUniverse>>,
}

/// One orbit of the `G`-action on lattice relations.
///
/// Relation orbits include identity relations. The canonical representative
/// is simply the relation with smallest deterministic relation id in the
/// orbit; this choice supplies stable output and is not an additional
/// mathematical structure.
pub struct RelationOrbit {
    canonical_relation_id: usize,
    canonical_representative: Edge,
    relation_ids: Vec<usize>,
    pub(crate) relations: Vec<Edge>,
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
/// Its elements are all actual subgroups, ordered by inclusion. An element
/// `g` acts by sending a subgroup `H` to the conjugate `g H g^-1`; conjugation
/// preserves inclusion, intersections, and generated joins, hence acts by
/// lattice automorphisms. This wrapper keeps both the Rust [`GLattice`] and the
/// GAP objects used to construct it alive.
/// The subgroup lattice of a finite GAP group with the conjugation action.
///
/// Its elements are all actual subgroups, ordered by inclusion, and labelled
/// by [`GapSubgroup`] values. An element `g` acts by sending a subgroup `H` to
/// the conjugate `g H g^-1`; conjugation preserves inclusion, intersections,
/// and generated joins, hence acts by lattice automorphisms.
///
/// This is a cheap-to-clone handle. It dereferences to its [`GLattice`], and
/// through that to the underlying [`Lattice`].
#[derive(Clone)]
pub struct SubgroupGLattice {
    data: Rc<SubgroupGLatticeData>,
}

struct SubgroupGLatticeData {
    g_lattice: GLattice,
    gap_lattice: GapValue,
    conjugacy_classes: GapValue,
    subgroup_list: GapValue,
    subgroups: Vec<GapValue>,
    subgroup_structure_descriptions: Vec<String>,
    subgroup_structure_descriptions_tex: Vec<String>,
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

struct GLatticeParts {
    lattice: Lattice,
    group: GapValue,
    element_action_homomorphism: GapValue,
    element_image_group: GapValue,
    relation_action: GapAction,
    element_generator_permutations: Vec<Vec<ElementId>>,
    relation_generator_permutations: Vec<Vec<usize>>,
    relations: Vec<Edge>,
    relation_ids: Vec<Vec<Option<usize>>>,
}

impl fmt::Debug for GLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLattice")
            .field("lattice", &self.data.lattice)
            .field(
                "element_generator_permutations",
                &self.data.element_generator_permutations,
            )
            .field("relation_orbits", &self.data.relation_orbits)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for SubgroupGLattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubgroupGLattice")
            .field("g_lattice", &self.data.g_lattice)
            .field("subgroup_count", &self.data.subgroups.len())
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

impl GLattice {
    /// Constructs a G-lattice from a GAP homomorphism to a permutation group.
    ///
    /// The homomorphism is interpreted as an action of `group` on the element
    /// ids `[1..n]` of the lattice, using GAP's one-based point convention.
    /// Each generator image is extracted, converted to zero-based Rust element
    /// ids, and validated as a lattice automorphism.  The induced action on
    /// lattice relations is then built automatically. Finally, GAP computes
    /// relation orbits, stabilizers, and transporters, which are cached in the
    /// result.
    pub fn from_gap_homomorphism(
        lattice: &Lattice,
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
            validate_lattice_automorphism(generator, lattice, permutation)?;
        }
        let relations = lattice
            .as_poset()
            .relations()
            .into_iter()
            .collect::<Vec<_>>();
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
                lattice: lattice.clone(),
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
        lattice: &Lattice,
        group: &GapValue,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        let mut gap = group_theory::global_gap()?;
        Self::from_generator_images_with_gap(&mut gap, lattice, group, generator_images)
    }

    fn from_generator_images_with_gap(
        gap: &mut Gap,
        lattice: &Lattice,
        group: &GapValue,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        group_theory::validate_finite_group(gap, group)?;
        let group = group.clone();
        let source_generators = group_theory::generators_of_group(gap, &group)?;
        group_theory::validate_generator_count(gap, &source_generators, generator_images.len())?;

        for (generator, image) in generator_images.iter().enumerate() {
            validate_lattice_automorphism(generator, lattice, image)?;
        }

        let element_action = group_theory::action_from_generator_permutations(
            gap,
            &group,
            &source_generators,
            &generator_images,
            lattice.size(),
        )?;
        let relations = lattice
            .as_poset()
            .relations()
            .into_iter()
            .collect::<Vec<_>>();
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
                lattice: lattice.clone(),
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
    pub fn lattice(&self) -> &Lattice {
        &self.data.lattice
    }

    /// Returns whether two handles refer to the very same G-lattice.
    pub fn ptr_eq(&self, other: &GLattice) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }

    /// Returns the opaque identity of this concrete group-action presentation.
    pub(crate) fn action_coordinates(&self) -> &Rc<()> {
        &self.data.action_coordinates
    }

    /// Returns the GAP group acting on the lattice.
    pub fn group(&self) -> &GapValue {
        &self.data.group
    }

    /// Returns the GAP homomorphism describing the action on lattice elements.
    pub fn element_action_homomorphism(&self) -> &GapValue {
        &self.data.element_action_homomorphism
    }

    /// Returns the image group of the element action.
    pub fn element_image_group(&self) -> &GapValue {
        &self.data.element_image_group
    }

    /// Returns the induced GAP homomorphism on lattice relations.
    pub fn relation_action_homomorphism(&self) -> &GapValue {
        &self.data.relation_action_homomorphism
    }

    /// Returns the image group of the induced relation action.
    pub fn relation_image_group(&self) -> &GapValue {
        &self.data.relation_image_group
    }

    /// Returns the validated generator permutations on lattice elements.
    ///
    /// Each inner vector is a zero-based permutation of element ids.
    pub fn element_generator_permutations(&self) -> &[Vec<ElementId>] {
        &self.data.element_generator_permutations
    }

    /// Returns the induced generator permutations on relation ids.
    ///
    /// Each inner vector is a zero-based permutation of the relation ids (see
    /// [`GLattice::relation`]).
    pub fn relation_generator_permutations(&self) -> &[Vec<usize>] {
        &self.data.relation_generator_permutations
    }

    /// Returns the relation with the given relation id.
    ///
    /// Relation ids number the relations `x <= y` of the lattice, identities
    /// included, in the order of [`Poset::relations`]: the relation with id
    /// `i` is the `i`-th element of `self.lattice().relations()`.
    ///
    /// Panics if `relation_id` is out of bounds.
    pub fn relation(&self, relation_id: usize) -> Edge {
        self.data.relations[relation_id]
    }

    /// Returns the relation id of `relation`, if it is a relation in the lattice.
    pub fn relation_id(&self, relation: Edge) -> Option<usize> {
        self.data
            .relation_ids
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }

    /// Returns the precomputed orbits of the action on all relations.
    pub fn relation_orbits(&self) -> &[RelationOrbit] {
        &self.data.relation_orbits
    }

    /// Returns the orbit containing the relation with the given relation id.
    ///
    /// Panics if `relation_id` is out of bounds.
    pub fn relation_orbit_by_id(&self, relation_id: usize) -> &RelationOrbit {
        &self.data.relation_orbits[self.data.relation_to_orbit[relation_id]]
    }

    /// Returns the orbit containing a relation, if it is a relation of the
    /// lattice.
    pub fn relation_orbit(&self, relation: Edge) -> Option<&RelationOrbit> {
        self.relation_id(relation)
            .map(|relation_id| self.relation_orbit_by_id(relation_id))
    }

    /// Returns labels for the non-identity relation orbits.
    ///
    /// These labels are the objects and attributes of the G-transfer-system
    /// formal context.
    pub fn non_identity_relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.data
            .relation_orbits
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

    pub(crate) fn cotransfer_universe_cell(
        &self,
    ) -> &OnceCell<Rc<crate::g_cotransfer_lattice::GCotransferUniverse>> {
        &self.data.cotransfer_universe
    }

    fn from_parts(gap: &mut Gap, parts: GLatticeParts) -> Result<Self, GLatticeError> {
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
            data: Rc::new(GLatticeData {
                action_coordinates: Rc::new(()),
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
                transfer_universe: OnceCell::new(),
                cotransfer_universe: OnceCell::new(),
            }),
        })
    }
}

impl SubgroupGLattice {
    /// Constructs the subgroup lattice of the finite group described by a GAP
    /// expression, with the conjugation action.
    ///
    /// ```no_run
    /// use hccr::g_lattice::SubgroupGLattice;
    ///
    /// let c4 = SubgroupGLattice::from_gap("CyclicGroup(4)")?;
    /// assert_eq!(c4.size(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_gap(expression: &str) -> Result<Self, GLatticeError> {
        let expression = expression.trim().trim_end_matches(';');
        let group = gap_sys::eval(&format!("{expression};"))
            .map_err(|error| GLatticeError::Gap(error.to_string()))?;
        Self::new(&group)
    }

    /// Constructs the subgroup lattice of a finite GAP group with conjugation action.
    pub fn new(group: &GapValue) -> Result<Self, GLatticeError> {
        let mut gap = group_theory::global_gap()?;
        let data = group_theory::subgroup_lattice_data(&mut gap, group)?;
        let subgroup_structure_descriptions_tex = data
            .structure_descriptions
            .iter()
            .map(|description| structure_description_to_tex(description))
            .collect();
        let lattice = Lattice::new(Poset::from_relation(
            data.labels.clone(),
            data.inclusion_relation,
        )?)?;
        let g_lattice = GLattice::from_generator_images_with_gap(
            &mut gap,
            &lattice,
            group,
            data.conjugation_generator_images,
        )?;

        Ok(Self {
            data: Rc::new(SubgroupGLatticeData {
                g_lattice,
                gap_lattice: data.gap_lattice,
                conjugacy_classes: data.conjugacy_classes,
                subgroup_list: data.subgroup_list,
                subgroups: data.subgroups,
                subgroup_structure_descriptions: data.structure_descriptions,
                subgroup_structure_descriptions_tex,
            }),
        })
    }

    /// Returns the G-lattice of subgroups.
    pub fn g_lattice(&self) -> &GLattice {
        &self.data.g_lattice
    }

    /// Returns the underlying subgroup lattice.
    pub fn lattice(&self) -> &Lattice {
        self.data.g_lattice.lattice()
    }

    /// Returns GAP's `LatticeSubgroups(group)` object.
    pub fn gap_lattice(&self) -> &GapValue {
        &self.data.gap_lattice
    }

    /// Returns GAP's conjugacy classes of subgroups for the subgroup lattice.
    pub fn conjugacy_classes(&self) -> &GapValue {
        &self.data.conjugacy_classes
    }

    /// Returns the rooted GAP list of subgroup objects in lattice element order.
    pub fn subgroup_list(&self) -> &GapValue {
        &self.data.subgroup_list
    }

    /// Returns the rooted GAP subgroup objects in lattice element order.
    pub fn subgroups(&self) -> &[GapValue] {
        &self.data.subgroups
    }

    /// Returns a GAP subgroup object by lattice element id.
    ///
    /// Panics if `id` is out of bounds.
    pub fn subgroup(&self, id: ElementId) -> &GapValue {
        &self.data.subgroups[id]
    }

    /// Returns GAP's `StructureDescription` strings in lattice element order.
    ///
    /// The strings are computed once while constructing this object, so they
    /// can later be used as labels in transfer-system diagrams without making
    /// further GAP calls. Conjugate but distinct subgroups may have the same
    /// structure description.
    pub fn subgroup_structure_descriptions(&self) -> &[String] {
        &self.data.subgroup_structure_descriptions
    }

    /// Returns GAP's `StructureDescription` string for one subgroup.
    ///
    /// Panics if `id` is out of bounds.
    pub fn subgroup_structure_description(&self, id: ElementId) -> &str {
        &self.data.subgroup_structure_descriptions[id]
    }

    /// Returns TeX math-mode labels for GAP's structure descriptions.
    ///
    /// These labels are cached in lattice element order. Wrapped in
    /// [`crate::tikz::TikzLabel::raw`], they can label the nodes of a drawing,
    /// as in [`crate::tikz::PosetDrawing::for_each_glyph`]. For example, GAP's
    /// `"C2"`
    /// becomes `"$C_{2}$"`, and `"C2 x C3"` becomes
    /// `"$C_{2} \\times C_{3}$"`.
    pub fn subgroup_structure_descriptions_tex(&self) -> &[String] {
        &self.data.subgroup_structure_descriptions_tex
    }

    /// Returns the TeX math-mode structure-description label for one subgroup.
    ///
    /// Panics if `id` is out of bounds.
    pub fn subgroup_structure_description_tex(&self, id: ElementId) -> &str {
        &self.data.subgroup_structure_descriptions_tex[id]
    }
}

impl Deref for GLattice {
    type Target = Lattice;

    fn deref(&self) -> &Lattice {
        &self.data.lattice
    }
}

impl Deref for SubgroupGLattice {
    type Target = GLattice;

    fn deref(&self) -> &GLattice {
        &self.data.g_lattice
    }
}

impl PartialEq for GLattice {
    /// G-lattices are equal only when they are the same object, because their
    /// relation-orbit coordinates depend on how the action was presented.
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl Eq for GLattice {}

impl std::hash::Hash for GLattice {
    /// Hashes the object's identity, matching `Eq`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
    }
}

impl PartialEq for SubgroupGLattice {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for SubgroupGLattice {}

impl std::hash::Hash for SubgroupGLattice {
    /// Hashes the object's identity, matching `Eq`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
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
    pub fn relations(&self) -> EdgeSet {
        self.relations.iter().copied().collect()
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

fn validate_lattice_automorphism(
    generator: usize,
    lattice: &Lattice,
    image: &[ElementId],
) -> Result<(), GLatticeError> {
    group_theory::validate_permutation(generator, lattice.size(), image)?;
    crate::morphism::LatticeMap::new(lattice, lattice, image.to_vec())
        .map_err(|source| GLatticeError::NotALatticeAutomorphism { generator, source })?;
    Ok(())
}

pub(crate) fn relation_id_matrix(len: usize, relations: &[Edge]) -> Vec<Vec<Option<usize>>> {
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
