use crate::lattice::{Lattice, LatticeError};
use crate::morphism::LatticeMapError;
use crate::poset::{Edge, EdgeSet, ElementId, Poset, PosetError};
use crate::transfer_lattice::bitvec_subset;
use bitvec::prelude::*;
use fcars::FormalContext;
use gap_sys::{Gap, GapElement, GapObj, GlobalGapGuard};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

pub type GTransferContext = FormalContext<RelationOrbitLabel, RelationOrbitLabel>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawGTransferSystem {
    /// Bitmask of the non-identity relation orbits in the transfer system.
    orbit_arrows: BitVec,
}

pub struct GLattice<A> {
    lattice: Arc<Lattice<A>>,
    group: GapObj,
    element_action_homomorphism: GapObj,
    element_image_group: GapObj,
    relation_action_homomorphism: GapObj,
    relation_image_group: GapObj,
    element_generator_permutations: Vec<Vec<ElementId>>,
    relation_generator_permutations: Vec<Vec<usize>>,
    relations: Vec<Edge>,
    relation_ids: Vec<Vec<Option<usize>>>,
    relation_to_orbit: Vec<usize>,
    relation_orbits: Vec<RelationOrbit>,
}

pub struct RelationOrbit {
    canonical_relation_id: usize,
    canonical_representative: Edge,
    relation_ids: Vec<usize>,
    relations: Vec<Edge>,
    stabilizer: GapObj,
    transporters: Vec<RelationTransporter>,
}

pub struct RelationTransporter {
    relation_id: usize,
    relation: Edge,
    group_element: GapObj,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationOrbitLabel {
    orbit_id: usize,
    canonical_relation_id: usize,
    canonical_representative: Edge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GapSubgroup {
    conjugacy_class: usize,
    class_element: usize,
}

pub struct SubgroupGLattice {
    g_lattice: GLattice<GapSubgroup>,
    gap_lattice: GapObj,
    conjugacy_classes: GapObj,
    subgroup_list: GapObj,
    subgroups: Vec<GapObj>,
}

/// Shared ambient data that gives raw G-transfer-system bitsets their meaning.
#[derive(Debug)]
pub struct GTransferUniverse<A> {
    underlying_lattice: Arc<Lattice<A>>,
    context: GTransferContext,
    relation_orbits: Vec<Vec<Edge>>,
}

/// An owned transfer system on a G-lattice together with its ambient data.
#[derive(Debug)]
pub struct GTransferSystem<A> {
    raw: RawGTransferSystem,
    universe: Arc<GTransferUniverse<A>>,
}

/// A lattice of transfer systems on a fixed G-lattice, ordered by containment.
#[derive(Debug, Clone)]
pub struct GTransferLattice<A> {
    universe: Arc<GTransferUniverse<A>>,
    lattice: Lattice<RawGTransferSystem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GLatticeError {
    Gap(String),
    Poset(PosetError),
    Lattice(LatticeError),
    GroupIsNotFinite,
    NotAGroupHomomorphism,
    GeneratorCountMismatch {
        expected: usize,
        actual: usize,
    },
    WrongPermutationLength {
        generator: usize,
        expected: usize,
        actual: usize,
    },
    PermutationImageOutOfBounds {
        generator: usize,
        element: ElementId,
        image: ElementId,
        len: usize,
    },
    DuplicatePermutationImage {
        generator: usize,
        first_element: ElementId,
        second_element: ElementId,
        image: ElementId,
    },
    NotALatticeAutomorphism {
        generator: usize,
        source: LatticeMapError,
    },
    HomomorphismByImagesFailed,
    RelationImageMissing {
        generator: usize,
        relation: Edge,
        image: Edge,
    },
    MissingTransporter {
        canonical: Edge,
        target: Edge,
    },
    MissingPreimage {
        canonical: Edge,
        target: Edge,
    },
    SubgroupConjugateNotFound {
        generator: usize,
        subgroup: GapSubgroup,
    },
}

struct GapAction {
    image_group: GapObj,
    homomorphism: GapObj,
}

struct GLatticeParts<A> {
    lattice: Arc<Lattice<A>>,
    group: GapObj,
    element_action_homomorphism: GapObj,
    element_image_group: GapObj,
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
                subgroup.conjugacy_class, subgroup.class_element
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

impl PartialOrd for RawGTransferSystem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.orbit_arrows == other.orbit_arrows {
            Some(std::cmp::Ordering::Equal)
        } else if bitvec_subset(&self.orbit_arrows, &other.orbit_arrows) {
            Some(std::cmp::Ordering::Less)
        } else if bitvec_subset(&other.orbit_arrows, &self.orbit_arrows) {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}

impl RawGTransferSystem {
    fn new(orbit_arrows: BitVec) -> Self {
        Self { orbit_arrows }
    }

    pub fn orbit_arrows(&self) -> &BitVec {
        &self.orbit_arrows
    }
}

impl RelationOrbitLabel {
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

    pub fn orbit_id(&self) -> usize {
        self.orbit_id
    }

    pub fn canonical_relation_id(&self) -> usize {
        self.canonical_relation_id
    }

    pub fn canonical_representative(&self) -> Edge {
        self.canonical_representative
    }
}

impl GapSubgroup {
    pub fn new(conjugacy_class: usize, class_element: usize) -> Self {
        Self {
            conjugacy_class,
            class_element,
        }
    }

    pub fn conjugacy_class(&self) -> usize {
        self.conjugacy_class
    }

    pub fn class_element(&self) -> usize {
        self.class_element
    }

    pub fn gap_conjugacy_class(&self) -> usize {
        self.conjugacy_class + 1
    }

    pub fn gap_class_element(&self) -> usize {
        self.class_element + 1
    }
}

impl<A> GLattice<A> {
    pub fn from_gap_homomorphism(
        lattice: Arc<Lattice<A>>,
        group: &GapElement,
        homomorphism: &GapElement,
    ) -> Result<Self, GLatticeError> {
        let mut gap = global_gap()?;
        validate_finite_group(&mut gap, group)?;
        validate_group_homomorphism(&mut gap, homomorphism)?;

        let group = gap.root(group.clone());
        let element_action_homomorphism = gap.root(homomorphism.clone());
        let element_image_group = call_global(
            &mut gap,
            "Image",
            &[element_action_homomorphism.as_element()],
        )?;
        let source_generators = call_global(&mut gap, "GeneratorsOfGroup", &[group.as_element()])?;
        let element_generator_permutations = generator_images_from_homomorphism(
            &mut gap,
            &lattice,
            &source_generators,
            homomorphism,
        )?;
        let relations = lattice.as_poset().all_relations_iter().collect::<Vec<_>>();
        let relation_ids = relation_id_matrix(lattice.size(), &relations);
        let relation_generator_permutations = relation_generator_permutations(
            &relations,
            &relation_ids,
            &element_generator_permutations,
        )?;
        let relation_action = gap_action_from_generator_permutations(
            &mut gap,
            group.as_element(),
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

    pub fn from_generator_images(
        lattice: Arc<Lattice<A>>,
        group: &GapElement,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        let mut gap = global_gap()?;
        Self::from_generator_images_with_gap(&mut gap, lattice, group, generator_images)
    }

    fn from_generator_images_with_gap(
        gap: &mut Gap,
        lattice: Arc<Lattice<A>>,
        group: &GapElement,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        validate_finite_group(gap, group)?;
        let group = gap.root(group.clone());
        let source_generators = call_global(gap, "GeneratorsOfGroup", &[group.as_element()])?;
        if generator_images.len() != gap.list_len(source_generators.as_element()) {
            return Err(GLatticeError::GeneratorCountMismatch {
                expected: gap.list_len(source_generators.as_element()),
                actual: generator_images.len(),
            });
        }

        for (generator, image) in generator_images.iter().enumerate() {
            validate_lattice_automorphism(generator, &lattice, image)?;
        }

        let element_action = gap_action_from_generator_permutations(
            gap,
            group.as_element(),
            &source_generators,
            &generator_images,
            lattice.size(),
        )?;
        let relations = lattice.as_poset().all_relations_iter().collect::<Vec<_>>();
        let relation_ids = relation_id_matrix(lattice.size(), &relations);
        let relation_generator_permutations =
            relation_generator_permutations(&relations, &relation_ids, &generator_images)?;
        let relation_action = gap_action_from_generator_permutations(
            gap,
            group.as_element(),
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

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.lattice
    }

    pub fn group(&self) -> &GapElement {
        self.group.as_element()
    }

    pub fn element_action_homomorphism(&self) -> &GapElement {
        self.element_action_homomorphism.as_element()
    }

    pub fn element_image_group(&self) -> &GapElement {
        self.element_image_group.as_element()
    }

    pub fn relation_action_homomorphism(&self) -> &GapElement {
        self.relation_action_homomorphism.as_element()
    }

    pub fn relation_image_group(&self) -> &GapElement {
        self.relation_image_group.as_element()
    }

    pub fn element_generator_permutations(&self) -> &[Vec<ElementId>] {
        &self.element_generator_permutations
    }

    pub fn relation_generator_permutations(&self) -> &[Vec<usize>] {
        &self.relation_generator_permutations
    }

    pub fn relations(&self) -> &[Edge] {
        &self.relations
    }

    pub fn relation(&self, relation_id: usize) -> Option<Edge> {
        self.relations.get(relation_id).copied()
    }

    pub fn relation_id(&self, relation: Edge) -> Option<usize> {
        self.relation_ids
            .get(relation.from)
            .and_then(|row| row.get(relation.to))
            .copied()
            .flatten()
    }

    pub fn relation_orbits(&self) -> &[RelationOrbit] {
        &self.relation_orbits
    }

    pub fn relation_orbit_by_id(&self, relation_id: usize) -> Option<&RelationOrbit> {
        self.relation_to_orbit
            .get(relation_id)
            .and_then(|&orbit| self.relation_orbits.get(orbit))
    }

    pub fn relation_orbit(&self, relation: Edge) -> Option<&RelationOrbit> {
        self.relation_id(relation)
            .and_then(|relation_id| self.relation_orbit_by_id(relation_id))
    }

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

    pub fn transfer_universe(&self) -> Arc<GTransferUniverse<A>> {
        Arc::new(GTransferUniverse::new(self))
    }

    pub fn transfer_systems_containment(&self) -> Result<GTransferLattice<A>, GLatticeError> {
        self.transfer_universe().containment_lattice()
    }

    fn from_parts(gap: &mut Gap, parts: GLatticeParts<A>) -> Result<Self, GLatticeError> {
        let relation_to_orbit = relation_to_orbit(
            parts.relations.len(),
            &parts.relation_generator_permutations,
            &parts.relations,
        );
        let relation_orbits = build_relation_orbits(
            gap,
            &parts.relations,
            &relation_to_orbit,
            parts.relation_action.image_group.as_element(),
            parts.relation_action.homomorphism.as_element(),
        )?;

        Ok(Self {
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
    pub fn from_subgroup_lattice(group: &GapElement) -> Result<SubgroupGLattice, GLatticeError> {
        SubgroupGLattice::new(group)
    }
}

impl SubgroupGLattice {
    pub fn new(group: &GapElement) -> Result<Self, GLatticeError> {
        let mut gap = global_gap()?;
        validate_finite_group(&mut gap, group)?;

        let gap_lattice = call_global(&mut gap, "LatticeSubgroups", &[group])?;
        let conjugacy_classes = call_global(
            &mut gap,
            "ConjugacyClassesSubgroups",
            &[gap_lattice.as_element()],
        )?;
        let (labels, subgroups) = subgroup_lattice_elements(&mut gap, &conjugacy_classes)?;
        let subgroup_elements = subgroups
            .iter()
            .map(|subgroup| subgroup.as_element().clone())
            .collect::<Vec<_>>();
        let subgroup_list = gap.list_rooted(&subgroup_elements);
        let relation = subgroup_inclusion_relation(&mut gap, &subgroups)?;
        let lattice = Arc::new(Lattice::new(Poset::from_relation(
            labels.clone(),
            relation,
        )?)?);
        let generator_images = subgroup_conjugation_generator_images(
            &mut gap,
            group,
            &subgroups,
            subgroup_list.as_element(),
            &labels,
        )?;
        let g_lattice =
            GLattice::from_generator_images_with_gap(&mut gap, lattice, group, generator_images)?;

        Ok(Self {
            g_lattice,
            gap_lattice,
            conjugacy_classes,
            subgroup_list,
            subgroups,
        })
    }

    pub fn g_lattice(&self) -> &GLattice<GapSubgroup> {
        &self.g_lattice
    }

    pub fn lattice(&self) -> &Arc<Lattice<GapSubgroup>> {
        self.g_lattice.lattice()
    }

    pub fn gap_lattice(&self) -> &GapElement {
        self.gap_lattice.as_element()
    }

    pub fn conjugacy_classes(&self) -> &GapElement {
        self.conjugacy_classes.as_element()
    }

    pub fn subgroup_list(&self) -> &GapElement {
        self.subgroup_list.as_element()
    }

    pub fn subgroups(&self) -> &[GapObj] {
        &self.subgroups
    }

    pub fn subgroup(&self, id: ElementId) -> Option<&GapElement> {
        self.subgroups.get(id).map(GapObj::as_element)
    }
}

impl<A> GTransferUniverse<A> {
    pub fn new(g_lattice: &GLattice<A>) -> Self {
        let context = g_lattice.transfer_context();
        let relation_orbits = context
            .objects
            .iter()
            .map(|label| {
                g_lattice.relation_orbits[label.orbit_id()]
                    .relations()
                    .to_vec()
            })
            .collect();
        Self {
            underlying_lattice: Arc::clone(g_lattice.lattice()),
            context,
            relation_orbits,
        }
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        &self.underlying_lattice
    }

    pub fn context(&self) -> &GTransferContext {
        &self.context
    }

    pub fn relation_orbit_labels(&self) -> &[RelationOrbitLabel] {
        &self.context.objects
    }

    pub fn relation_orbit_relations(&self, orbit_label_id: usize) -> Option<&[Edge]> {
        self.relation_orbits.get(orbit_label_id).map(Vec::as_slice)
    }

    pub fn transfer_systems(self: &Arc<Self>) -> Vec<GTransferSystem<A>> {
        all_g_transfer_systems(self)
            .into_iter()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(self)))
            .collect()
    }

    pub fn containment_lattice(self: &Arc<Self>) -> Result<GTransferLattice<A>, GLatticeError> {
        Ok(g_containment_lattice(
            Arc::clone(self),
            all_g_transfer_systems(self),
        )?)
    }
}

impl<A> GTransferSystem<A> {
    pub fn new(raw: RawGTransferSystem, universe: Arc<GTransferUniverse<A>>) -> Self {
        Self { raw, universe }
    }

    pub fn raw(&self) -> &RawGTransferSystem {
        &self.raw
    }

    pub fn universe(&self) -> &Arc<GTransferUniverse<A>> {
        &self.universe
    }

    pub fn lattice(&self) -> &Arc<Lattice<A>> {
        self.universe.lattice()
    }

    pub fn relation_orbit_labels(&self) -> Vec<RelationOrbitLabel> {
        self.raw
            .orbit_arrows()
            .iter_ones()
            .map(|orbit_label_id| self.universe.relation_orbit_labels()[orbit_label_id])
            .collect()
    }

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

    pub fn universe(&self) -> &Arc<GTransferUniverse<A>> {
        &self.universe
    }

    pub fn raw_lattice(&self) -> &Lattice<RawGTransferSystem> {
        &self.lattice
    }

    pub fn as_poset(&self) -> &Poset<RawGTransferSystem> {
        self.lattice.as_poset()
    }

    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    pub fn meet_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.meet_id(left, right)
    }

    pub fn join_id(&self, left: ElementId, right: ElementId) -> ElementId {
        self.lattice.join_id(left, right)
    }

    pub fn bottom(&self) -> ElementId {
        self.lattice.bottom()
    }

    pub fn top(&self) -> ElementId {
        self.lattice.top()
    }

    pub fn system(&self, id: ElementId) -> Option<GTransferSystem<A>> {
        self.lattice
            .element(id)
            .cloned()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    pub fn systems(&self) -> impl Iterator<Item = GTransferSystem<A>> + '_ {
        self.lattice
            .elements()
            .iter()
            .cloned()
            .map(|raw| GTransferSystem::new(raw, Arc::clone(&self.universe)))
    }

    pub fn to_system_lattice(&self) -> Lattice<GTransferSystem<A>> {
        self.lattice
            .relabelled(|raw| GTransferSystem::new(raw.clone(), Arc::clone(&self.universe)))
    }
}

impl RelationOrbit {
    pub fn canonical_relation_id(&self) -> usize {
        self.canonical_relation_id
    }

    pub fn canonical_representative(&self) -> Edge {
        self.canonical_representative
    }

    pub fn relation_ids(&self) -> &[usize] {
        &self.relation_ids
    }

    pub fn relations(&self) -> &[Edge] {
        &self.relations
    }

    pub fn stabilizer(&self) -> &GapElement {
        self.stabilizer.as_element()
    }

    pub fn transporters(&self) -> &[RelationTransporter] {
        &self.transporters
    }

    pub fn transporter_for_relation_id(&self, relation_id: usize) -> Option<&RelationTransporter> {
        self.transporters
            .iter()
            .find(|transporter| transporter.relation_id == relation_id)
    }

    pub fn transporter_for_relation(&self, relation: Edge) -> Option<&RelationTransporter> {
        self.transporters
            .iter()
            .find(|transporter| transporter.relation == relation)
    }
}

impl RelationTransporter {
    pub fn relation_id(&self) -> usize {
        self.relation_id
    }

    pub fn relation(&self) -> Edge {
        self.relation
    }

    pub fn group_element(&self) -> &GapElement {
        self.group_element.as_element()
    }
}

fn subgroup_lattice_elements(
    gap: &mut Gap,
    conjugacy_classes: &GapObj,
) -> Result<(Vec<GapSubgroup>, Vec<GapObj>), GLatticeError> {
    let class_count = gap.list_len(conjugacy_classes.as_element());
    let mut labels = Vec::new();
    let mut subgroups = Vec::new();

    for conjugacy_class in 0..class_count {
        let class = get_list_elem(gap, conjugacy_classes.as_element(), conjugacy_class)?;
        let class_size = call_global(gap, "Size", &[&class])?;
        let class_size = integer_usize(gap, class_size.as_element())?;
        for class_element in 0..class_size {
            let n = gap.int((class_element + 1) as isize);
            let subgroup = call_global(gap, "ClassElementLattice", &[&class, &n])?;
            labels.push(GapSubgroup::new(conjugacy_class, class_element));
            subgroups.push(subgroup);
        }
    }

    Ok((labels, subgroups))
}

fn subgroup_inclusion_relation(
    gap: &mut Gap,
    subgroups: &[GapObj],
) -> Result<Vec<BitVec>, GLatticeError> {
    let n = subgroups.len();
    let mut relation = vec![BitVec::repeat(false, n); n];
    for (lower, lower_subgroup) in subgroups.iter().enumerate() {
        for (upper, upper_subgroup) in subgroups.iter().enumerate() {
            let is_subgroup = call_global(
                gap,
                "IsSubgroup",
                &[upper_subgroup.as_element(), lower_subgroup.as_element()],
            )?;
            relation[lower].set(upper, boolean(gap, is_subgroup.as_element())?);
        }
    }
    Ok(relation)
}

fn subgroup_conjugation_generator_images(
    gap: &mut Gap,
    group: &GapElement,
    subgroups: &[GapObj],
    subgroup_list: &GapElement,
    labels: &[GapSubgroup],
) -> Result<Vec<Vec<ElementId>>, GLatticeError> {
    let generators = call_global(gap, "GeneratorsOfGroup", &[group])?;
    let generator_count = gap.list_len(generators.as_element());
    let mut result = Vec::with_capacity(generator_count);

    for generator in 0..generator_count {
        let gap_generator = get_list_elem(gap, generators.as_element(), generator)?;
        let mut image = Vec::with_capacity(subgroups.len());
        for (subgroup_id, subgroup) in subgroups.iter().enumerate() {
            let conjugate = call_global(gap, "OnPoints", &[subgroup.as_element(), &gap_generator])?;
            let position = call_global(gap, "Position", &[subgroup_list, conjugate.as_element()])?;
            if gap.is_fail(position.as_element()) {
                return Err(GLatticeError::SubgroupConjugateNotFound {
                    generator,
                    subgroup: labels[subgroup_id],
                });
            }
            let position = integer_usize(gap, position.as_element())?;
            let image_id =
                position
                    .checked_sub(1)
                    .ok_or(GLatticeError::SubgroupConjugateNotFound {
                        generator,
                        subgroup: labels[subgroup_id],
                    })?;
            image.push(image_id);
        }
        result.push(image);
    }

    Ok(result)
}

fn validate_finite_group(gap: &mut Gap, group: &GapElement) -> Result<(), GLatticeError> {
    let is_finite = call_global(gap, "IsFinite", &[group])?;
    if boolean(gap, is_finite.as_element())? {
        Ok(())
    } else {
        Err(GLatticeError::GroupIsNotFinite)
    }
}

fn validate_group_homomorphism(
    gap: &mut Gap,
    homomorphism: &GapElement,
) -> Result<(), GLatticeError> {
    let is_homomorphism = call_global(gap, "IsGroupHomomorphism", &[homomorphism])?;
    if boolean(gap, is_homomorphism.as_element())? {
        Ok(())
    } else {
        Err(GLatticeError::NotAGroupHomomorphism)
    }
}

fn generator_images_from_homomorphism<A>(
    gap: &mut Gap,
    lattice: &Arc<Lattice<A>>,
    source_generators: &GapObj,
    homomorphism: &GapElement,
) -> Result<Vec<Vec<ElementId>>, GLatticeError> {
    let generator_count = gap.list_len(source_generators.as_element());
    let mut result = Vec::with_capacity(generator_count);
    for generator in 0..generator_count {
        let source_generator = get_list_elem(gap, source_generators.as_element(), generator)?;
        let image = call_global(gap, "Image", &[homomorphism, &source_generator])?;
        let permutation = permutation_images_zero_based(gap, image.as_element(), lattice.size())?;
        validate_lattice_automorphism(generator, lattice, &permutation)?;
        result.push(permutation);
    }
    Ok(result)
}

fn validate_lattice_automorphism<A>(
    generator: usize,
    lattice: &Arc<Lattice<A>>,
    image: &[ElementId],
) -> Result<(), GLatticeError> {
    validate_permutation(generator, lattice.size(), image)?;
    crate::morphism::LatticeMap::new(Arc::clone(lattice), Arc::clone(lattice), image.to_vec())
        .map_err(|source| GLatticeError::NotALatticeAutomorphism { generator, source })?;
    Ok(())
}

fn validate_permutation(
    generator: usize,
    len: usize,
    image: &[ElementId],
) -> Result<(), GLatticeError> {
    if image.len() != len {
        return Err(GLatticeError::WrongPermutationLength {
            generator,
            expected: len,
            actual: image.len(),
        });
    }

    let mut seen = vec![None; len];
    for (element, &target) in image.iter().enumerate() {
        if target >= len {
            return Err(GLatticeError::PermutationImageOutOfBounds {
                generator,
                element,
                image: target,
                len,
            });
        }
        if let Some(first_element) = seen[target] {
            return Err(GLatticeError::DuplicatePermutationImage {
                generator,
                first_element,
                second_element: element,
                image: target,
            });
        }
        seen[target] = Some(element);
    }
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
        bitvec_subset(left.orbit_arrows(), right.orbit_arrows())
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

fn gap_action_from_generator_permutations(
    gap: &mut Gap,
    group: &GapElement,
    source_generators: &GapObj,
    generator_permutations: &[Vec<usize>],
    degree: usize,
) -> Result<GapAction, GLatticeError> {
    let gap_generators = generator_permutations
        .iter()
        .map(|permutation| permutation_from_zero_based_images(gap, permutation))
        .collect::<Result<Vec<_>, _>>()?;

    let image_group_generators = if gap_generators.is_empty() {
        vec![permutation_from_zero_based_images(
            gap,
            &(0..degree).collect::<Vec<_>>(),
        )?]
    } else {
        gap_generators.clone()
    };

    let image_group_generator_elements = image_group_generators
        .iter()
        .map(|generator| generator.as_element().clone())
        .collect::<Vec<_>>();
    let image_group_generator_list = gap.list_rooted(&image_group_generator_elements);
    let image_group = call_global(gap, "Group", &[image_group_generator_list.as_element()])?;

    let homomorphism_image_elements = gap_generators
        .iter()
        .map(|generator| generator.as_element().clone())
        .collect::<Vec<_>>();
    let homomorphism_image_list = gap.list_rooted(&homomorphism_image_elements);
    let homomorphism = call_global(
        gap,
        "GroupHomomorphismByImages",
        &[
            group,
            image_group.as_element(),
            source_generators.as_element(),
            homomorphism_image_list.as_element(),
        ],
    )?;

    if gap.is_fail(homomorphism.as_element()) {
        return Err(GLatticeError::HomomorphismByImagesFailed);
    }
    validate_group_homomorphism(gap, homomorphism.as_element())?;

    Ok(GapAction {
        image_group,
        homomorphism,
    })
}

fn relation_to_orbit(
    relation_count: usize,
    generator_permutations: &[Vec<usize>],
    relations: &[Edge],
) -> Vec<usize> {
    let mut relation_to_orbit = vec![usize::MAX; relation_count];
    let mut next_orbit = 0;

    for seed in 0..relation_count {
        if relation_to_orbit[seed] != usize::MAX {
            continue;
        }

        let mut queue = VecDeque::from([seed]);
        relation_to_orbit[seed] = next_orbit;
        while let Some(relation_id) = queue.pop_front() {
            for permutation in generator_permutations {
                let image = permutation[relation_id];
                if relation_to_orbit[image] == usize::MAX {
                    relation_to_orbit[image] = next_orbit;
                    queue.push_back(image);
                }
            }
        }

        next_orbit += 1;
    }

    debug_assert!(relation_to_orbit.iter().all(|&orbit| orbit != usize::MAX));
    debug_assert!(relations.len() == relation_to_orbit.len());
    relation_to_orbit
}

fn build_relation_orbits(
    gap: &mut Gap,
    relations: &[Edge],
    relation_to_orbit: &[usize],
    relation_image_group: &GapElement,
    relation_action_homomorphism: &GapElement,
) -> Result<Vec<RelationOrbit>, GLatticeError> {
    let orbit_count = relation_to_orbit
        .iter()
        .copied()
        .max()
        .map_or(0, |orbit| orbit + 1);
    let mut orbit_relation_ids = vec![Vec::new(); orbit_count];
    for (relation_id, &orbit) in relation_to_orbit.iter().enumerate() {
        orbit_relation_ids[orbit].push(relation_id);
    }

    orbit_relation_ids
        .into_iter()
        .map(|mut relation_ids| {
            relation_ids.sort_unstable();
            let canonical_relation_id = relation_ids[0];
            let canonical_representative = relations[canonical_relation_id];
            let canonical_point = gap.int((canonical_relation_id + 1) as isize);
            let on_points = global_obj(gap, "OnPoints")?;
            let image_stabilizer = call_global(
                gap,
                "Stabilizer",
                &[
                    relation_image_group,
                    &canonical_point,
                    on_points.as_element(),
                ],
            )?;
            let stabilizer = call_global(
                gap,
                "PreImage",
                &[relation_action_homomorphism, image_stabilizer.as_element()],
            )?;
            let mut orbit_relations = Vec::with_capacity(relation_ids.len());
            let mut transporters = Vec::with_capacity(relation_ids.len());

            for &relation_id in &relation_ids {
                let relation = relations[relation_id];
                orbit_relations.push(relation);
                let target_point = gap.int((relation_id + 1) as isize);
                let image_transporter = call_global(
                    gap,
                    "RepresentativeAction",
                    &[
                        relation_image_group,
                        &canonical_point,
                        &target_point,
                        on_points.as_element(),
                    ],
                )?;
                if gap.is_fail(image_transporter.as_element()) {
                    return Err(GLatticeError::MissingTransporter {
                        canonical: canonical_representative,
                        target: relation,
                    });
                }
                let group_element = call_global(
                    gap,
                    "PreImagesRepresentative",
                    &[relation_action_homomorphism, image_transporter.as_element()],
                )?;
                if gap.is_fail(group_element.as_element()) {
                    return Err(GLatticeError::MissingPreimage {
                        canonical: canonical_representative,
                        target: relation,
                    });
                }
                transporters.push(RelationTransporter {
                    relation_id,
                    relation,
                    group_element,
                });
            }

            Ok(RelationOrbit {
                canonical_relation_id,
                canonical_representative,
                relation_ids,
                relations: orbit_relations,
                stabilizer,
                transporters,
            })
        })
        .collect()
}

fn global_gap() -> Result<GlobalGapGuard, GLatticeError> {
    gap_sys::global().map_err(gap_error)
}

fn global_obj(gap: &mut Gap, name: &str) -> Result<GapObj, GLatticeError> {
    gap.global_rooted(name).map_err(gap_error)
}

fn call_global(gap: &mut Gap, name: &str, args: &[&GapElement]) -> Result<GapObj, GLatticeError> {
    gap.call_global_rooted(name, args).map_err(gap_error)
}

fn get_list_elem(
    gap: &mut Gap,
    list: &GapElement,
    idx: usize,
) -> Result<GapElement, GLatticeError> {
    gap.get_list_elem(list, idx).map_err(gap_error)
}

fn integer_usize(gap: &mut Gap, element: &GapElement) -> Result<usize, GLatticeError> {
    gap.integer_usize(element).map_err(gap_error)
}

fn boolean(gap: &mut Gap, element: &GapElement) -> Result<bool, GLatticeError> {
    gap.boolean(element).map_err(gap_error)
}

fn permutation_from_zero_based_images(
    gap: &mut Gap,
    images: &[usize],
) -> Result<GapObj, GLatticeError> {
    gap.permutation_from_zero_based_images_rooted(images)
        .map_err(gap_error)
}

fn permutation_images_zero_based(
    gap: &mut Gap,
    permutation: &GapElement,
    degree: usize,
) -> Result<Vec<usize>, GLatticeError> {
    gap.permutation_images_zero_based(permutation, degree)
        .map_err(gap_error)
}

fn gap_error(error: impl fmt::Display) -> GLatticeError {
    GLatticeError::Gap(error.to_string())
}
