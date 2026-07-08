use crate::lattice::{Lattice, LatticeError};
use crate::morphism::LatticeMapError;
use crate::poset::{Edge, ElementId, Poset, PosetError};
use bitvec::prelude::*;
use gap_sys::{Gap, GapElement};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone)]
pub struct GapSession {
    inner: Rc<RefCell<Gap>>,
}

pub struct RootedGapElement {
    session: GapSession,
    element: GapElement,
}

pub struct GLattice<A> {
    session: GapSession,
    lattice: Arc<Lattice<A>>,
    group: RootedGapElement,
    element_action_homomorphism: RootedGapElement,
    element_image_group: RootedGapElement,
    relation_action_homomorphism: RootedGapElement,
    relation_image_group: RootedGapElement,
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
    stabilizer: RootedGapElement,
    transporters: Vec<RelationTransporter>,
}

pub struct RelationTransporter {
    relation_id: usize,
    relation: Edge,
    group_element: RootedGapElement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GapSubgroup {
    conjugacy_class: usize,
    class_element: usize,
}

pub struct SubgroupGLattice {
    g_lattice: GLattice<GapSubgroup>,
    gap_lattice: RootedGapElement,
    conjugacy_classes: RootedGapElement,
    subgroup_list: RootedGapElement,
    subgroups: Vec<RootedGapElement>,
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
    image_group: RootedGapElement,
    homomorphism: RootedGapElement,
}

struct GLatticeParts<A> {
    lattice: Arc<Lattice<A>>,
    group: RootedGapElement,
    element_action_homomorphism: RootedGapElement,
    element_image_group: RootedGapElement,
    relation_action: GapAction,
    element_generator_permutations: Vec<Vec<ElementId>>,
    relation_generator_permutations: Vec<Vec<usize>>,
    relations: Vec<Edge>,
    relation_ids: Vec<Vec<Option<usize>>>,
}

impl fmt::Debug for GapSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GapSession").finish_non_exhaustive()
    }
}

impl fmt::Debug for RootedGapElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RootedGapElement")
            .field(&format_args!("{:p}", self.element))
            .finish()
    }
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

impl GapSession {
    pub fn try_new() -> Result<Self, GLatticeError> {
        Ok(Self {
            inner: Rc::new(RefCell::new(Gap::try_init().map_err(gap_error)?)),
        })
    }

    pub fn eval(&self, cmd: &str) -> Result<RootedGapElement, GLatticeError> {
        let element = self.with_gap_result(|gap| gap.eval(cmd))?;
        Ok(self.root(element))
    }

    pub fn global(&self, name: &str) -> Result<RootedGapElement, GLatticeError> {
        let element = self.with_gap_result(|gap| gap.global(name))?;
        Ok(self.root(element))
    }

    pub fn call_global(
        &self,
        name: &str,
        args: &[&GapElement],
    ) -> Result<RootedGapElement, GLatticeError> {
        let element = self.with_gap_result(|gap| gap.call_global(name, args))?;
        Ok(self.root(element))
    }

    pub fn element_string(&self, element: &GapElement) -> String {
        self.inner.borrow_mut().elem_string(element)
    }

    pub fn integer_usize(&self, element: &GapElement) -> Result<usize, GLatticeError> {
        self.with_gap_result(|gap| gap.integer_usize(element))
    }

    pub fn boolean(&self, element: &GapElement) -> Result<bool, GLatticeError> {
        self.with_gap_result(|gap| gap.boolean(element))
    }

    pub fn permutation_images_zero_based(
        &self,
        permutation: &GapElement,
        degree: usize,
    ) -> Result<Vec<usize>, GLatticeError> {
        self.with_gap_result(|gap| gap.permutation_images_zero_based(permutation, degree))
    }

    pub fn root(&self, element: GapElement) -> RootedGapElement {
        self.inner.borrow().alloc(&element);
        RootedGapElement {
            session: self.clone(),
            element,
        }
    }

    fn int(&self, value: usize) -> GapElement {
        self.inner.borrow().int(value as isize)
    }

    fn list(&self, elements: &[GapElement]) -> RootedGapElement {
        let list = self.inner.borrow().list(elements);
        self.root(list)
    }

    fn list_len(&self, list: &GapElement) -> usize {
        self.inner.borrow().list_len(list)
    }

    fn get_list_elem(&self, list: &GapElement, idx: usize) -> Result<GapElement, GLatticeError> {
        self.with_gap_result(|gap| gap.get_list_elem(list, idx))
    }

    fn permutation_from_zero_based_images(
        &self,
        images: &[usize],
    ) -> Result<RootedGapElement, GLatticeError> {
        let element = self.with_gap_result(|gap| gap.permutation_from_zero_based_images(images))?;
        Ok(self.root(element))
    }

    fn is_fail(&self, element: &GapElement) -> bool {
        self.inner.borrow().is_fail(element)
    }

    fn with_gap_result<T, E, F>(&self, f: F) -> Result<T, GLatticeError>
    where
        E: fmt::Display,
        F: FnOnce(&Gap) -> Result<T, E>,
    {
        f(&self.inner.borrow()).map_err(gap_error)
    }
}

impl RootedGapElement {
    pub fn as_element(&self) -> &GapElement {
        &self.element
    }
}

impl Drop for RootedGapElement {
    fn drop(&mut self) {
        if let Ok(gap) = self.session.inner.try_borrow() {
            gap.free(&self.element);
        }
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
        session: &GapSession,
        lattice: Arc<Lattice<A>>,
        group: &GapElement,
        homomorphism: &GapElement,
    ) -> Result<Self, GLatticeError> {
        validate_finite_group(session, group)?;
        validate_group_homomorphism(session, homomorphism)?;

        let group = session.root(group.clone());
        let element_action_homomorphism = session.root(homomorphism.clone());
        let element_image_group =
            session.call_global("Image", &[element_action_homomorphism.as_element()])?;
        let source_generators = session.call_global("GeneratorsOfGroup", &[group.as_element()])?;
        let element_generator_permutations = generator_images_from_homomorphism(
            session,
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
            session,
            group.as_element(),
            &source_generators,
            &relation_generator_permutations,
            relations.len(),
        )?;

        Self::from_parts(
            session,
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
        session: &GapSession,
        lattice: Arc<Lattice<A>>,
        group: &GapElement,
        generator_images: Vec<Vec<ElementId>>,
    ) -> Result<Self, GLatticeError> {
        validate_finite_group(session, group)?;

        let group = session.root(group.clone());
        let source_generators = session.call_global("GeneratorsOfGroup", &[group.as_element()])?;
        if generator_images.len() != session.list_len(source_generators.as_element()) {
            return Err(GLatticeError::GeneratorCountMismatch {
                expected: session.list_len(source_generators.as_element()),
                actual: generator_images.len(),
            });
        }

        for (generator, image) in generator_images.iter().enumerate() {
            validate_lattice_automorphism(generator, &lattice, image)?;
        }

        let element_action = gap_action_from_generator_permutations(
            session,
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
            session,
            group.as_element(),
            &source_generators,
            &relation_generator_permutations,
            relations.len(),
        )?;

        Self::from_parts(
            session,
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

    pub fn session(&self) -> &GapSession {
        &self.session
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

    fn from_parts(session: &GapSession, parts: GLatticeParts<A>) -> Result<Self, GLatticeError> {
        let relation_to_orbit = relation_to_orbit(
            parts.relations.len(),
            &parts.relation_generator_permutations,
            &parts.relations,
        );
        let relation_orbits = build_relation_orbits(
            session,
            &parts.relations,
            &relation_to_orbit,
            parts.relation_action.image_group.as_element(),
            parts.relation_action.homomorphism.as_element(),
        )?;

        Ok(Self {
            session: session.clone(),
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
    pub fn from_subgroup_lattice(
        session: &GapSession,
        group: &GapElement,
    ) -> Result<SubgroupGLattice, GLatticeError> {
        SubgroupGLattice::new(session, group)
    }
}

impl SubgroupGLattice {
    pub fn new(session: &GapSession, group: &GapElement) -> Result<Self, GLatticeError> {
        validate_finite_group(session, group)?;

        let gap_lattice = session.call_global("LatticeSubgroups", &[group])?;
        let conjugacy_classes =
            session.call_global("ConjugacyClassesSubgroups", &[gap_lattice.as_element()])?;
        let (labels, subgroups) = subgroup_lattice_elements(session, &conjugacy_classes)?;
        let subgroup_elements = subgroups
            .iter()
            .map(|subgroup| subgroup.as_element().clone())
            .collect::<Vec<_>>();
        let subgroup_list = session.list(&subgroup_elements);
        let relation = subgroup_inclusion_relation(session, &subgroups)?;
        let lattice = Arc::new(Lattice::new(Poset::from_relation(
            labels.clone(),
            relation,
        )?)?);
        let generator_images = subgroup_conjugation_generator_images(
            session,
            group,
            &subgroups,
            subgroup_list.as_element(),
            &labels,
        )?;
        let g_lattice = GLattice::from_generator_images(session, lattice, group, generator_images)?;

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

    pub fn subgroups(&self) -> &[RootedGapElement] {
        &self.subgroups
    }

    pub fn subgroup(&self, id: ElementId) -> Option<&GapElement> {
        self.subgroups.get(id).map(RootedGapElement::as_element)
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
    session: &GapSession,
    conjugacy_classes: &RootedGapElement,
) -> Result<(Vec<GapSubgroup>, Vec<RootedGapElement>), GLatticeError> {
    let class_count = session.list_len(conjugacy_classes.as_element());
    let mut labels = Vec::new();
    let mut subgroups = Vec::new();

    for conjugacy_class in 0..class_count {
        let class = session.get_list_elem(conjugacy_classes.as_element(), conjugacy_class)?;
        let class_size = session.call_global("Size", &[&class])?;
        let class_size = session.integer_usize(class_size.as_element())?;
        for class_element in 0..class_size {
            let n = session.int(class_element + 1);
            let subgroup = session.call_global("ClassElementLattice", &[&class, &n])?;
            labels.push(GapSubgroup::new(conjugacy_class, class_element));
            subgroups.push(subgroup);
        }
    }

    Ok((labels, subgroups))
}

fn subgroup_inclusion_relation(
    session: &GapSession,
    subgroups: &[RootedGapElement],
) -> Result<Vec<BitVec>, GLatticeError> {
    let n = subgroups.len();
    let mut relation = vec![BitVec::repeat(false, n); n];
    for (lower, lower_subgroup) in subgroups.iter().enumerate() {
        for (upper, upper_subgroup) in subgroups.iter().enumerate() {
            let is_subgroup = session.call_global(
                "IsSubgroup",
                &[upper_subgroup.as_element(), lower_subgroup.as_element()],
            )?;
            relation[lower].set(upper, session.boolean(is_subgroup.as_element())?);
        }
    }
    Ok(relation)
}

fn subgroup_conjugation_generator_images(
    session: &GapSession,
    group: &GapElement,
    subgroups: &[RootedGapElement],
    subgroup_list: &GapElement,
    labels: &[GapSubgroup],
) -> Result<Vec<Vec<ElementId>>, GLatticeError> {
    let generators = session.call_global("GeneratorsOfGroup", &[group])?;
    let generator_count = session.list_len(generators.as_element());
    let mut result = Vec::with_capacity(generator_count);

    for generator in 0..generator_count {
        let gap_generator = session.get_list_elem(generators.as_element(), generator)?;
        let mut image = Vec::with_capacity(subgroups.len());
        for (subgroup_id, subgroup) in subgroups.iter().enumerate() {
            let conjugate =
                session.call_global("OnPoints", &[subgroup.as_element(), &gap_generator])?;
            let position =
                session.call_global("Position", &[subgroup_list, conjugate.as_element()])?;
            if session.is_fail(position.as_element()) {
                return Err(GLatticeError::SubgroupConjugateNotFound {
                    generator,
                    subgroup: labels[subgroup_id],
                });
            }
            let position = session.integer_usize(position.as_element())?;
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

fn validate_finite_group(session: &GapSession, group: &GapElement) -> Result<(), GLatticeError> {
    let is_finite = session.call_global("IsFinite", &[group])?;
    if session.boolean(is_finite.as_element())? {
        Ok(())
    } else {
        Err(GLatticeError::GroupIsNotFinite)
    }
}

fn validate_group_homomorphism(
    session: &GapSession,
    homomorphism: &GapElement,
) -> Result<(), GLatticeError> {
    let is_homomorphism = session.call_global("IsGroupHomomorphism", &[homomorphism])?;
    if session.boolean(is_homomorphism.as_element())? {
        Ok(())
    } else {
        Err(GLatticeError::NotAGroupHomomorphism)
    }
}

fn generator_images_from_homomorphism<A>(
    session: &GapSession,
    lattice: &Arc<Lattice<A>>,
    source_generators: &RootedGapElement,
    homomorphism: &GapElement,
) -> Result<Vec<Vec<ElementId>>, GLatticeError> {
    let generator_count = session.list_len(source_generators.as_element());
    let mut result = Vec::with_capacity(generator_count);
    for generator in 0..generator_count {
        let source_generator = session.get_list_elem(source_generators.as_element(), generator)?;
        let image = session.call_global("Image", &[homomorphism, &source_generator])?;
        let permutation =
            session.permutation_images_zero_based(image.as_element(), lattice.size())?;
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

fn gap_action_from_generator_permutations(
    session: &GapSession,
    group: &GapElement,
    source_generators: &RootedGapElement,
    generator_permutations: &[Vec<usize>],
    degree: usize,
) -> Result<GapAction, GLatticeError> {
    let mut gap_generators = generator_permutations
        .iter()
        .map(|permutation| session.permutation_from_zero_based_images(permutation))
        .collect::<Result<Vec<_>, _>>()?;

    let image_group_generators = if gap_generators.is_empty() {
        vec![session.permutation_from_zero_based_images(&(0..degree).collect::<Vec<_>>())?]
    } else {
        gap_generators
            .iter()
            .map(|generator| session.root(generator.as_element().clone()))
            .collect()
    };

    let image_group_generator_elements = image_group_generators
        .iter()
        .map(|generator| generator.as_element().clone())
        .collect::<Vec<_>>();
    let image_group_generator_list = session.list(&image_group_generator_elements);
    let image_group = session.call_global("Group", &[image_group_generator_list.as_element()])?;

    let homomorphism_image_elements = gap_generators
        .iter_mut()
        .map(|generator| generator.as_element().clone())
        .collect::<Vec<_>>();
    let homomorphism_image_list = session.list(&homomorphism_image_elements);
    let homomorphism = session.call_global(
        "GroupHomomorphismByImages",
        &[
            group,
            image_group.as_element(),
            source_generators.as_element(),
            homomorphism_image_list.as_element(),
        ],
    )?;

    if session.is_fail(homomorphism.as_element()) {
        return Err(GLatticeError::HomomorphismByImagesFailed);
    }
    validate_group_homomorphism(session, homomorphism.as_element())?;

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
    session: &GapSession,
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
            let canonical_point = session.int(canonical_relation_id + 1);
            let on_points = session.global("OnPoints")?;
            let image_stabilizer = session.call_global(
                "Stabilizer",
                &[
                    relation_image_group,
                    &canonical_point,
                    on_points.as_element(),
                ],
            )?;
            let stabilizer = session.call_global(
                "PreImage",
                &[relation_action_homomorphism, image_stabilizer.as_element()],
            )?;
            let mut orbit_relations = Vec::with_capacity(relation_ids.len());
            let mut transporters = Vec::with_capacity(relation_ids.len());

            for &relation_id in &relation_ids {
                let relation = relations[relation_id];
                orbit_relations.push(relation);
                let target_point = session.int(relation_id + 1);
                let image_transporter = session.call_global(
                    "RepresentativeAction",
                    &[
                        relation_image_group,
                        &canonical_point,
                        &target_point,
                        on_points.as_element(),
                    ],
                )?;
                if session.is_fail(image_transporter.as_element()) {
                    return Err(GLatticeError::MissingTransporter {
                        canonical: canonical_representative,
                        target: relation,
                    });
                }
                let group_element = session.call_global(
                    "PreImagesRepresentative",
                    &[relation_action_homomorphism, image_transporter.as_element()],
                )?;
                if session.is_fail(group_element.as_element()) {
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

fn gap_error(error: impl fmt::Display) -> GLatticeError {
    GLatticeError::Gap(error.to_string())
}
