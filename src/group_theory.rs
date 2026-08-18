//! Finite-group and permutation-action operations backed by GAP.
//!
//! [`crate::group_theory::GapSubgroup`] is the public label used for subgroups
//! in GAP's subgroup enumeration. The remaining implementation in this module
//! composes `gap-sys` primitives for use by the crate's G-lattice
//! constructions; `gap-sys` itself remains responsible for interpreter access,
//! rooted values, generic calls, and Rust/GAP representation conversion.

use bitvec::prelude::*;
use gap_sys::{Gap, GapValue, GlobalGapGuard};
use std::collections::VecDeque;

/// A label for a subgroup in GAP's subgroup-lattice enumeration.
///
/// GAP's `LatticeSubgroups` organizes subgroups into conjugacy classes. This
/// label records a zero-based conjugacy class index and a zero-based element
/// index within that class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GapSubgroup {
    conjugacy_class: usize,
    class_element: usize,
}

/// A GAP homomorphism describing a permutation action and its image group.
pub(crate) struct GapAction {
    pub(crate) image_group: GapValue,
    pub(crate) homomorphism: GapValue,
}

/// The group-theoretic data used to construct a subgroup G-lattice.
pub(crate) struct SubgroupLatticeData {
    pub(crate) gap_lattice: GapValue,
    pub(crate) conjugacy_classes: GapValue,
    pub(crate) subgroup_list: GapValue,
    pub(crate) labels: Vec<GapSubgroup>,
    pub(crate) subgroups: Vec<GapValue>,
    pub(crate) structure_descriptions: Vec<String>,
    pub(crate) inclusion_relation: Vec<BitVec>,
    pub(crate) conjugation_generator_images: Vec<Vec<usize>>,
}

/// Orbits and orbit membership for a finite permutation action.
pub(crate) struct PointOrbits {
    pub(crate) point_to_orbit: Vec<usize>,
    pub(crate) orbits: Vec<PointOrbit>,
}

/// Stabilizer and transporter data for one orbit of a permutation action.
pub(crate) struct PointOrbit {
    pub(crate) canonical_point: usize,
    pub(crate) points: Vec<usize>,
    pub(crate) stabilizer: GapValue,
    pub(crate) transporters: Vec<PointTransporter>,
}

/// A chosen group element carrying an orbit representative to a target point.
pub(crate) struct PointTransporter {
    pub(crate) point: usize,
    pub(crate) group_element: GapValue,
}

#[derive(Debug)]
pub(crate) enum GroupTheoryError {
    Gap(String),
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
        element: usize,
        image: usize,
        len: usize,
    },
    DuplicatePermutationImage {
        generator: usize,
        first_element: usize,
        second_element: usize,
        image: usize,
    },
    HomomorphismByImagesFailed,
    SubgroupConjugateNotFound {
        generator: usize,
        subgroup: GapSubgroup,
    },
}

#[derive(Debug)]
pub(crate) enum PointOrbitError {
    Group(GroupTheoryError),
    MissingTransporter { canonical: usize, target: usize },
    MissingPreimage { canonical: usize, target: usize },
}

impl GapSubgroup {
    /// Constructs a zero-based subgroup-lattice label.
    pub fn new(conjugacy_class: usize, class_element: usize) -> Self {
        Self {
            conjugacy_class,
            class_element,
        }
    }

    /// Returns the zero-based conjugacy class index.
    pub fn conjugacy_class(&self) -> usize {
        self.conjugacy_class
    }

    /// Returns the zero-based index inside the conjugacy class.
    pub fn class_element(&self) -> usize {
        self.class_element
    }

    /// Returns the one-based conjugacy class index used by GAP.
    pub fn gap_conjugacy_class(&self) -> usize {
        self.conjugacy_class + 1
    }

    /// Returns the one-based class-element index used by GAP.
    pub fn gap_class_element(&self) -> usize {
        self.class_element + 1
    }
}

pub(crate) fn global_gap() -> Result<GlobalGapGuard, GroupTheoryError> {
    gap_sys::global().map_err(gap_error)
}

pub(crate) fn validate_finite_group(
    gap: &mut Gap,
    group: &GapValue,
) -> Result<(), GroupTheoryError> {
    let is_finite = call_global(gap, "IsFinite", &[group])?;
    if boolean(gap, &is_finite)? {
        Ok(())
    } else {
        Err(GroupTheoryError::GroupIsNotFinite)
    }
}

pub(crate) fn validate_group_homomorphism(
    gap: &mut Gap,
    homomorphism: &GapValue,
) -> Result<(), GroupTheoryError> {
    let is_homomorphism = call_global(gap, "IsGroupHomomorphism", &[homomorphism])?;
    if boolean(gap, &is_homomorphism)? {
        Ok(())
    } else {
        Err(GroupTheoryError::NotAGroupHomomorphism)
    }
}

pub(crate) fn generators_of_group(
    gap: &mut Gap,
    group: &GapValue,
) -> Result<GapValue, GroupTheoryError> {
    call_global(gap, "GeneratorsOfGroup", &[group])
}

pub(crate) fn homomorphism_image(
    gap: &mut Gap,
    homomorphism: &GapValue,
) -> Result<GapValue, GroupTheoryError> {
    call_global(gap, "Image", &[homomorphism])
}

pub(crate) fn generator_permutations_from_homomorphism(
    gap: &mut Gap,
    source_generators: &GapValue,
    homomorphism: &GapValue,
    degree: usize,
) -> Result<Vec<Vec<usize>>, GroupTheoryError> {
    let generator_count = gap.list_len(source_generators);
    (0..generator_count)
        .map(|generator| {
            let source_generator = get_list_elem(gap, source_generators, generator)?;
            let image = call_global(gap, "Image", &[homomorphism, &source_generator])?;
            gap.permutation_images_zero_based(&image, degree)
                .map_err(gap_error)
        })
        .collect()
}

pub(crate) fn validate_permutation(
    generator: usize,
    len: usize,
    image: &[usize],
) -> Result<(), GroupTheoryError> {
    if image.len() != len {
        return Err(GroupTheoryError::WrongPermutationLength {
            generator,
            expected: len,
            actual: image.len(),
        });
    }

    let mut seen = vec![None; len];
    for (element, &target) in image.iter().enumerate() {
        if target >= len {
            return Err(GroupTheoryError::PermutationImageOutOfBounds {
                generator,
                element,
                image: target,
                len,
            });
        }
        if let Some(first_element) = seen[target] {
            return Err(GroupTheoryError::DuplicatePermutationImage {
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

pub(crate) fn validate_generator_count(
    gap: &Gap,
    source_generators: &GapValue,
    actual: usize,
) -> Result<(), GroupTheoryError> {
    let expected = gap.list_len(source_generators);
    if actual == expected {
        Ok(())
    } else {
        Err(GroupTheoryError::GeneratorCountMismatch { expected, actual })
    }
}

pub(crate) fn action_from_generator_permutations(
    gap: &mut Gap,
    group: &GapValue,
    source_generators: &GapValue,
    generator_permutations: &[Vec<usize>],
    degree: usize,
) -> Result<GapAction, GroupTheoryError> {
    validate_generator_count(gap, source_generators, generator_permutations.len())?;

    let gap_generators = generator_permutations
        .iter()
        .map(|permutation| {
            gap.permutation_from_zero_based_images(permutation)
                .map_err(gap_error)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let image_group_generators = if gap_generators.is_empty() {
        vec![
            gap.permutation_from_zero_based_images(&(0..degree).collect::<Vec<_>>())
                .map_err(gap_error)?,
        ]
    } else {
        gap_generators.clone()
    };

    let image_group_generator_list = gap.list(&image_group_generators);
    let image_group = call_global(gap, "Group", &[&image_group_generator_list])?;

    let homomorphism_image_list = gap.list(&gap_generators);
    let homomorphism = call_global(
        gap,
        "GroupHomomorphismByImages",
        &[
            group,
            &image_group,
            source_generators,
            &homomorphism_image_list,
        ],
    )?;

    if gap.is_fail(&homomorphism) {
        return Err(GroupTheoryError::HomomorphismByImagesFailed);
    }
    validate_group_homomorphism(gap, &homomorphism)?;

    Ok(GapAction {
        image_group,
        homomorphism,
    })
}

pub(crate) fn point_orbits(
    gap: &mut Gap,
    point_count: usize,
    generator_permutations: &[Vec<usize>],
    image_group: &GapValue,
    action_homomorphism: &GapValue,
) -> Result<PointOrbits, PointOrbitError> {
    let point_to_orbit = point_orbit_ids(point_count, generator_permutations);
    let orbit_count = point_to_orbit
        .iter()
        .copied()
        .max()
        .map_or(0, |orbit| orbit + 1);
    let mut orbit_points = vec![Vec::new(); orbit_count];
    for (point, &orbit) in point_to_orbit.iter().enumerate() {
        orbit_points[orbit].push(point);
    }

    let on_points = global_obj(gap, "OnPoints").map_err(PointOrbitError::Group)?;
    let orbits = orbit_points
        .into_iter()
        .map(|points| {
            let canonical_point = points[0];
            let canonical_gap_point = gap.int((canonical_point + 1) as isize);
            let image_stabilizer = call_global(
                gap,
                "Stabilizer",
                &[image_group, &canonical_gap_point, &on_points],
            )
            .map_err(PointOrbitError::Group)?;
            let stabilizer =
                call_global(gap, "PreImage", &[action_homomorphism, &image_stabilizer])
                    .map_err(PointOrbitError::Group)?;

            let transporters = points
                .iter()
                .copied()
                .map(|point| {
                    let target_gap_point = gap.int((point + 1) as isize);
                    let image_transporter = call_global(
                        gap,
                        "RepresentativeAction",
                        &[
                            image_group,
                            &canonical_gap_point,
                            &target_gap_point,
                            &on_points,
                        ],
                    )
                    .map_err(PointOrbitError::Group)?;
                    if gap.is_fail(&image_transporter) {
                        return Err(PointOrbitError::MissingTransporter {
                            canonical: canonical_point,
                            target: point,
                        });
                    }
                    let group_element = call_global(
                        gap,
                        "PreImagesRepresentative",
                        &[action_homomorphism, &image_transporter],
                    )
                    .map_err(PointOrbitError::Group)?;
                    if gap.is_fail(&group_element) {
                        return Err(PointOrbitError::MissingPreimage {
                            canonical: canonical_point,
                            target: point,
                        });
                    }
                    Ok(PointTransporter {
                        point,
                        group_element,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(PointOrbit {
                canonical_point,
                points,
                stabilizer,
                transporters,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PointOrbits {
        point_to_orbit,
        orbits,
    })
}

pub(crate) fn subgroup_lattice_data(
    gap: &mut Gap,
    group: &GapValue,
) -> Result<SubgroupLatticeData, GroupTheoryError> {
    validate_finite_group(gap, group)?;
    let gap_lattice = call_global(gap, "LatticeSubgroups", &[group])?;
    let conjugacy_classes = call_global(gap, "ConjugacyClassesSubgroups", &[&gap_lattice])?;
    let (labels, subgroups) = subgroup_lattice_elements(gap, &conjugacy_classes)?;
    let structure_descriptions = subgroup_structure_descriptions(gap, &subgroups)?;
    let subgroup_list = gap.list(&subgroups);
    let inclusion_relation = subgroup_inclusion_relation(gap, &subgroups)?;
    let conjugation_generator_images =
        subgroup_conjugation_generator_images(gap, group, &subgroups, &subgroup_list, &labels)?;

    Ok(SubgroupLatticeData {
        gap_lattice,
        conjugacy_classes,
        subgroup_list,
        labels,
        subgroups,
        structure_descriptions,
        inclusion_relation,
        conjugation_generator_images,
    })
}

fn point_orbit_ids(point_count: usize, generator_permutations: &[Vec<usize>]) -> Vec<usize> {
    let mut point_to_orbit = vec![usize::MAX; point_count];
    let mut next_orbit = 0;

    for seed in 0..point_count {
        if point_to_orbit[seed] != usize::MAX {
            continue;
        }

        let mut queue = VecDeque::from([seed]);
        point_to_orbit[seed] = next_orbit;
        while let Some(point) = queue.pop_front() {
            for permutation in generator_permutations {
                let image = permutation[point];
                if point_to_orbit[image] == usize::MAX {
                    point_to_orbit[image] = next_orbit;
                    queue.push_back(image);
                }
            }
        }
        next_orbit += 1;
    }

    debug_assert!(point_to_orbit.iter().all(|&orbit| orbit != usize::MAX));
    point_to_orbit
}

fn subgroup_lattice_elements(
    gap: &mut Gap,
    conjugacy_classes: &GapValue,
) -> Result<(Vec<GapSubgroup>, Vec<GapValue>), GroupTheoryError> {
    let class_count = gap.list_len(conjugacy_classes);
    let mut labels = Vec::new();
    let mut subgroups = Vec::new();

    for conjugacy_class in 0..class_count {
        let class = get_list_elem(gap, conjugacy_classes, conjugacy_class)?;
        let class_size = call_global(gap, "Size", &[&class])?;
        let class_size = integer_usize(gap, &class_size)?;
        for class_element in 0..class_size {
            let n = gap.int((class_element + 1) as isize);
            let subgroup = call_global(gap, "ClassElementLattice", &[&class, &n])?;
            labels.push(GapSubgroup::new(conjugacy_class, class_element));
            subgroups.push(subgroup);
        }
    }
    Ok((labels, subgroups))
}

fn subgroup_structure_descriptions(
    gap: &mut Gap,
    subgroups: &[GapValue],
) -> Result<Vec<String>, GroupTheoryError> {
    subgroups
        .iter()
        .map(|subgroup| {
            let description = call_global(gap, "StructureDescription", &[subgroup])?;
            Ok(description.as_unrooted().to_string())
        })
        .collect()
}

fn subgroup_inclusion_relation(
    gap: &mut Gap,
    subgroups: &[GapValue],
) -> Result<Vec<BitVec>, GroupTheoryError> {
    let n = subgroups.len();
    let mut relation = vec![BitVec::repeat(false, n); n];
    for (lower, lower_subgroup) in subgroups.iter().enumerate() {
        for (upper, upper_subgroup) in subgroups.iter().enumerate() {
            let is_subgroup = call_global(gap, "IsSubgroup", &[upper_subgroup, lower_subgroup])?;
            relation[lower].set(upper, boolean(gap, &is_subgroup)?);
        }
    }
    Ok(relation)
}

fn subgroup_conjugation_generator_images(
    gap: &mut Gap,
    group: &GapValue,
    subgroups: &[GapValue],
    subgroup_list: &GapValue,
    labels: &[GapSubgroup],
) -> Result<Vec<Vec<usize>>, GroupTheoryError> {
    let generators = generators_of_group(gap, group)?;
    let generator_count = gap.list_len(&generators);
    let mut result = Vec::with_capacity(generator_count);

    for generator in 0..generator_count {
        let gap_generator = get_list_elem(gap, &generators, generator)?;
        let mut image = Vec::with_capacity(subgroups.len());
        for (subgroup_id, subgroup) in subgroups.iter().enumerate() {
            let conjugate = call_global(gap, "OnPoints", &[subgroup, &gap_generator])?;
            let position = call_global(gap, "Position", &[subgroup_list, &conjugate])?;
            if gap.is_fail(&position) {
                return Err(GroupTheoryError::SubgroupConjugateNotFound {
                    generator,
                    subgroup: labels[subgroup_id],
                });
            }
            let position = integer_usize(gap, &position)?;
            let image_id =
                position
                    .checked_sub(1)
                    .ok_or(GroupTheoryError::SubgroupConjugateNotFound {
                        generator,
                        subgroup: labels[subgroup_id],
                    })?;
            image.push(image_id);
        }
        result.push(image);
    }
    Ok(result)
}

fn global_obj(gap: &mut Gap, name: &str) -> Result<GapValue, GroupTheoryError> {
    gap.get_global(name).map_err(gap_error)
}

fn call_global(
    gap: &mut Gap,
    name: &str,
    args: &[&GapValue],
) -> Result<GapValue, GroupTheoryError> {
    gap.call_global(name, args).map_err(gap_error)
}

fn get_list_elem(gap: &mut Gap, list: &GapValue, idx: usize) -> Result<GapValue, GroupTheoryError> {
    gap.list_get(list, idx).map_err(gap_error)
}

fn integer_usize(gap: &mut Gap, element: &GapValue) -> Result<usize, GroupTheoryError> {
    gap.to_usize(element).map_err(gap_error)
}

fn boolean(gap: &mut Gap, element: &GapValue) -> Result<bool, GroupTheoryError> {
    gap.to_bool(element).map_err(gap_error)
}

fn gap_error(error: impl std::fmt::Display) -> GroupTheoryError {
    GroupTheoryError::Gap(error.to_string())
}
