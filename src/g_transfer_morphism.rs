//! Transfer-system maps between finite lattices with group actions.
//!
//! For a monotone map `u: L -> M`, [`crate::g_transfer_morphism::pushforward`] produces the least
//! target-invariant transfer system containing the image arrows, while
//! [`crate::g_transfer_morphism::pullback`] produces its right adjoint on invariant transfer systems.
//! Applying this pair to the subgroup-image and subgroup-preimage maps of a
//! group homomorphism `f: G -> H` gives Rubin's four functors:
//!
//! | subgroup map | pushforward | pullback |
//! | --- | --- | --- |
//! | `K |-> f(K)` | `f_L` | `f_R^{-1}` |
//! | `K |-> f^{-1}(K)` | `f_L^{-1}` | `f_R` |
//!
//! In particular, Rubin's restriction `f_L^{-1}` is pushforward along the
//! subgroup-preimage map. It is not the generated inverse image along the
//! subgroup-image map.
//!
//! The generic action-aware operations below need not commute with composition
//! of arbitrary monotone maps. Rubin's four operations do commute with
//! composition of group homomorphisms.

use crate::bitvec_utils::is_subset;
use crate::g_lattice::{GTransferLattice, GTransferSystem, GTransferUniverse, RawGTransferSystem};
use crate::group_theory::GapSubgroup;
use crate::morphism::{MonotoneMap, PosetMap};
use crate::poset::{Edge, ElementId};
use crate::subgroup_morphism::SubgroupMaps;
use bitvec::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Errors produced while applying or materializing an equivariant transfer map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GTransferMapError {
    /// The input expected to lie over the monotone map's domain does not.
    DomainMismatch,
    /// The input expected to lie over the monotone map's codomain does not.
    CodomainMismatch,
    /// A proper source relation did not map to an indexed codomain relation.
    InvalidImageRelation {
        /// The source relation.
        source: Edge,
        /// Its purported image.
        image: Edge,
    },
    /// A computed pushforward was absent from an enumerated codomain order.
    PushforwardImageMissing,
    /// A computed pullback was absent from an enumerated domain order.
    PullbackImageMissing,
    /// An operation called induction requires an injective group homomorphism.
    NotInjective,
    /// Inflation and fixed points require a surjective group homomorphism.
    NotSurjective,
}

impl fmt::Display for GTransferMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainMismatch => write!(
                formatter,
                "the supplied G-transfer data does not use the monotone map's domain coordinates"
            ),
            Self::CodomainMismatch => write!(
                formatter,
                "the supplied G-transfer data does not use the monotone map's codomain coordinates"
            ),
            Self::InvalidImageRelation { source, image } => write!(
                formatter,
                "source relation {} <= {} maps to unindexed codomain relation {} <= {}",
                source.from, source.to, image.from, image.to
            ),
            Self::PushforwardImageMissing => write!(
                formatter,
                "the computed G-transfer pushforward is absent from the enumerated codomain order"
            ),
            Self::PullbackImageMissing => write!(
                formatter,
                "the computed G-transfer pullback is absent from the enumerated domain order"
            ),
            Self::NotInjective => write!(formatter, "induction requires an injective map"),
            Self::NotSurjective => write!(
                formatter,
                "inflation and fixed points require a surjective map"
            ),
        }
    }
}

impl std::error::Error for GTransferMapError {}

/// Computes the invariant transfer-system pushforward along a monotone map.
///
/// The result is the least transfer system fixed by the codomain action that
/// contains the image of every relation in `system`. The supplied codomain
/// universe fixes the action, relation-orbit coordinates, and ownership of the
/// result. This pointwise operation does not enumerate transfer systems.
/// For arbitrary monotone maps it need not commute with composition.
pub fn pushforward<A, B, M>(
    map: &M,
    system: &GTransferSystem<A>,
    codomain: &Arc<GTransferUniverse<B>>,
) -> Result<GTransferSystem<B>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_domain(map, system.universe())?;
    validate_codomain(map, codomain)?;

    let mut generators = BitVec::repeat(false, codomain.relation_orbit_labels().len());
    for source_orbit_id in system.raw().orbit_arrows().iter_ones() {
        let relations = system
            .universe()
            .relation_orbit_relations(source_orbit_id)
            .expect("selected relation-orbit id should belong to its universe");
        for &source in relations {
            let image = image_edge(map, source);
            if image.is_identity() {
                continue;
            }
            let Some(target_orbit_id) = codomain.relation_orbit_label_id(image) else {
                return Err(GTransferMapError::InvalidImageRelation { source, image });
            };
            generators.set(target_orbit_id, true);
        }
    }

    let raw = RawGTransferSystem::new(codomain.close_orbit_arrows(&generators));
    Ok(GTransferSystem::new(raw, Arc::clone(codomain)))
}

/// Computes the right-adjoint pullback on invariant transfer systems.
///
/// A source relation orbit belongs to the result precisely when all of its
/// conjugates and all of their restrictions map to relations in `system`.
/// Equivalently, this is the greatest source-invariant transfer system whose
/// pushforward is contained in `system`. The supplied domain universe fixes
/// the result's action, coordinates, and ownership.
/// For arbitrary monotone maps it need not commute with composition.
pub fn pullback<A, B, M>(
    map: &M,
    system: &GTransferSystem<B>,
    domain: &Arc<GTransferUniverse<A>>,
) -> Result<GTransferSystem<A>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_codomain(map, system.universe())?;
    validate_domain(map, domain)?;

    let requirements = orbit_requirements(map, domain, system.universe())?;
    let raw = raw_pullback(system.raw(), domain, &requirements);
    debug_assert_eq!(
        domain.close_orbit_arrows(raw.orbit_arrows()),
        raw.orbit_arrows().clone()
    );
    Ok(GTransferSystem::new(raw, Arc::clone(domain)))
}

/// Computes Rubin's `f_L` using the subgroup-image map of `f`.
///
/// This construction is defined for every group homomorphism. For an
/// injective homomorphism, [`induction`] is its conventional name.
pub fn image_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    codomain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    validate_rubin_codomain_action(f, codomain)?;
    pushforward(f.image_map(), system, codomain)
}

/// Computes Rubin's `f_R^{-1}`, right adjoint to [`image_pushforward`].
pub fn image_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    domain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    validate_rubin_domain_action(f, domain)?;
    pullback(f.image_map(), system, domain)
}

/// Computes Rubin's `f_L^{-1}` using the subgroup-preimage map of `f`.
///
/// This is restriction along an arbitrary group homomorphism. For a quotient
/// homomorphism it is inflation.
pub fn preimage_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    domain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    validate_rubin_domain_action(f, domain)?;
    pushforward(f.preimage_map(), system, domain)
}

/// Computes Rubin's `f_R`, right adjoint to [`preimage_pushforward`].
///
/// This is coinduction along an arbitrary group homomorphism. For a quotient
/// homomorphism it is fixed points.
pub fn preimage_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    codomain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    validate_rubin_codomain_action(f, codomain)?;
    pullback(f.preimage_map(), system, codomain)
}

/// Induces a transfer system along an injective group homomorphism.
///
/// This is the conventional injective case of [`image_pushforward`].
pub fn induction(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    codomain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    if !f.is_injective() {
        return Err(GTransferMapError::NotInjective);
    }
    image_pushforward(f, system, codomain)
}

/// Restricts a transfer system along an arbitrary group homomorphism.
///
/// This is an ergonomic name for [`preimage_pushforward`].
pub fn restriction(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    domain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    preimage_pushforward(f, system, domain)
}

/// Coinduces a transfer system along an arbitrary group homomorphism.
///
/// This is an ergonomic name for [`preimage_pullback`].
pub fn coinduction(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    codomain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    preimage_pullback(f, system, codomain)
}

/// Inflates a transfer system along a surjective group homomorphism.
///
/// Inflation is exactly transfer pushforward along subgroup preimage.
pub fn inflation(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    domain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pushforward(f, system, domain)
}

/// Takes fixed points along a surjective group homomorphism.
///
/// This is the right adjoint to [`inflation`], realized by pullback along the
/// subgroup-preimage map.
pub fn fixed_points(
    f: &SubgroupMaps,
    system: &GTransferSystem<GapSubgroup>,
    codomain: &Arc<GTransferUniverse<GapSubgroup>>,
) -> Result<GTransferSystem<GapSubgroup>, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pullback(f, system, codomain)
}

/// Materializes Rubin's `f_L` between containment lattices.
pub fn image_pushforward_containment_map(
    f: &SubgroupMaps,
    domain: &GTransferLattice<GapSubgroup>,
    codomain: &GTransferLattice<GapSubgroup>,
) -> Result<PosetMap<GTransferSystem<GapSubgroup>, GTransferSystem<GapSubgroup>>, GTransferMapError>
{
    validate_rubin_domain_action(f, domain.universe())?;
    validate_rubin_codomain_action(f, codomain.universe())?;
    pushforward_containment_map(f.image_map(), domain, codomain)
}

/// Materializes Rubin's `f_R^{-1}` between containment lattices.
pub fn image_pullback_containment_map(
    f: &SubgroupMaps,
    codomain: &GTransferLattice<GapSubgroup>,
    domain: &GTransferLattice<GapSubgroup>,
) -> Result<PosetMap<GTransferSystem<GapSubgroup>, GTransferSystem<GapSubgroup>>, GTransferMapError>
{
    validate_rubin_codomain_action(f, codomain.universe())?;
    validate_rubin_domain_action(f, domain.universe())?;
    pullback_containment_map(f.image_map(), codomain, domain)
}

/// Materializes Rubin's `f_L^{-1}` between containment lattices.
pub fn preimage_pushforward_containment_map(
    f: &SubgroupMaps,
    codomain: &GTransferLattice<GapSubgroup>,
    domain: &GTransferLattice<GapSubgroup>,
) -> Result<PosetMap<GTransferSystem<GapSubgroup>, GTransferSystem<GapSubgroup>>, GTransferMapError>
{
    validate_rubin_codomain_action(f, codomain.universe())?;
    validate_rubin_domain_action(f, domain.universe())?;
    pushforward_containment_map(f.preimage_map(), codomain, domain)
}

/// Materializes Rubin's `f_R` between containment lattices.
pub fn preimage_pullback_containment_map(
    f: &SubgroupMaps,
    domain: &GTransferLattice<GapSubgroup>,
    codomain: &GTransferLattice<GapSubgroup>,
) -> Result<PosetMap<GTransferSystem<GapSubgroup>, GTransferSystem<GapSubgroup>>, GTransferMapError>
{
    validate_rubin_domain_action(f, domain.universe())?;
    validate_rubin_codomain_action(f, codomain.universe())?;
    pullback_containment_map(f.preimage_map(), domain, codomain)
}

/// Constructs the pushforward map between containment lattices.
///
/// Together with [`pullback_containment_map`], the returned map is the left
/// adjoint. Endpoint labels are user-facing [`GTransferSystem`] values.
pub fn pushforward_containment_map<A, B, M>(
    map: &M,
    domain: &GTransferLattice<A>,
    codomain: &GTransferLattice<B>,
) -> Result<PosetMap<GTransferSystem<A>, GTransferSystem<B>>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_domain(map, domain.universe())?;
    validate_codomain(map, codomain.universe())?;

    let orbit_images = orbit_image_masks(map, domain.universe(), codomain.universe())?;
    let target_ids = target_system_ids(codomain.raw_lattice().elements());
    let mut closure_cache = HashMap::<BitVec, RawGTransferSystem>::new();
    let mut images = Vec::with_capacity(domain.size());

    for source in domain.raw_lattice().elements() {
        let mut generators =
            BitVec::repeat(false, codomain.universe().relation_orbit_labels().len());
        for source_orbit_id in source.orbit_arrows().iter_ones() {
            generators |= &orbit_images[source_orbit_id];
        }
        let image = closure_cache.entry(generators.clone()).or_insert_with(|| {
            RawGTransferSystem::new(codomain.universe().close_orbit_arrows(&generators))
        });
        images.push(
            target_ids
                .get(image)
                .copied()
                .ok_or(GTransferMapError::PushforwardImageMissing)?,
        );
    }

    Ok(PosetMap::from_validated(
        Arc::new(domain.to_system_poset()),
        Arc::new(codomain.to_system_poset()),
        images,
    ))
}

/// Constructs the right-adjoint pullback map between containment lattices.
///
/// The first lattice lies over the monotone map's codomain and is therefore
/// the domain of the returned map. Endpoint labels are user-facing
/// [`GTransferSystem`] values.
pub fn pullback_containment_map<A, B, M>(
    map: &M,
    codomain: &GTransferLattice<B>,
    domain: &GTransferLattice<A>,
) -> Result<PosetMap<GTransferSystem<B>, GTransferSystem<A>>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_codomain(map, codomain.universe())?;
    validate_domain(map, domain.universe())?;

    let requirements = orbit_requirements(map, domain.universe(), codomain.universe())?;
    let target_ids = target_system_ids(domain.raw_lattice().elements());
    let images = codomain
        .raw_lattice()
        .elements()
        .iter()
        .map(|source| {
            let image = raw_pullback(source, domain.universe(), &requirements);
            target_ids
                .get(&image)
                .copied()
                .ok_or(GTransferMapError::PullbackImageMissing)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PosetMap::from_validated(
        Arc::new(codomain.to_system_poset()),
        Arc::new(domain.to_system_poset()),
        images,
    ))
}

fn orbit_image_masks<A, B, M>(
    map: &M,
    domain: &GTransferUniverse<A>,
    codomain: &GTransferUniverse<B>,
) -> Result<Vec<BitVec>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    (0..domain.relation_orbit_labels().len())
        .map(|source_orbit_id| {
            let mut images = BitVec::repeat(false, codomain.relation_orbit_labels().len());
            for &source in domain
                .relation_orbit_relations(source_orbit_id)
                .expect("relation-orbit label should index its universe")
            {
                let image = image_edge(map, source);
                if image.is_identity() {
                    continue;
                }
                let Some(target_orbit_id) = codomain.relation_orbit_label_id(image) else {
                    return Err(GTransferMapError::InvalidImageRelation { source, image });
                };
                images.set(target_orbit_id, true);
            }
            Ok(images)
        })
        .collect()
}

fn orbit_requirements<A, B, M>(
    map: &M,
    domain: &GTransferUniverse<A>,
    codomain: &GTransferUniverse<B>,
) -> Result<Vec<BitVec>, GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let lattice = domain.lattice();
    (0..domain.relation_orbit_labels().len())
        .map(|source_orbit_id| {
            let mut requirements = BitVec::repeat(false, codomain.relation_orbit_labels().len());
            for &source in domain
                .relation_orbit_relations(source_orbit_id)
                .expect("relation-orbit label should index its universe")
            {
                for restriction_target in
                    lattice.as_poset().relation_matrix_transpose()[source.to].iter_ones()
                {
                    let restriction_source = lattice.meet_id(source.from, restriction_target);
                    let restricted = Edge::new(restriction_source, restriction_target);
                    let image = image_edge(map, restricted);
                    if image.is_identity() {
                        continue;
                    }
                    let Some(target_orbit_id) = codomain.relation_orbit_label_id(image) else {
                        return Err(GTransferMapError::InvalidImageRelation {
                            source: restricted,
                            image,
                        });
                    };
                    requirements.set(target_orbit_id, true);
                }
            }
            Ok(requirements)
        })
        .collect()
}

fn raw_pullback<A>(
    source: &RawGTransferSystem,
    domain: &GTransferUniverse<A>,
    requirements: &[BitVec],
) -> RawGTransferSystem {
    let mut arrows = BitVec::repeat(false, domain.relation_orbit_labels().len());
    for (source_orbit_id, required_orbits) in requirements.iter().enumerate() {
        if is_subset(required_orbits, source.orbit_arrows()) {
            arrows.set(source_orbit_id, true);
        }
    }
    RawGTransferSystem::new(arrows)
}

fn target_system_ids(systems: &[RawGTransferSystem]) -> HashMap<&RawGTransferSystem, ElementId> {
    systems
        .iter()
        .enumerate()
        .map(|(id, raw)| (raw, id))
        .collect()
}

fn image_edge<A, B, M>(map: &M, edge: Edge) -> Edge
where
    M: MonotoneMap<A, B> + ?Sized,
{
    Edge::new(map.images()[edge.from], map.images()[edge.to])
}

fn validate_domain<A, B, M>(
    map: &M,
    universe: &GTransferUniverse<A>,
) -> Result<(), GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    if map
        .domain_poset()
        .shares_order_coordinates_with(universe.lattice().as_poset())
    {
        Ok(())
    } else {
        Err(GTransferMapError::DomainMismatch)
    }
}

fn validate_codomain<A, B, M>(
    map: &M,
    universe: &GTransferUniverse<B>,
) -> Result<(), GTransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    if map
        .codomain_poset()
        .shares_order_coordinates_with(universe.lattice().as_poset())
    {
        Ok(())
    } else {
        Err(GTransferMapError::CodomainMismatch)
    }
}

fn validate_rubin_domain_action(
    f: &SubgroupMaps,
    universe: &GTransferUniverse<GapSubgroup>,
) -> Result<(), GTransferMapError> {
    if Arc::ptr_eq(
        f.domain().g_lattice().action_coordinates(),
        universe.action_coordinates(),
    ) {
        Ok(())
    } else {
        Err(GTransferMapError::DomainMismatch)
    }
}

fn validate_rubin_codomain_action(
    f: &SubgroupMaps,
    universe: &GTransferUniverse<GapSubgroup>,
) -> Result<(), GTransferMapError> {
    if Arc::ptr_eq(
        f.codomain().g_lattice().action_coordinates(),
        universe.action_coordinates(),
    ) {
        Ok(())
    } else {
        Err(GTransferMapError::CodomainMismatch)
    }
}
