//! Transfer-system maps between finite lattices with group actions.
//!
//! These are the action-invariant analogues of the operations in
//! [`crate::transfer_morphism`]. For a monotone map `u: L -> M`,
//! [`pushforward`] maps all arrows, closes under the transfer axioms, and adds
//! their whole target orbits. [`pullback`] is its right adjoint on invariant
//! transfer systems: it returns the greatest invariant source system whose
//! pushforward is contained in the given target system.
//!
//! A group homomorphism `f: G -> H` gives two monotone maps between subgroup
//! lattices: direct image sends `K <= G` to `f(K) <= H`, and inverse image
//! sends `J <= H` to `f^-1(J) <= G`. Applying pushforward and pullback to these
//! two maps gives Rubin's four functors:
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
use crate::g_lattice::{GLattice, GTransferLattice, GTransferSystem, GTransferUniverse};
use crate::morphism::{MonotoneMap, PosetMap};
use crate::poset::{Edge, ElementId};
use crate::subgroup_morphism::SubgroupMaps;
use bitvec::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

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
/// The result is the least transfer system on `codomain` fixed by its action
/// that contains the image of every relation in `system`. The map's codomain
/// must be the underlying lattice of `codomain`.
/// For arbitrary monotone maps it need not commute with composition.
pub fn pushforward<M>(
    map: &M,
    system: &GTransferSystem,
    codomain: &GLattice,
) -> Result<GTransferSystem, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let codomain = codomain.transfer_universe();
    validate_domain(map, system.universe())?;
    validate_codomain(map, codomain)?;

    let mut generators = BitVec::repeat(false, codomain.relation_orbit_labels().len());
    for source_orbit_id in system.orbit_arrows().iter_ones() {
        for &source in system.universe().relation_orbit_relations(source_orbit_id) {
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

    Ok(GTransferSystem::new(
        Rc::clone(codomain),
        codomain.close(&generators),
    ))
}

/// Computes the right-adjoint pullback on invariant transfer systems.
///
/// A source relation orbit belongs to the result precisely when all of its
/// conjugates and all of their restrictions map to relations in `system`.
/// Equivalently, this is the greatest invariant transfer system on `domain`
/// whose pushforward is contained in `system`. The map's domain must be the
/// underlying lattice of `domain`.
/// For arbitrary monotone maps it need not commute with composition.
pub fn pullback<M>(
    map: &M,
    system: &GTransferSystem,
    domain: &GLattice,
) -> Result<GTransferSystem, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let domain = domain.transfer_universe();
    validate_codomain(map, system.universe())?;
    validate_domain(map, domain)?;

    let requirements = orbit_requirements(map, domain, system.universe())?;
    let arrows = raw_pullback(system.orbit_arrows(), &requirements);
    debug_assert_eq!(domain.close(&arrows), arrows);
    Ok(GTransferSystem::new(Rc::clone(domain), arrows))
}

/// Computes Rubin's `f_L` using the subgroup-image map of `f`.
///
/// This construction is defined for every group homomorphism. For an
/// injective homomorphism, [`induction`] is its conventional name.
pub fn image_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    pushforward(f.image_map(), system, f.codomain())
}

/// Computes Rubin's `f_R^{-1}`, right adjoint to [`image_pushforward`].
pub fn image_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    pullback(f.image_map(), system, f.domain())
}

/// Computes Rubin's `f_L^{-1}` using the subgroup-preimage map of `f`.
///
/// This is restriction along an arbitrary group homomorphism. For a quotient
/// homomorphism it is inflation.
pub fn preimage_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    pushforward(f.preimage_map(), system, f.domain())
}

/// Computes Rubin's `f_R`, right adjoint to [`preimage_pushforward`].
///
/// This is coinduction along an arbitrary group homomorphism. For a quotient
/// homomorphism it is fixed points.
pub fn preimage_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    pullback(f.preimage_map(), system, f.codomain())
}

/// Induces a transfer system along an injective group homomorphism.
///
/// This is the conventional injective case of [`image_pushforward`].
pub fn induction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_injective() {
        return Err(GTransferMapError::NotInjective);
    }
    image_pushforward(f, system)
}

/// Restricts a transfer system along an arbitrary group homomorphism.
///
/// This is an ergonomic name for [`preimage_pushforward`].
pub fn restriction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    preimage_pushforward(f, system)
}

/// Coinduces a transfer system along an arbitrary group homomorphism.
///
/// This is an ergonomic name for [`preimage_pullback`].
pub fn coinduction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    preimage_pullback(f, system)
}

/// Inflates a transfer system along a surjective group homomorphism.
///
/// Inflation is exactly transfer pushforward along subgroup preimage.
pub fn inflation(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pushforward(f, system)
}

/// Takes fixed points along a surjective group homomorphism.
///
/// This is the right adjoint to [`inflation`], realized by pullback along the
/// subgroup-preimage map.
pub fn fixed_points(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pullback(f, system)
}

impl SubgroupMaps {
    /// Rubin's `f_L`; see [`image_pushforward`].
    pub fn image_pushforward(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        image_pushforward(self, system)
    }

    /// Rubin's `f_R^{-1}`; see [`image_pullback`].
    pub fn image_pullback(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        image_pullback(self, system)
    }

    /// Rubin's `f_L^{-1}`; see [`preimage_pushforward`].
    pub fn preimage_pushforward(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        preimage_pushforward(self, system)
    }

    /// Rubin's `f_R`; see [`preimage_pullback`].
    pub fn preimage_pullback(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        preimage_pullback(self, system)
    }

    /// Induction along an injective homomorphism; see [`induction`].
    pub fn induction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        induction(self, system)
    }

    /// Restriction along a homomorphism; see [`restriction`].
    pub fn restriction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        restriction(self, system)
    }

    /// Coinduction along a homomorphism; see [`coinduction`].
    pub fn coinduction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        coinduction(self, system)
    }

    /// Inflation along a surjective homomorphism; see [`inflation`].
    pub fn inflation(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        inflation(self, system)
    }

    /// Fixed points along a surjective homomorphism; see [`fixed_points`].
    pub fn fixed_points(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        fixed_points(self, system)
    }
}

/// Materializes Rubin's `f_L` between containment lattices.
pub fn image_pushforward_containment_map(
    f: &SubgroupMaps,
    domain: &GTransferLattice,
    codomain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError> {
    validate_rubin_domain_action(f, domain.universe())?;
    validate_rubin_codomain_action(f, codomain.universe())?;
    pushforward_containment_map(f.image_map(), domain, codomain)
}

/// Materializes Rubin's `f_R^{-1}` between containment lattices.
pub fn image_pullback_containment_map(
    f: &SubgroupMaps,
    codomain: &GTransferLattice,
    domain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError> {
    validate_rubin_codomain_action(f, codomain.universe())?;
    validate_rubin_domain_action(f, domain.universe())?;
    pullback_containment_map(f.image_map(), codomain, domain)
}

/// Materializes Rubin's `f_L^{-1}` between containment lattices.
pub fn preimage_pushforward_containment_map(
    f: &SubgroupMaps,
    codomain: &GTransferLattice,
    domain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError> {
    validate_rubin_codomain_action(f, codomain.universe())?;
    validate_rubin_domain_action(f, domain.universe())?;
    pushforward_containment_map(f.preimage_map(), codomain, domain)
}

/// Materializes Rubin's `f_R` between containment lattices.
pub fn preimage_pullback_containment_map(
    f: &SubgroupMaps,
    domain: &GTransferLattice,
    codomain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError> {
    validate_rubin_domain_action(f, domain.universe())?;
    validate_rubin_codomain_action(f, codomain.universe())?;
    pullback_containment_map(f.preimage_map(), domain, codomain)
}

/// Constructs the pushforward map between containment lattices.
///
/// Together with [`pullback_containment_map`], the returned map is the left
/// adjoint. Its endpoints are the lattices of G-transfer systems themselves.
pub fn pushforward_containment_map<M>(
    map: &M,
    domain: &GTransferLattice,
    codomain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    validate_domain(map, domain.universe())?;
    validate_codomain(map, codomain.universe())?;

    let target = codomain.universe();
    let orbit_images = orbit_image_masks(map, domain.universe(), target)?;
    let mut closure_cache = HashMap::<BitVec, Option<ElementId>>::new();
    let images = domain
        .systems()
        .iter()
        .map(|source| {
            let mut generators = BitVec::repeat(false, target.relation_orbit_labels().len());
            for source_orbit_id in source.orbit_arrows().iter_ones() {
                generators |= &orbit_images[source_orbit_id];
            }
            closure_cache
                .entry(generators.clone())
                .or_insert_with(|| {
                    codomain.id_of(&GTransferSystem::new(
                        Rc::clone(target),
                        target.close(&generators),
                    ))
                })
                .ok_or(GTransferMapError::PushforwardImageMissing)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PosetMap::from_validated(
        domain.as_poset().clone(),
        codomain.as_poset().clone(),
        images,
    ))
}

/// Constructs the right-adjoint pullback map between containment lattices.
///
/// The first lattice lies over the monotone map's codomain and is therefore
/// the domain of the returned map.
pub fn pullback_containment_map<M>(
    map: &M,
    codomain: &GTransferLattice,
    domain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    validate_codomain(map, codomain.universe())?;
    validate_domain(map, domain.universe())?;

    let requirements = orbit_requirements(map, domain.universe(), codomain.universe())?;
    let images = codomain
        .systems()
        .iter()
        .map(|source| {
            let arrows = raw_pullback(source.orbit_arrows(), &requirements);
            domain
                .id_of(&GTransferSystem::new(Rc::clone(domain.universe()), arrows))
                .ok_or(GTransferMapError::PullbackImageMissing)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PosetMap::from_validated(
        codomain.as_poset().clone(),
        domain.as_poset().clone(),
        images,
    ))
}

fn orbit_image_masks<M>(
    map: &M,
    domain: &GTransferUniverse,
    codomain: &GTransferUniverse,
) -> Result<Vec<BitVec>, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    (0..domain.relation_orbit_labels().len())
        .map(|source_orbit_id| {
            let mut images = BitVec::repeat(false, codomain.relation_orbit_labels().len());
            for &source in domain.relation_orbit_relations(source_orbit_id) {
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

fn orbit_requirements<M>(
    map: &M,
    domain: &GTransferUniverse,
    codomain: &GTransferUniverse,
) -> Result<Vec<BitVec>, GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let lattice = domain.lattice();
    (0..domain.relation_orbit_labels().len())
        .map(|source_orbit_id| {
            let mut requirements = BitVec::repeat(false, codomain.relation_orbit_labels().len());
            for &source in domain.relation_orbit_relations(source_orbit_id) {
                for restriction_target in lattice.relation_matrix_transpose()[source.to].iter_ones()
                {
                    let restriction_source = lattice.meet(source.from, restriction_target);
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

fn raw_pullback(source: &BitVec, requirements: &[BitVec]) -> BitVec {
    requirements
        .iter()
        .map(|required_orbits| is_subset(required_orbits, source))
        .collect()
}

fn image_edge<M>(map: &M, edge: Edge) -> Edge
where
    M: MonotoneMap + ?Sized,
{
    Edge::new(map.images()[edge.from], map.images()[edge.to])
}

fn validate_domain<M>(map: &M, universe: &GTransferUniverse) -> Result<(), GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    if map.domain_poset() == universe.lattice().as_poset() {
        Ok(())
    } else {
        Err(GTransferMapError::DomainMismatch)
    }
}

fn validate_codomain<M>(map: &M, universe: &GTransferUniverse) -> Result<(), GTransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    if map.codomain_poset() == universe.lattice().as_poset() {
        Ok(())
    } else {
        Err(GTransferMapError::CodomainMismatch)
    }
}

fn validate_rubin_domain_action(
    f: &SubgroupMaps,
    universe: &GTransferUniverse,
) -> Result<(), GTransferMapError> {
    if Rc::ptr_eq(
        f.domain().action_coordinates(),
        universe.action_coordinates(),
    ) {
        Ok(())
    } else {
        Err(GTransferMapError::DomainMismatch)
    }
}

fn validate_rubin_codomain_action(
    f: &SubgroupMaps,
    universe: &GTransferUniverse,
) -> Result<(), GTransferMapError> {
    if Rc::ptr_eq(
        f.codomain().action_coordinates(),
        universe.action_coordinates(),
    ) {
        Ok(())
    } else {
        Err(GTransferMapError::CodomainMismatch)
    }
}
