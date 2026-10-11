//! G-transfer systems along maps between finite lattices with group actions.
//!
//! These are the action-invariant analogues of the operations in
//! [`crate::transfer_morphism`]. For a monotone map `u: L -> M` of the
//! underlying lattices, [`GTransferSystem::pushforward`] maps every relation,
//! closes under the transfer axioms, and adds the whole orbits of the image
//! relations. [`GTransferSystem::pullback`] is its right adjoint: it returns the
//! greatest invariant system on `L` whose pushforward is contained in the given
//! one. For arbitrary monotone maps these need not respect composition.
//!
//! A group homomorphism `f: G -> H` gives two monotone maps between subgroup
//! lattices: direct image sends `K <= G` to `f(K) <= H`, and inverse image sends
//! `J <= H` to `f^-1(J) <= G`. Pushing forward and pulling back along these two
//! maps gives Rubin's four functors, which are methods of [`SubgroupMaps`] and do
//! respect composition of homomorphisms:
//!
//! | subgroup map | pushforward | pullback |
//! | --- | --- | --- |
//! | `K |-> f(K)` | `f_L` ([`SubgroupMaps::image_pushforward`]) | `f_R^{-1}` ([`SubgroupMaps::image_pullback`]) |
//! | `J |-> f^{-1}(J)` | `f_L^{-1}` ([`SubgroupMaps::preimage_pushforward`]) | `f_R` ([`SubgroupMaps::preimage_pullback`]) |
//!
//! In particular, Rubin's restriction `f_L^{-1}` is the pushforward along the
//! inverse-image map, not the generated inverse image along the direct-image
//! map. [`SubgroupMaps`] also provides the classical names: induction,
//! restriction, coinduction, inflation and fixed points.
//!
//! The free functions in this module package these operations as maps between
//! lattices of G-transfer systems, for example
//! [`image_pushforward_containment_map`].

use crate::bitvec_utils::is_subset;
use crate::g_lattice::GLattice;
use crate::g_transfer_lattice::{GTransferLattice, GTransferSystem, GTransferUniverse};
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
    /// An input does not lie over the domain of the map.
    ///
    /// For [`SubgroupMaps`] methods, and for the containment-map functions
    /// that take one, "domain" means the domain `f.domain()` of the group
    /// homomorphism `f`, not of the induced monotone map.
    DomainMismatch,
    /// An input does not lie over the codomain of the map.
    ///
    /// For [`SubgroupMaps`] methods, and for the containment-map functions
    /// that take one, "codomain" means the codomain `f.codomain()` of the
    /// group homomorphism `f`, not of the induced monotone map.
    CodomainMismatch,
    /// The image of a proper source relation is not a relation of the
    /// codomain lattice.
    InvalidImageRelation {
        /// The source relation.
        source: Edge,
        /// Its purported image.
        image: Edge,
    },
    /// The pushforward of some system is not an element of the supplied
    /// codomain [`GTransferLattice`], which can happen when that is a
    /// sublattice such as
    /// [`GLattice::saturated_transfer_systems`].
    PushforwardImageMissing,
    /// The pullback of some system is not an element of the supplied domain
    /// [`GTransferLattice`], which can happen when that is a sublattice such
    /// as [`GLattice::saturated_transfer_systems`].
    PullbackImageMissing,
    /// Induction was requested along a non-injective group homomorphism.
    NotInjective,
    /// Inflation or fixed points was requested along a non-surjective group
    /// homomorphism.
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

fn pushforward<M>(
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

fn pullback<M>(
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

fn image_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    pushforward(f.image_map(), system, f.codomain())
}

fn image_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    pullback(f.image_map(), system, f.domain())
}

fn preimage_pushforward(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_codomain_action(f, system.universe())?;
    pushforward(f.preimage_map(), system, f.domain())
}

fn preimage_pullback(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    validate_rubin_domain_action(f, system.universe())?;
    pullback(f.preimage_map(), system, f.codomain())
}

fn induction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_injective() {
        return Err(GTransferMapError::NotInjective);
    }
    image_pushforward(f, system)
}

fn restriction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    preimage_pushforward(f, system)
}

fn coinduction(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    preimage_pullback(f, system)
}

fn inflation(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pushforward(f, system)
}

fn fixed_points(
    f: &SubgroupMaps,
    system: &GTransferSystem,
) -> Result<GTransferSystem, GTransferMapError> {
    if !f.is_surjective() {
        return Err(GTransferMapError::NotSurjective);
    }
    preimage_pullback(f, system)
}

impl GTransferSystem {
    /// Computes the invariant pushforward of this G-transfer system along a
    /// monotone map out of its lattice.
    ///
    /// Writing `u` for `map`, the result is the least G-transfer system on
    /// `codomain` containing `u(x) -> u(y)` for every relation `x -> y` of
    /// this system. The map need not be equivariant. For arbitrary monotone
    /// maps this operation need not commute with composition.
    ///
    /// Returns [`GTransferMapError::DomainMismatch`] if the map's domain is
    /// not this system's lattice, and [`GTransferMapError::CodomainMismatch`]
    /// if its codomain is not the underlying lattice of `codomain`.
    pub fn pushforward<M>(
        &self,
        map: &M,
        codomain: &GLattice,
    ) -> Result<GTransferSystem, GTransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pushforward(map, self, codomain)
    }

    /// Computes the right-adjoint pullback of this G-transfer system along a
    /// monotone map into its lattice.
    ///
    /// A relation orbit of `domain` belongs to the result exactly when every
    /// relation in it, and every restriction of such a relation, maps to a
    /// relation of this system. Equivalently, this is the greatest G-transfer
    /// system on `domain` whose pushforward is contained in this one. For
    /// arbitrary monotone maps this operation need not commute with
    /// composition.
    ///
    /// Returns [`GTransferMapError::CodomainMismatch`] if the map's codomain
    /// is not this system's lattice, and [`GTransferMapError::DomainMismatch`]
    /// if its domain is not the underlying lattice of `domain`.
    pub fn pullback<M>(
        &self,
        map: &M,
        domain: &GLattice,
    ) -> Result<GTransferSystem, GTransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pullback(map, self, domain)
    }
}

impl SubgroupMaps {
    /// Pushes a G-transfer system on `self.domain()` forward along
    /// `K |-> f(K)`, giving an H-transfer system on `self.codomain()`
    /// (Rubin's `f_L`).
    ///
    /// This construction is defined for every group homomorphism. For an
    /// injective homomorphism, [`SubgroupMaps::induction`] is its conventional
    /// name. Returns [`GTransferMapError::DomainMismatch`] if `system` is not
    /// on `self.domain()`.
    pub fn image_pushforward(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        image_pushforward(self, system)
    }

    /// Pulls an H-transfer system on `self.codomain()` back along
    /// `K |-> f(K)`, giving a G-transfer system on `self.domain()` (Rubin's
    /// `f_R^{-1}`).
    ///
    /// This is right adjoint to [`SubgroupMaps::image_pushforward`]. Returns
    /// [`GTransferMapError::CodomainMismatch`] if `system` is not on
    /// `self.codomain()`.
    pub fn image_pullback(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        image_pullback(self, system)
    }

    /// Pushes an H-transfer system on `self.codomain()` forward along
    /// `J |-> f^{-1}(J)`, giving a G-transfer system on `self.domain()`
    /// (Rubin's `f_L^{-1}`).
    ///
    /// This is restriction along an arbitrary group homomorphism. For a
    /// quotient homomorphism it is inflation. Returns
    /// [`GTransferMapError::CodomainMismatch`] if `system` is not on
    /// `self.codomain()`.
    pub fn preimage_pushforward(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        preimage_pushforward(self, system)
    }

    /// Pulls a G-transfer system on `self.domain()` back along
    /// `J |-> f^{-1}(J)`, giving an H-transfer system on `self.codomain()`
    /// (Rubin's `f_R`).
    ///
    /// This is right adjoint to [`SubgroupMaps::preimage_pushforward`], and is
    /// coinduction along an arbitrary group homomorphism. For a quotient
    /// homomorphism it is fixed points. Returns
    /// [`GTransferMapError::DomainMismatch`] if `system` is not on
    /// `self.domain()`.
    pub fn preimage_pullback(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        preimage_pullback(self, system)
    }

    /// Induces a transfer system along an injective group homomorphism.
    ///
    /// This is the conventional injective case of
    /// [`SubgroupMaps::image_pushforward`]. Returns
    /// [`GTransferMapError::NotInjective`] if `f` is not injective.
    pub fn induction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        induction(self, system)
    }

    /// Restricts a transfer system along an arbitrary group homomorphism.
    ///
    /// This is an ergonomic name for [`SubgroupMaps::preimage_pushforward`].
    pub fn restriction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        restriction(self, system)
    }

    /// Coinduces a transfer system along an arbitrary group homomorphism.
    ///
    /// This is an ergonomic name for [`SubgroupMaps::preimage_pullback`].
    pub fn coinduction(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        coinduction(self, system)
    }

    /// Inflates a transfer system along a surjective group homomorphism.
    ///
    /// This is [`SubgroupMaps::preimage_pushforward`], restricted to
    /// surjective `f`. Returns [`GTransferMapError::NotSurjective`] if `f` is
    /// not surjective.
    pub fn inflation(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        inflation(self, system)
    }

    /// Takes fixed points along a surjective group homomorphism.
    ///
    /// This is the right adjoint to [`SubgroupMaps::inflation`]: it is
    /// [`SubgroupMaps::preimage_pullback`], restricted to surjective `f`.
    /// Returns [`GTransferMapError::NotSurjective`] if `f` is not surjective.
    pub fn fixed_points(
        &self,
        system: &GTransferSystem,
    ) -> Result<GTransferSystem, GTransferMapError> {
        fixed_points(self, system)
    }
}

/// Materializes Rubin's `f_L` between containment lattices.
///
/// `domain` must be a lattice of G-transfer systems on `f.domain()` and
/// `codomain` one on `f.codomain()`. The returned [`PosetMap`] goes from
/// `domain` to `codomain`.
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
///
/// `codomain` must be a lattice of H-transfer systems on `f.codomain()` and
/// `domain` one on `f.domain()`. The returned [`PosetMap`] goes from the first
/// lattice argument, `codomain`, to the second, `domain`.
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
///
/// `codomain` must be a lattice of H-transfer systems on `f.codomain()` and
/// `domain` one on `f.domain()`. The returned [`PosetMap`] goes from the first
/// lattice argument, `codomain`, to the second, `domain`.
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
///
/// `domain` must be a lattice of G-transfer systems on `f.domain()` and
/// `codomain` one on `f.codomain()`. The returned [`PosetMap`] goes from
/// `domain` to `codomain`.
pub fn preimage_pullback_containment_map(
    f: &SubgroupMaps,
    domain: &GTransferLattice,
    codomain: &GTransferLattice,
) -> Result<PosetMap, GTransferMapError> {
    validate_rubin_domain_action(f, domain.universe())?;
    validate_rubin_codomain_action(f, codomain.universe())?;
    pullback_containment_map(f.preimage_map(), domain, codomain)
}

/// Returns the map `system |-> system.pushforward(map, ..)` as a [`PosetMap`]
/// from `domain` to `codomain`.
///
/// `domain` and `codomain` are lattices of G-transfer systems over the
/// domain and codomain of `map`. On full lattices of G-transfer systems this
/// map is left adjoint to [`pullback_containment_map`].
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

/// Returns the map `system |-> system.pullback(map, ..)` as a [`PosetMap`]
/// from `codomain` to `domain`.
///
/// The first lattice argument, `codomain`, lies over the codomain of `map`
/// and is the domain of the returned map. On full lattices of G-transfer
/// systems this map is right adjoint to [`pushforward_containment_map`].
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
