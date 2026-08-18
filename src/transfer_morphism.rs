//! Transfer-system maps induced by lattice homomorphisms.
//!
//! A lattice homomorphism `f: L -> L'` induces two functions on transfer
//! systems.  The pushforward sends a transfer system to the transfer-system
//! closure of its image arrows, while the pullback tests membership after
//! applying `f`.  Both functions are monotone for containment and form an
//! adjoint pair.  For the composition-closed order, monotonicity is checked
//! separately in each direction.
//!
//! Universes and enumerated orders must reuse the exact `Arc<Lattice<_>>`
//! stored by the [`crate::morphism::LatticeMap`]. A separately allocated,
//! structurally equal lattice does not share its element-coordinate identity.

use crate::morphism::{LatticeMap, PosetMap};
use crate::poset::{Edge, ElementId};
use crate::transfer_lattice::{
    RawTransferSystem, TransferLattice, TransferPoset, TransferSystem, TransferUniverse,
    factorization_failure_for_raw,
};
use bitvec::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Errors produced while applying or materializing an induced transfer-system map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferMapError {
    /// The input expected to lie over the lattice homomorphism's domain does not.
    DomainMismatch,
    /// The input expected to lie over the lattice homomorphism's codomain does not.
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
    /// A computed pullback was absent from an enumerated codomain order.
    PullbackImageMissing,
}

impl fmt::Display for TransferMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferMapError::DomainMismatch => write!(
                formatter,
                "the supplied transfer-system data does not use the lattice homomorphism's domain"
            ),
            TransferMapError::CodomainMismatch => write!(
                formatter,
                "the supplied transfer-system data does not use the lattice homomorphism's codomain"
            ),
            TransferMapError::InvalidImageRelation { source, image } => write!(
                formatter,
                "source relation {} <= {} maps to unindexed codomain relation {} <= {}",
                source.from, source.to, image.from, image.to
            ),
            TransferMapError::PushforwardImageMissing => write!(
                formatter,
                "the computed pushforward is absent from the enumerated codomain order"
            ),
            TransferMapError::PullbackImageMissing => write!(
                formatter,
                "the computed pullback is absent from the enumerated codomain order"
            ),
        }
    }
}

impl std::error::Error for TransferMapError {}

/// Failure to make an induced function monotone for the composition-closed order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionMapError {
    /// The lattice, universe, or enumerated-order data are incompatible.
    Transfer(TransferMapError),
    /// A source cover is sent to a pair that is not composition-closed comparable.
    NotMonotone {
        /// The cover relation in the source transfer-system order.
        source_cover: Edge,
        /// The image of the cover's lower endpoint.
        lower_image: ElementId,
        /// The image of the cover's upper endpoint.
        upper_image: ElementId,
        /// A pair of target arrows forming an unsplittable square.
        failed_square: (Edge, Edge),
    },
}

impl fmt::Display for CompositionMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompositionMapError::Transfer(error) => write!(formatter, "{error}"),
            CompositionMapError::NotMonotone {
                source_cover,
                lower_image,
                upper_image,
                failed_square,
            } => {
                write!(
                    formatter,
                    "source cover {} <= {} maps to non-comparable target elements {} and {}",
                    source_cover.from, source_cover.to, lower_image, upper_image
                )?;
                let (first, second) = failed_square;
                write!(
                    formatter,
                    "; arrows {} <= {} and {} <= {} have no factorization witness",
                    first.from, first.to, second.from, second.to
                )
            }
        }
    }
}

impl std::error::Error for CompositionMapError {}

impl From<TransferMapError> for CompositionMapError {
    fn from(error: TransferMapError) -> Self {
        Self::Transfer(error)
    }
}

/// Computes the pushforward of one transfer system.
///
/// The supplied codomain universe determines the relation coordinates and
/// ownership of the result. It must use the same codomain lattice `Arc` as
/// `homomorphism`. This operation does not enumerate either transfer-system
/// order.
pub fn pushforward<A, B>(
    homomorphism: &LatticeMap<A, B>,
    system: &TransferSystem<A>,
    codomain: &Arc<TransferUniverse<B>>,
) -> Result<TransferSystem<B>, TransferMapError> {
    validate_domain_lattice(homomorphism, system.universe())?;
    validate_codomain_lattice(homomorphism, codomain)?;

    let mut generators = BitVec::repeat(false, codomain.proper_edges().len());
    for source_edge_id in system.raw().arrows().iter_ones() {
        let source = system.universe().proper_edges()[source_edge_id];
        let image = image_edge(homomorphism, source);
        if image.is_identity() {
            continue;
        }
        let Some(target_edge_id) = codomain.relation_index().proper_edge_id(image) else {
            return Err(TransferMapError::InvalidImageRelation { source, image });
        };
        generators.set(target_edge_id, true);
    }

    let raw = RawTransferSystem::new(codomain.close_arrows(&generators));
    Ok(TransferSystem::new(raw, Arc::clone(codomain)))
}

/// Computes the pullback of one transfer system.
///
/// The supplied domain universe determines the relation coordinates and
/// ownership of the result. It must use the same domain lattice `Arc` as
/// `homomorphism`. This operation does not enumerate either transfer-system
/// order.
pub fn pullback<A, B>(
    homomorphism: &LatticeMap<A, B>,
    system: &TransferSystem<B>,
    domain: &Arc<TransferUniverse<A>>,
) -> Result<TransferSystem<A>, TransferMapError> {
    validate_codomain_lattice(homomorphism, system.universe())?;
    validate_domain_lattice(homomorphism, domain)?;

    let mut arrows = BitVec::repeat(false, domain.proper_edges().len());
    for (source_edge_id, source) in domain.proper_edges().iter().copied().enumerate() {
        let image = image_edge(homomorphism, source);
        let belongs_to_pullback = if image.is_identity() {
            true
        } else {
            let Some(target_edge_id) = system.universe().relation_index().proper_edge_id(image)
            else {
                return Err(TransferMapError::InvalidImageRelation { source, image });
            };
            system.raw().arrows()[target_edge_id]
        };
        arrows.set(source_edge_id, belongs_to_pullback);
    }
    let raw = RawTransferSystem::new(arrows);
    debug_assert_eq!(domain.close_arrows(raw.arrows()), raw.arrows().clone());
    Ok(TransferSystem::new(raw, Arc::clone(domain)))
}

/// Constructs the pushforward poset homomorphism for the containment orders.
///
/// Together with [`pullback_containment_map`], the returned map is the left
/// adjoint. Its endpoint labels are user-facing [`TransferSystem`] values.
pub fn pushforward_containment_map<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferLattice<A>,
    codomain: &TransferLattice<B>,
) -> Result<PosetMap<TransferSystem<A>, TransferSystem<B>>, TransferMapError> {
    validate_containment_orders(homomorphism, domain, codomain)?;
    let images = materialize_pushforwards(
        homomorphism,
        domain.universe(),
        codomain.universe(),
        domain.raw_lattice().elements(),
        codomain.raw_lattice().elements(),
    )?;

    Ok(PosetMap::from_validated(
        Arc::new(domain.to_system_poset()),
        Arc::new(codomain.to_system_poset()),
        images,
    ))
}

/// Constructs the pullback poset homomorphism for the containment orders.
///
/// The first order is on the homomorphism's codomain and is therefore the
/// domain of the returned map. Together with [`pushforward_containment_map`],
/// the returned map is the right adjoint. Its endpoint labels are user-facing
/// [`TransferSystem`] values.
pub fn pullback_containment_map<A, B>(
    homomorphism: &LatticeMap<A, B>,
    codomain: &TransferLattice<B>,
    domain: &TransferLattice<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, TransferMapError> {
    validate_containment_orders(homomorphism, domain, codomain)?;
    let images = materialize_pullbacks(
        homomorphism,
        domain.universe(),
        codomain.universe(),
        codomain.raw_lattice().elements(),
        domain.raw_lattice().elements(),
    )?;

    Ok(PosetMap::from_validated(
        Arc::new(codomain.to_system_poset()),
        Arc::new(domain.to_system_poset()),
        images,
    ))
}

/// Attempts to construct the pushforward map for the composition-closed orders.
///
/// Failure includes a source cover whose images are not comparable and an
/// unsplittable target square witnessing the failure. Every source cover is
/// checked, so success is equivalent to monotonicity, not a heuristic.
pub fn try_pushforward_composition_map<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferPoset<A>,
    codomain: &TransferPoset<B>,
) -> Result<PosetMap<TransferSystem<A>, TransferSystem<B>>, CompositionMapError> {
    validate_composition_orders(homomorphism, domain, codomain)?;
    let images = materialize_pushforwards(
        homomorphism,
        domain.universe(),
        codomain.universe(),
        domain.raw_poset().elements(),
        codomain.raw_poset().elements(),
    )?;
    validate_composition_images(domain, codomain, &images)?;

    Ok(PosetMap::from_validated(
        Arc::new(domain.to_system_poset()),
        Arc::new(codomain.to_system_poset()),
        images,
    ))
}

/// Attempts to construct the pullback map for the composition-closed orders.
///
/// The first order is on the homomorphism's codomain and is therefore the
/// domain of the returned map. Every source cover is checked, so success is
/// equivalent to monotonicity, not a heuristic.
pub fn try_pullback_composition_map<A, B>(
    homomorphism: &LatticeMap<A, B>,
    codomain: &TransferPoset<B>,
    domain: &TransferPoset<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, CompositionMapError> {
    validate_composition_orders(homomorphism, domain, codomain)?;
    let images = materialize_pullbacks(
        homomorphism,
        domain.universe(),
        codomain.universe(),
        codomain.raw_poset().elements(),
        domain.raw_poset().elements(),
    )?;
    validate_composition_images(codomain, domain, &images)?;

    Ok(PosetMap::from_validated(
        Arc::new(codomain.to_system_poset()),
        Arc::new(domain.to_system_poset()),
        images,
    ))
}

fn image_edge<A, B>(homomorphism: &LatticeMap<A, B>, edge: Edge) -> Edge {
    Edge::new(homomorphism.map()[edge.from], homomorphism.map()[edge.to])
}

fn edge_images<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
) -> Result<Vec<Option<usize>>, TransferMapError> {
    domain
        .proper_edges()
        .iter()
        .copied()
        .map(|source| {
            let image = image_edge(homomorphism, source);
            if image.is_identity() {
                Ok(None)
            } else {
                codomain
                    .relation_index()
                    .proper_edge_id(image)
                    .map(Some)
                    .ok_or(TransferMapError::InvalidImageRelation { source, image })
            }
        })
        .collect()
}

fn pullback_raw<A>(
    source: &RawTransferSystem,
    domain: &TransferUniverse<A>,
    edge_images: &[Option<usize>],
) -> RawTransferSystem {
    let mut arrows = BitVec::repeat(false, domain.proper_edges().len());
    for (source_edge_id, target_edge_id) in edge_images.iter().copied().enumerate() {
        if target_edge_id.is_none_or(|target_edge_id| source.arrows()[target_edge_id]) {
            arrows.set(source_edge_id, true);
        }
    }
    RawTransferSystem::new(arrows)
}

fn materialize_pushforwards<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
    source_systems: &[RawTransferSystem],
    target_systems: &[RawTransferSystem],
) -> Result<Vec<ElementId>, TransferMapError> {
    let edge_images = edge_images(homomorphism, domain, codomain)?;
    let target_ids = target_system_ids(target_systems);
    let mut closure_cache = HashMap::<BitVec, RawTransferSystem>::new();
    let mut result = Vec::with_capacity(source_systems.len());

    for source in source_systems {
        let mut generators = BitVec::repeat(false, codomain.proper_edges().len());
        for source_edge_id in source.arrows().iter_ones() {
            if let Some(target_edge_id) = edge_images[source_edge_id] {
                generators.set(target_edge_id, true);
            }
        }
        let image = closure_cache
            .entry(generators.clone())
            .or_insert_with(|| RawTransferSystem::new(codomain.close_arrows(&generators)));
        result.push(
            target_ids
                .get(image)
                .copied()
                .ok_or(TransferMapError::PushforwardImageMissing)?,
        );
    }
    Ok(result)
}

fn materialize_pullbacks<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
    source_systems: &[RawTransferSystem],
    target_systems: &[RawTransferSystem],
) -> Result<Vec<ElementId>, TransferMapError> {
    let edge_images = edge_images(homomorphism, domain, codomain)?;
    let target_ids = target_system_ids(target_systems);
    source_systems
        .iter()
        .map(|source| {
            let image = pullback_raw(source, domain, &edge_images);
            target_ids
                .get(&image)
                .copied()
                .ok_or(TransferMapError::PullbackImageMissing)
        })
        .collect()
}

fn target_system_ids(systems: &[RawTransferSystem]) -> HashMap<&RawTransferSystem, ElementId> {
    systems
        .iter()
        .enumerate()
        .map(|(id, raw)| (raw, id))
        .collect()
}

fn validate_composition_images<A, B>(
    source: &TransferPoset<A>,
    target: &TransferPoset<B>,
    images: &[ElementId],
) -> Result<(), CompositionMapError> {
    let mut source_covers = source.cover_relations().into_iter().collect::<Vec<_>>();
    source_covers.sort_unstable();
    for source_cover in source_covers {
        let lower_image = images[source_cover.from];
        let upper_image = images[source_cover.to];
        if target.raw_poset().leq(lower_image, upper_image) {
            continue;
        }

        let lower = target
            .raw_poset()
            .element(lower_image)
            .expect("an induced map image should be a target element");
        let upper = target
            .raw_poset()
            .element(upper_image)
            .expect("an induced map image should be a target element");
        let failure = factorization_failure_for_raw(target.universe(), lower, upper).expect(
            "containment-monotone induced images can fail only the factorization condition",
        );
        return Err(CompositionMapError::NotMonotone {
            source_cover,
            lower_image,
            upper_image,
            failed_square: (failure.first, failure.second),
        });
    }
    Ok(())
}

fn validate_domain_lattice<A, B>(
    homomorphism: &LatticeMap<A, B>,
    universe: &TransferUniverse<A>,
) -> Result<(), TransferMapError> {
    if Arc::ptr_eq(homomorphism.domain(), universe.lattice()) {
        Ok(())
    } else {
        Err(TransferMapError::DomainMismatch)
    }
}

fn validate_codomain_lattice<A, B>(
    homomorphism: &LatticeMap<A, B>,
    universe: &TransferUniverse<B>,
) -> Result<(), TransferMapError> {
    if Arc::ptr_eq(homomorphism.codomain(), universe.lattice()) {
        Ok(())
    } else {
        Err(TransferMapError::CodomainMismatch)
    }
}

fn validate_containment_orders<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferLattice<A>,
    codomain: &TransferLattice<B>,
) -> Result<(), TransferMapError> {
    validate_domain_lattice(homomorphism, domain.universe())?;
    validate_codomain_lattice(homomorphism, codomain.universe())
}

fn validate_composition_orders<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferPoset<A>,
    codomain: &TransferPoset<B>,
) -> Result<(), TransferMapError> {
    validate_domain_lattice(homomorphism, domain.universe())?;
    validate_codomain_lattice(homomorphism, codomain.universe())
}
