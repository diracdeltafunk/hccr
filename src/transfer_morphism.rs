//! Transfer-system maps induced by monotone maps between finite lattices.
//!
//! A monotone map `f: L -> M` sends a relation `x <= y` to
//! `f(x) <= f(y)`. Sending every arrow of a transfer system this way may lose
//! restriction closure, so
//! [`pushforward`] takes the least transfer system containing the image arrows.
//! Its right adjoint [`pullback`] takes the greatest transfer system contained
//! in the pointwise inverse image. In
//! symbols, for transfer systems `S` on `L` and `T` on `M`,
//! `pushforward(S) <= T` exactly when `S <= pullback(T)`.
//!
//! There is also a superficially similar operation,
//! [`generated_inverse_image`], which takes the *least* transfer system
//! containing the pointwise inverse image. It is generally different from the
//! pullback and is not right adjoint to pushforward.
//!
//! For a meet-preserving map, including every
//! [`crate::morphism::LatticeMap`], the raw inverse image is already a transfer
//! system, so the two inverse-image operations agree. For the
//! composition-closed order, monotonicity is checked separately for every
//! operation.
//!
//! For merely monotone maps, these constructions need not respect
//! composition: transfer closure at an intermediate lattice can change the
//! result. They are functorial when the maps are lattice homomorphisms.
//!
//! Map endpoints, universes, and enumerated orders must share the same order
//! coordinates. An independently reconstructed, structurally equal lattice
//! does not share that concrete presentation identity.

use crate::bitvec_utils::is_subset;
use crate::morphism::{MonotoneMap, PosetMap};
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
    /// A computed pullback was absent from an enumerated codomain order.
    PullbackImageMissing,
    /// A computed generated inverse image was absent from an enumerated order.
    GeneratedInverseImageMissing,
}

impl fmt::Display for TransferMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferMapError::DomainMismatch => write!(
                formatter,
                "the supplied transfer-system data does not use the monotone map's domain coordinates"
            ),
            TransferMapError::CodomainMismatch => write!(
                formatter,
                "the supplied transfer-system data does not use the monotone map's codomain coordinates"
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
            TransferMapError::GeneratedInverseImageMissing => write!(
                formatter,
                "the computed generated inverse image is absent from the enumerated codomain order"
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

/// Computes the pushforward of one transfer system along a monotone map.
///
/// Each selected arrow is mapped pointwise; identity images are discarded
/// because identities are implicit, duplicate images collapse to one bit, and
/// the resulting set is closed under the transfer-system axioms.
///
/// The supplied codomain universe determines the relation coordinates and
/// ownership of the result. It must share order coordinates with the map's
/// codomain. This operation does not enumerate either transfer-system order.
/// For merely monotone maps it need not commute with composition; it does for
/// lattice homomorphisms.
pub fn pushforward<A, B, M>(
    map: &M,
    system: &TransferSystem<A>,
    codomain: &Arc<TransferUniverse<B>>,
) -> Result<TransferSystem<B>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_domain_lattice(map, system.universe())?;
    validate_codomain_lattice(map, codomain)?;

    let mut generators = BitVec::repeat(false, codomain.proper_edges().len());
    for source_edge_id in system.raw().arrows().iter_ones() {
        let source = system.universe().proper_edges()[source_edge_id];
        let image = image_edge(map, source);
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

/// Computes the right-adjoint pullback of one transfer system.
///
/// A relation `x -> y` belongs to the result exactly when, for every `z <= y`,
/// the relation `f(x /\ z) -> f(z)` belongs to `system`. Checking every such
/// restriction directly constructs the greatest
/// transfer system contained in the raw inverse image, and it is right adjoint
/// to [`pushforward`] under containment.
///
/// The supplied domain universe determines the result's coordinates and
/// ownership and must share order coordinates with the map's domain. This
/// operation does not enumerate either transfer-system order.
/// For merely monotone maps it need not commute with composition, despite
/// being right adjoint for each individual map; it does for lattice
/// homomorphisms.
pub fn pullback<A, B, M>(
    map: &M,
    system: &TransferSystem<B>,
    domain: &Arc<TransferUniverse<A>>,
) -> Result<TransferSystem<A>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_codomain_lattice(map, system.universe())?;
    validate_domain_lattice(map, domain)?;

    let raw = if map.is_known_meet_preserving() {
        pointwise_raw_inverse_image(map, system.raw(), domain, system.universe())?
    } else {
        pointwise_right_adjoint(map, system.raw(), domain, system.universe())?
    };
    debug_assert_eq!(domain.close_arrows(raw.arrows()), raw.arrows().clone());
    Ok(TransferSystem::new(raw, Arc::clone(domain)))
}

/// Generates a transfer system from the raw inverse image of `system`.
///
/// Unlike [`pullback`], this is the least transfer system containing every
/// relation `x -> y` for which `f(x) -> f(y)` belongs to `system`, then applies
/// transfer-system closure. It is
/// containment-monotone but is generally not right adjoint to [`pushforward`].
/// The two operations agree whenever the map is meet-preserving.
/// For merely monotone maps this operation need not commute with composition;
/// it does for lattice homomorphisms.
pub fn generated_inverse_image<A, B, M>(
    map: &M,
    system: &TransferSystem<B>,
    domain: &Arc<TransferUniverse<A>>,
) -> Result<TransferSystem<A>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_codomain_lattice(map, system.universe())?;
    validate_domain_lattice(map, domain)?;

    let inverse_image = pointwise_raw_inverse_image(map, system.raw(), domain, system.universe())?;
    let raw = if map.is_known_meet_preserving() {
        inverse_image
    } else {
        RawTransferSystem::new(domain.close_arrows(inverse_image.arrows()))
    };
    debug_assert_eq!(domain.close_arrows(raw.arrows()), raw.arrows().clone());
    Ok(TransferSystem::new(raw, Arc::clone(domain)))
}

/// Constructs the pushforward poset map for the containment orders.
///
/// Together with [`pullback_containment_map`], the returned map is the left
/// adjoint. This materializes [`pushforward`] and inherits its composition
/// caveat. Its endpoint labels are user-facing [`TransferSystem`] values.
pub fn pushforward_containment_map<A, B, M>(
    map: &M,
    domain: &TransferLattice<A>,
    codomain: &TransferLattice<B>,
) -> Result<PosetMap<TransferSystem<A>, TransferSystem<B>>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_containment_orders(map, domain, codomain)?;
    let images = materialize_pushforwards(
        map,
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

/// Constructs the pullback poset map for the containment orders.
///
/// The first order is on the map's codomain and is therefore the
/// domain of the returned map. Together with [`pushforward_containment_map`],
/// the returned map is the right adjoint. Its endpoint labels are user-facing
/// [`TransferSystem`] values. This materializes [`pullback`] and inherits its
/// composition caveat.
pub fn pullback_containment_map<A, B, M>(
    map: &M,
    codomain: &TransferLattice<B>,
    domain: &TransferLattice<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_containment_orders(map, domain, codomain)?;
    let images = materialize_pullbacks(
        map,
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

/// Constructs the generated inverse-image map for the containment orders.
///
/// The first order is on the monotone map's codomain and is therefore the
/// domain of the returned map. This operation is always containment-monotone,
/// but unlike [`pullback_containment_map`] it is not generally right adjoint to
/// [`pushforward_containment_map`]. It materializes
/// [`generated_inverse_image`] and inherits its composition caveat.
pub fn generated_inverse_image_containment_map<A, B, M>(
    map: &M,
    codomain: &TransferLattice<B>,
    domain: &TransferLattice<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_containment_orders(map, domain, codomain)?;
    let images = materialize_generated_inverse_images(
        map,
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
/// This materializes [`pushforward`] and inherits its composition caveat.
pub fn try_pushforward_composition_map<A, B, M>(
    map: &M,
    domain: &TransferPoset<A>,
    codomain: &TransferPoset<B>,
) -> Result<PosetMap<TransferSystem<A>, TransferSystem<B>>, CompositionMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_composition_orders(map, domain, codomain)?;
    let images = materialize_pushforwards(
        map,
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
/// The first order is on the map's codomain and is therefore the
/// domain of the returned map. Every source cover is checked, so success is
/// equivalent to monotonicity, not a heuristic.
/// This materializes [`pullback`] and inherits its composition caveat.
pub fn try_pullback_composition_map<A, B, M>(
    map: &M,
    codomain: &TransferPoset<B>,
    domain: &TransferPoset<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, CompositionMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_composition_orders(map, domain, codomain)?;
    let images = materialize_pullbacks(
        map,
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

/// Attempts to construct the generated inverse-image map for the
/// composition-closed orders.
///
/// The first order is on the monotone map's codomain and is therefore the
/// domain of the returned map. Every source cover is checked, so success is
/// equivalent to monotonicity, not a heuristic.
/// This materializes [`generated_inverse_image`] and inherits its composition
/// caveat.
pub fn try_generated_inverse_image_composition_map<A, B, M>(
    map: &M,
    codomain: &TransferPoset<B>,
    domain: &TransferPoset<A>,
) -> Result<PosetMap<TransferSystem<B>, TransferSystem<A>>, CompositionMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_composition_orders(map, domain, codomain)?;
    let images = materialize_generated_inverse_images(
        map,
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

fn image_edge<A, B, M>(map: &M, edge: Edge) -> Edge
where
    M: MonotoneMap<A, B> + ?Sized,
{
    Edge::new(map.images()[edge.from], map.images()[edge.to])
}

fn edge_images<A, B, M>(
    map: &M,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
) -> Result<Vec<Option<usize>>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    domain
        .proper_edges()
        .iter()
        .copied()
        .map(|source| {
            let image = image_edge(map, source);
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

fn pointwise_raw_inverse_image<A, B, M>(
    map: &M,
    source: &RawTransferSystem,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
) -> Result<RawTransferSystem, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let mut arrows = BitVec::repeat(false, domain.proper_edges().len());
    for (source_edge_id, source_edge) in domain.proper_edges().iter().copied().enumerate() {
        let image = image_edge(map, source_edge);
        let belongs = if image.is_identity() {
            true
        } else {
            let Some(target_edge_id) = codomain.relation_index().proper_edge_id(image) else {
                return Err(TransferMapError::InvalidImageRelation {
                    source: source_edge,
                    image,
                });
            };
            source.arrows()[target_edge_id]
        };
        arrows.set(source_edge_id, belongs);
    }
    Ok(RawTransferSystem::new(arrows))
}

fn pointwise_right_adjoint<A, B, M>(
    map: &M,
    source: &RawTransferSystem,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
) -> Result<RawTransferSystem, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let lattice = domain.lattice();
    let mut arrows = BitVec::repeat(false, domain.proper_edges().len());

    for (source_edge_id, edge) in domain.proper_edges().iter().copied().enumerate() {
        let mut belongs = true;
        for restriction_target in
            lattice.as_poset().relation_matrix_transpose()[edge.to].iter_ones()
        {
            let restriction_source = lattice.meet_id(edge.from, restriction_target);
            let restricted = Edge::new(restriction_source, restriction_target);
            let image = image_edge(map, restricted);
            if image.is_identity() {
                continue;
            }
            let Some(target_edge_id) = codomain.relation_index().proper_edge_id(image) else {
                return Err(TransferMapError::InvalidImageRelation {
                    source: restricted,
                    image,
                });
            };
            if !source.arrows()[target_edge_id] {
                belongs = false;
                break;
            }
        }
        arrows.set(source_edge_id, belongs);
    }

    Ok(RawTransferSystem::new(arrows))
}

fn raw_inverse_image<A>(
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

fn pullback_requirements<A, B, M>(
    map: &M,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
) -> Result<Vec<BitVec>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let lattice = domain.lattice();
    domain
        .proper_edges()
        .iter()
        .copied()
        .map(|source| {
            let mut requirements = BitVec::repeat(false, codomain.proper_edges().len());
            for restriction_target in
                lattice.as_poset().relation_matrix_transpose()[source.to].iter_ones()
            {
                let restriction_source = lattice.meet_id(source.from, restriction_target);
                let restricted = Edge::new(restriction_source, restriction_target);
                let image = image_edge(map, restricted);
                if image.is_identity() {
                    continue;
                }
                let Some(target_edge_id) = codomain.relation_index().proper_edge_id(image) else {
                    return Err(TransferMapError::InvalidImageRelation {
                        source: restricted,
                        image,
                    });
                };
                requirements.set(target_edge_id, true);
            }
            Ok(requirements)
        })
        .collect()
}

fn right_adjoint_raw<A>(
    source: &RawTransferSystem,
    domain: &TransferUniverse<A>,
    requirements: &[BitVec],
) -> RawTransferSystem {
    let mut arrows = BitVec::repeat(false, domain.proper_edges().len());
    for (source_edge_id, required_arrows) in requirements.iter().enumerate() {
        if is_subset(required_arrows, source.arrows()) {
            arrows.set(source_edge_id, true);
        }
    }
    RawTransferSystem::new(arrows)
}

fn materialize_pushforwards<A, B, M>(
    map: &M,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
    source_systems: &[RawTransferSystem],
    target_systems: &[RawTransferSystem],
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let edge_images = edge_images(map, domain, codomain)?;
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

fn materialize_pullbacks<A, B, M>(
    map: &M,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
    source_systems: &[RawTransferSystem],
    target_systems: &[RawTransferSystem],
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let target_ids = target_system_ids(target_systems);
    let edge_images = map
        .is_known_meet_preserving()
        .then(|| edge_images(map, domain, codomain))
        .transpose()?;
    let requirements = edge_images
        .is_none()
        .then(|| pullback_requirements(map, domain, codomain))
        .transpose()?;

    source_systems
        .iter()
        .map(|source| {
            let image = match (&edge_images, &requirements) {
                (Some(edge_images), None) => raw_inverse_image(source, domain, edge_images),
                (None, Some(requirements)) => right_adjoint_raw(source, domain, requirements),
                _ => unreachable!("exactly one pullback strategy is selected"),
            };
            target_ids
                .get(&image)
                .copied()
                .ok_or(TransferMapError::PullbackImageMissing)
        })
        .collect()
}

fn materialize_generated_inverse_images<A, B, M>(
    map: &M,
    domain: &TransferUniverse<A>,
    codomain: &TransferUniverse<B>,
    source_systems: &[RawTransferSystem],
    target_systems: &[RawTransferSystem],
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    let edge_images = edge_images(map, domain, codomain)?;
    let target_ids = target_system_ids(target_systems);
    let mut closure_cache = HashMap::<RawTransferSystem, RawTransferSystem>::new();
    let raw_inverse_is_closed = map.is_known_meet_preserving();

    source_systems
        .iter()
        .map(|source| {
            let inverse_image = raw_inverse_image(source, domain, &edge_images);
            let image = if raw_inverse_is_closed {
                &inverse_image
            } else {
                closure_cache
                    .entry(inverse_image.clone())
                    .or_insert_with(|| {
                        RawTransferSystem::new(domain.close_arrows(inverse_image.arrows()))
                    })
            };
            target_ids
                .get(image)
                .copied()
                .ok_or(TransferMapError::GeneratedInverseImageMissing)
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

fn validate_domain_lattice<A, B, M>(
    map: &M,
    universe: &TransferUniverse<A>,
) -> Result<(), TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    if map
        .domain_poset()
        .shares_order_coordinates_with(universe.lattice().as_poset())
    {
        Ok(())
    } else {
        Err(TransferMapError::DomainMismatch)
    }
}

fn validate_codomain_lattice<A, B, M>(
    map: &M,
    universe: &TransferUniverse<B>,
) -> Result<(), TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    if map
        .codomain_poset()
        .shares_order_coordinates_with(universe.lattice().as_poset())
    {
        Ok(())
    } else {
        Err(TransferMapError::CodomainMismatch)
    }
}

fn validate_containment_orders<A, B, M>(
    map: &M,
    domain: &TransferLattice<A>,
    codomain: &TransferLattice<B>,
) -> Result<(), TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_domain_lattice(map, domain.universe())?;
    validate_codomain_lattice(map, codomain.universe())
}

fn validate_composition_orders<A, B, M>(
    map: &M,
    domain: &TransferPoset<A>,
    codomain: &TransferPoset<B>,
) -> Result<(), TransferMapError>
where
    M: MonotoneMap<A, B> + ?Sized,
{
    validate_domain_lattice(map, domain.universe())?;
    validate_codomain_lattice(map, codomain.universe())
}
