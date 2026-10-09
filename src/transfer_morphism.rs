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
//! The same operations are available as methods:
//! [`TransferSystem::pushforward`], [`TransferSystem::pullback`], and
//! [`TransferSystem::generated_inverse_image`].
//!
//! ```
//! use hccr::lattice::Lattice;
//! use hccr::morphism::LatticeMap;
//!
//! // The quotient [2] -> [1] identifying the bottom and middle elements.
//! let c3 = Lattice::chain(2);
//! let c2 = Lattice::chain(1);
//! let quotient = LatticeMap::new(&c3, &c2, vec![0, 0, 1])?;
//!
//! let system = c3.transfer_system_generated_by([(1, 2)])?;
//! let image = system.pushforward(&quotient)?;
//! assert!(image.contains_relation((0, 1).into()));
//! assert!(system <= image.pullback(&quotient)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::bitvec_utils::is_subset;
use crate::lattice::Lattice;
use crate::morphism::{MonotoneMap, PosetMap};
use crate::poset::{Edge, ElementId};
use crate::transfer_lattice::{
    TransferLattice, TransferPoset, TransferSystem, factorization_failure_between,
};
use bitvec::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Errors produced while applying or materializing an induced transfer-system map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferMapError {
    /// The input expected to lie over the monotone map's domain does not.
    DomainMismatch,
    /// The input expected to lie over the monotone map's codomain does not.
    CodomainMismatch,
    /// The monotone map's domain or codomain is not a lattice.
    NotALattice,
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
            TransferMapError::NotALattice => write!(
                formatter,
                "transfer systems need the map's domain and codomain to be lattices"
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
/// because identities are implicit, duplicate images collapse, and the
/// resulting set is closed under the transfer-system axioms. The result lives
/// on the map's codomain, which must be a lattice.
///
/// For merely monotone maps this need not commute with composition; it does
/// for lattice homomorphisms.
pub fn pushforward<M>(map: &M, system: &TransferSystem) -> Result<TransferSystem, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (_, codomain) = endpoints(map, Some(system.lattice()), None)?;
    let edge_images = edge_images(map, system.lattice(), &codomain)?;
    let universe = codomain.transfer_universe();
    let mut generators = BitVec::repeat(false, universe.proper_edges().len());
    for source_edge_id in system.arrows().iter_ones() {
        if let Some(target_edge_id) = edge_images[source_edge_id] {
            generators.set(target_edge_id, true);
        }
    }
    let arrows = universe.close(&generators);
    Ok(TransferSystem::new(codomain, arrows))
}

/// Computes the right-adjoint pullback of one transfer system.
///
/// A relation `x -> y` belongs to the result exactly when, for every `z <= y`,
/// the relation `f(x /\ z) -> f(z)` belongs to `system`. This is the greatest
/// transfer system contained in the raw inverse image, and it is right adjoint
/// to [`pushforward`] under containment. The result lives on the map's
/// domain, which must be a lattice.
///
/// For merely monotone maps it need not commute with composition, despite
/// being right adjoint for each individual map; it does for lattice
/// homomorphisms.
pub fn pullback<M>(map: &M, system: &TransferSystem) -> Result<TransferSystem, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (domain, codomain) = endpoints(map, None, Some(system.lattice()))?;
    let arrows = if map.is_known_meet_preserving() {
        raw_inverse_image(system.arrows(), &edge_images(map, &domain, &codomain)?)
    } else {
        right_adjoint_raw(
            system.arrows(),
            &pullback_requirements(map, &domain, &codomain)?,
        )
    };
    debug_assert_eq!(domain.transfer_universe().close(&arrows), arrows);
    Ok(TransferSystem::new(domain, arrows))
}

/// Generates a transfer system from the raw inverse image of `system`.
///
/// Unlike [`pullback`], this is the least transfer system containing every
/// relation `x -> y` for which `f(x) -> f(y)` belongs to `system`. It is
/// containment-monotone but is generally not right adjoint to [`pushforward`].
/// The two operations agree whenever the map is meet-preserving.
/// For merely monotone maps this operation need not commute with composition;
/// it does for lattice homomorphisms.
pub fn generated_inverse_image<M>(
    map: &M,
    system: &TransferSystem,
) -> Result<TransferSystem, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (domain, codomain) = endpoints(map, None, Some(system.lattice()))?;
    let inverse_image = raw_inverse_image(system.arrows(), &edge_images(map, &domain, &codomain)?);
    let arrows = if map.is_known_meet_preserving() {
        inverse_image
    } else {
        domain.transfer_universe().close(&inverse_image)
    };
    Ok(TransferSystem::new(domain, arrows))
}

impl TransferSystem {
    /// Pushes this transfer system forward along a monotone map out of its
    /// lattice. See [`pushforward`].
    pub fn pushforward<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pushforward(map, self)
    }

    /// Pulls this transfer system back along a monotone map into its lattice.
    /// See [`pullback`].
    pub fn pullback<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pullback(map, self)
    }

    /// Generates a transfer system from the raw inverse image of this one
    /// along a monotone map into its lattice. See [`generated_inverse_image`].
    pub fn generated_inverse_image<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        generated_inverse_image(map, self)
    }
}

/// Constructs the pushforward map `Tr(L) -> Tr(M)` of containment lattices.
///
/// `domain` and `codomain` must be transfer-system lattices on the map's
/// domain and codomain. Together with [`pullback_containment_map`], the
/// returned map is the left adjoint. It inherits the composition caveat of
/// [`pushforward`].
pub fn pushforward_containment_map<M>(
    map: &M,
    domain: &TransferLattice,
    codomain: &TransferLattice,
) -> Result<PosetMap, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_pushforwards(map, &source, &target, domain.systems(), |system| {
        codomain.id_of(system)
    })?;
    Ok(PosetMap::from_validated(
        domain.as_poset().clone(),
        codomain.as_poset().clone(),
        images,
    ))
}

/// Constructs the pullback map `Tr(M) -> Tr(L)` of containment lattices.
///
/// The first order is on the map's codomain and is therefore the
/// domain of the returned map. Together with [`pushforward_containment_map`],
/// the returned map is the right adjoint. It inherits the composition caveat
/// of [`pullback`].
pub fn pullback_containment_map<M>(
    map: &M,
    codomain: &TransferLattice,
    domain: &TransferLattice,
) -> Result<PosetMap, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_pullbacks(map, &source, &target, codomain.systems(), |system| {
        domain.id_of(system)
    })?;
    Ok(PosetMap::from_validated(
        codomain.as_poset().clone(),
        domain.as_poset().clone(),
        images,
    ))
}

/// Constructs the generated inverse-image map `Tr(M) -> Tr(L)`.
///
/// The first order is on the monotone map's codomain and is therefore the
/// domain of the returned map. This operation is always containment-monotone,
/// but unlike [`pullback_containment_map`] it is not generally right adjoint to
/// [`pushforward_containment_map`].
pub fn generated_inverse_image_containment_map<M>(
    map: &M,
    codomain: &TransferLattice,
    domain: &TransferLattice,
) -> Result<PosetMap, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_generated_inverse_images(
        map,
        &source,
        &target,
        codomain.systems(),
        |system| domain.id_of(system),
    )?;
    Ok(PosetMap::from_validated(
        codomain.as_poset().clone(),
        domain.as_poset().clone(),
        images,
    ))
}

/// Attempts to construct the pushforward map for the composition-closed orders.
///
/// Failure includes a source cover whose images are not comparable and an
/// unsplittable target square witnessing the failure. Every source cover is
/// checked, so success is equivalent to monotonicity, not a heuristic.
pub fn try_pushforward_composition_map<M>(
    map: &M,
    domain: &TransferPoset,
    codomain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_pushforwards(map, &source, &target, domain.systems(), |system| {
        codomain.id_of(system)
    })?;
    validate_composition_images(domain, codomain, &images)?;
    Ok(PosetMap::from_validated(
        domain.as_poset().clone(),
        codomain.as_poset().clone(),
        images,
    ))
}

/// Attempts to construct the pullback map for the composition-closed orders.
///
/// The first order is on the map's codomain and is therefore the
/// domain of the returned map. Every source cover is checked, so success is
/// equivalent to monotonicity, not a heuristic.
pub fn try_pullback_composition_map<M>(
    map: &M,
    codomain: &TransferPoset,
    domain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_pullbacks(map, &source, &target, codomain.systems(), |system| {
        domain.id_of(system)
    })?;
    validate_composition_images(codomain, domain, &images)?;
    Ok(PosetMap::from_validated(
        codomain.as_poset().clone(),
        domain.as_poset().clone(),
        images,
    ))
}

/// Attempts to construct the generated inverse-image map for the
/// composition-closed orders.
///
/// The first order is on the monotone map's codomain and is therefore the
/// domain of the returned map. Every source cover is checked, so success is
/// equivalent to monotonicity, not a heuristic.
pub fn try_generated_inverse_image_composition_map<M>(
    map: &M,
    codomain: &TransferPoset,
    domain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    let (source, target) = endpoints(
        map,
        Some(domain.base_lattice()),
        Some(codomain.base_lattice()),
    )?;
    let images = materialize_generated_inverse_images(
        map,
        &source,
        &target,
        codomain.systems(),
        |system| domain.id_of(system),
    )?;
    validate_composition_images(codomain, domain, &images)?;
    Ok(PosetMap::from_validated(
        codomain.as_poset().clone(),
        domain.as_poset().clone(),
        images,
    ))
}

/// Resolves the domain and codomain of `map` as lattices and checks them
/// against the lattices that the inputs live on, when those are given.
fn endpoints<M>(
    map: &M,
    expected_domain: Option<&Lattice>,
    expected_codomain: Option<&Lattice>,
) -> Result<(Lattice, Lattice), TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let domain =
        Lattice::new(map.domain_poset().clone()).map_err(|_| TransferMapError::NotALattice)?;
    let codomain =
        Lattice::new(map.codomain_poset().clone()).map_err(|_| TransferMapError::NotALattice)?;
    if expected_domain.is_some_and(|expected| *expected != domain) {
        return Err(TransferMapError::DomainMismatch);
    }
    if expected_codomain.is_some_and(|expected| *expected != codomain) {
        return Err(TransferMapError::CodomainMismatch);
    }
    Ok((domain, codomain))
}

fn image_edge<M>(map: &M, edge: Edge) -> Edge
where
    M: MonotoneMap + ?Sized,
{
    Edge::new(map.images()[edge.from], map.images()[edge.to])
}

fn edge_images<M>(
    map: &M,
    domain: &Lattice,
    codomain: &Lattice,
) -> Result<Vec<Option<usize>>, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let target = codomain.transfer_universe();
    domain
        .transfer_universe()
        .proper_edges()
        .iter()
        .copied()
        .map(|source| {
            let image = image_edge(map, source);
            if image.is_identity() {
                Ok(None)
            } else {
                target
                    .proper_edge_id(image)
                    .map(Some)
                    .ok_or(TransferMapError::InvalidImageRelation { source, image })
            }
        })
        .collect()
}

fn raw_inverse_image(source: &BitVec, edge_images: &[Option<usize>]) -> BitVec {
    edge_images
        .iter()
        .map(|target_edge_id| target_edge_id.is_none_or(|target_edge_id| source[target_edge_id]))
        .collect()
}

fn pullback_requirements<M>(
    map: &M,
    domain: &Lattice,
    codomain: &Lattice,
) -> Result<Vec<BitVec>, TransferMapError>
where
    M: MonotoneMap + ?Sized,
{
    let target = codomain.transfer_universe();
    domain
        .transfer_universe()
        .proper_edges()
        .iter()
        .copied()
        .map(|source| {
            let mut requirements = BitVec::repeat(false, target.proper_edges().len());
            for restriction_target in domain.relation_matrix_transpose()[source.to].iter_ones() {
                let restriction_source = domain.meet(source.from, restriction_target);
                let restricted = Edge::new(restriction_source, restriction_target);
                let image = image_edge(map, restricted);
                if image.is_identity() {
                    continue;
                }
                let Some(target_edge_id) = target.proper_edge_id(image) else {
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

fn right_adjoint_raw(source: &BitVec, requirements: &[BitVec]) -> BitVec {
    requirements
        .iter()
        .map(|required_arrows| is_subset(required_arrows, source))
        .collect()
}

fn materialize_pushforwards<M, F>(
    map: &M,
    domain: &Lattice,
    codomain: &Lattice,
    source_systems: &[TransferSystem],
    target_id: F,
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap + ?Sized,
    F: Fn(&TransferSystem) -> Option<ElementId>,
{
    let edge_images = edge_images(map, domain, codomain)?;
    let universe = codomain.transfer_universe();
    let mut closure_cache = HashMap::<BitVec, Option<ElementId>>::new();
    source_systems
        .iter()
        .map(|source| {
            let mut generators = BitVec::repeat(false, universe.proper_edges().len());
            for source_edge_id in source.arrows().iter_ones() {
                if let Some(target_edge_id) = edge_images[source_edge_id] {
                    generators.set(target_edge_id, true);
                }
            }
            closure_cache
                .entry(generators.clone())
                .or_insert_with(|| {
                    target_id(&TransferSystem::new(
                        codomain.clone(),
                        universe.close(&generators),
                    ))
                })
                .ok_or(TransferMapError::PushforwardImageMissing)
        })
        .collect()
}

fn materialize_pullbacks<M, F>(
    map: &M,
    domain: &Lattice,
    codomain: &Lattice,
    source_systems: &[TransferSystem],
    target_id: F,
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap + ?Sized,
    F: Fn(&TransferSystem) -> Option<ElementId>,
{
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
            let arrows = match (&edge_images, &requirements) {
                (Some(edge_images), None) => raw_inverse_image(source.arrows(), edge_images),
                (None, Some(requirements)) => right_adjoint_raw(source.arrows(), requirements),
                _ => unreachable!("exactly one pullback strategy is selected"),
            };
            target_id(&TransferSystem::new(domain.clone(), arrows))
                .ok_or(TransferMapError::PullbackImageMissing)
        })
        .collect()
}

fn materialize_generated_inverse_images<M, F>(
    map: &M,
    domain: &Lattice,
    codomain: &Lattice,
    source_systems: &[TransferSystem],
    target_id: F,
) -> Result<Vec<ElementId>, TransferMapError>
where
    M: MonotoneMap + ?Sized,
    F: Fn(&TransferSystem) -> Option<ElementId>,
{
    let edge_images = edge_images(map, domain, codomain)?;
    let universe = domain.transfer_universe();
    let raw_inverse_is_closed = map.is_known_meet_preserving();
    source_systems
        .iter()
        .map(|source| {
            let inverse_image = raw_inverse_image(source.arrows(), &edge_images);
            let arrows = if raw_inverse_is_closed {
                inverse_image
            } else {
                universe.close(&inverse_image)
            };
            target_id(&TransferSystem::new(domain.clone(), arrows))
                .ok_or(TransferMapError::GeneratedInverseImageMissing)
        })
        .collect()
}

fn validate_composition_images(
    source: &TransferPoset,
    target: &TransferPoset,
    images: &[ElementId],
) -> Result<(), CompositionMapError> {
    for source_cover in source.sorted_cover_relations() {
        let lower_image = images[source_cover.from];
        let upper_image = images[source_cover.to];
        if target.leq(lower_image, upper_image) {
            continue;
        }

        let failure =
            factorization_failure_between(target.system(lower_image), target.system(upper_image))
                .expect(
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
