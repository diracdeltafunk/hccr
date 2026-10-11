//! Transfer systems along monotone maps between finite lattices.
//!
//! A monotone map `f: L -> M` sends a relation `x <= y` to `f(x) <= f(y)`.
//! Sending every relation of a transfer system this way may break restriction
//! closure, so there are three operations, all methods of [`TransferSystem`]:
//!
//! - [`TransferSystem::pushforward`] takes the least transfer system on `M`
//!   containing the image relations;
//! - [`TransferSystem::pullback`] takes the greatest transfer system on `L`
//!   contained in the inverse image. It is right adjoint to pushforward: for
//!   transfer systems `S` on `L` and `T` on `M`, `pushforward(S) <= T` exactly
//!   when `S <= pullback(T)`;
//! - [`TransferSystem::generated_inverse_image`] takes the *least* transfer
//!   system containing the inverse image. In general it differs from the
//!   pullback and is not adjoint to pushforward, but the two agree when `f`
//!   preserves meets, for instance for every
//!   [`LatticeMap`](crate::morphism::LatticeMap).
//!
//! For merely monotone maps these operations need not respect composition,
//! because closing at an intermediate lattice can change the result. They do
//! respect composition of lattice homomorphisms.
//!
//! The free functions in this module package the same operations as maps of
//! posets: between the containment lattices `Tr(L)` and `Tr(M)` (for example
//! [`pushforward_containment_map`]), and, when monotone, between the
//! composition-closed orders (for example [`try_pushforward_composition_map`],
//! which reports why it fails when the operation is not monotone).
//!
//! ```
//! use hccr::lattice::Lattice;
//! use hccr::morphism::LatticeMap;
//!
//! // The quotient of the chain 0 < 1 < 2 onto 0 < 1 identifying 0 and 1.
//! let three = Lattice::chain(2);
//! let two = Lattice::chain(1);
//! let quotient = LatticeMap::new(&three, &two, vec![0, 0, 1])?;
//!
//! let system = three.transfer_system_generated_by([(1, 2)])?;
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
    TransferLattice, TransferOrder, TransferPoset, TransferSystem, factorization_failure_between,
};
use bitvec::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Errors produced while computing a transfer system along a monotone map, or
/// a map between orders of transfer systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferMapError {
    /// A transfer system, or order of transfer systems, that should live on
    /// the monotone map's domain lives on a different lattice.
    DomainMismatch,
    /// A transfer system, or order of transfer systems, that should live on
    /// the monotone map's codomain lives on a different lattice.
    CodomainMismatch,
    /// The monotone map's domain or codomain is not a lattice.
    NotALattice,
    /// The image `f(x) -> f(y)` of a relation `x -> y` of the domain is not a
    /// relation of the codomain.
    ///
    /// This would mean the map is not monotone, so it cannot occur for the
    /// validated maps [`PosetMap`] and
    /// [`LatticeMap`](crate::morphism::LatticeMap).
    InvalidImageRelation {
        /// The relation `x -> y` of the domain.
        source: Edge,
        /// Its image `f(x) -> f(y)`.
        image: Edge,
    },
    /// The pushforward of some transfer system is not an element of the
    /// supplied order on the map's codomain (for example, because that order
    /// is [`Lattice::saturated_transfer_systems`]).
    PushforwardImageMissing,
    /// The pullback of some transfer system is not an element of the supplied
    /// order on the map's domain.
    PullbackImageMissing,
    /// The generated inverse image of some transfer system is not an element
    /// of the supplied order on the map's domain.
    GeneratedInverseImageMissing,
}

impl fmt::Display for TransferMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferMapError::DomainMismatch => write!(
                formatter,
                "a transfer system or order that should live on the map's domain lives on a different lattice"
            ),
            TransferMapError::CodomainMismatch => write!(
                formatter,
                "a transfer system or order that should live on the map's codomain lives on a different lattice"
            ),
            TransferMapError::NotALattice => write!(
                formatter,
                "transfer systems need the map's domain and codomain to be lattices"
            ),
            TransferMapError::InvalidImageRelation { source, image } => write!(
                formatter,
                "relation {} -> {} maps to {} -> {}, which is not a relation of the codomain",
                source.from, source.to, image.from, image.to
            ),
            TransferMapError::PushforwardImageMissing => write!(
                formatter,
                "a pushforward is not an element of the supplied order on the map's codomain"
            ),
            TransferMapError::PullbackImageMissing => write!(
                formatter,
                "a pullback is not an element of the supplied order on the map's domain"
            ),
            TransferMapError::GeneratedInverseImageMissing => write!(
                formatter,
                "a generated inverse image is not an element of the supplied order on the map's domain"
            ),
        }
    }
}

impl std::error::Error for TransferMapError {}

/// Failure to construct an induced map between composition-closed orders of
/// transfer systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionMapError {
    /// Computing the induced map itself failed; see [`TransferMapError`].
    Transfer(TransferMapError),
    /// A supplied [`TransferPoset`] is not a composition-closed order, that is,
    /// it was not returned by
    /// [`Lattice::transfer_systems_composition_closed`].
    NotCompositionClosedOrder,
    /// The induced function is not monotone: it sends a cover of the source
    /// order to a pair that is not related in the target order.
    NotMonotone {
        /// The cover `S -> S'` of the source order, given by element ids of
        /// the source [`TransferPoset`] (not of the underlying lattice).
        source_cover: Edge,
        /// The image of `S`, as an element id of the target [`TransferPoset`].
        lower_image: ElementId,
        /// The image of `S'`, as an element id of the target [`TransferPoset`].
        upper_image: ElementId,
        /// Relations `x -> y` of the image of `S` and `x' -> y'` of the image
        /// of `S'` for which the factorization condition of
        /// [`Lattice::transfer_systems_composition_closed`] fails.
        failed_square: (Edge, Edge),
    },
}

impl fmt::Display for CompositionMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompositionMapError::Transfer(error) => write!(formatter, "{error}"),
            CompositionMapError::NotCompositionClosedOrder => write!(
                formatter,
                "a supplied order of transfer systems is not the composition-closed order"
            ),
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

fn pushforward<M>(map: &M, system: &TransferSystem) -> Result<TransferSystem, TransferMapError>
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

fn pullback<M>(map: &M, system: &TransferSystem) -> Result<TransferSystem, TransferMapError>
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

fn generated_inverse_image<M>(
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
    /// Returns the pushforward of this transfer system along a monotone map
    /// `f` (that is, `map`) out of its lattice: the least transfer system on
    /// the codomain containing `f(x) -> f(y)` for every relation `x -> y` of
    /// this system.
    ///
    /// For merely monotone maps this need not commute with composition; it does
    /// for lattice homomorphisms.
    ///
    /// # Errors
    ///
    /// Returns [`TransferMapError::DomainMismatch`] if this system does not
    /// live on the map's domain, and [`TransferMapError::NotALattice`] if the
    /// map's domain or codomain is not a lattice.
    pub fn pushforward<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pushforward(map, self)
    }

    /// Returns the pullback of this transfer system along a monotone map `f`
    /// (that is, `map`) into its lattice.
    ///
    /// A relation `x -> y` of the domain belongs to the result exactly when,
    /// for every `z <= y`, the relation `f(x /\ z) -> f(z)` belongs to this
    /// system. This is the greatest transfer system contained in the inverse
    /// image (the relations `x -> y` of the domain with `f(x) -> f(y)` in this
    /// system), and it is right adjoint to [`TransferSystem::pushforward`]
    /// under containment.
    ///
    /// For merely monotone maps it need not commute with composition, despite
    /// being right adjoint for each individual map; it does for lattice
    /// homomorphisms.
    ///
    /// # Errors
    ///
    /// Returns [`TransferMapError::CodomainMismatch`] if this system does not
    /// live on the map's codomain, and [`TransferMapError::NotALattice`] if the
    /// map's domain or codomain is not a lattice.
    pub fn pullback<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        pullback(map, self)
    }

    /// Returns the transfer system generated by the inverse image of this one
    /// along a monotone map `f` (that is, `map`) into its lattice: the least
    /// transfer system containing every relation `x -> y` of the domain for
    /// which `f(x) -> f(y)` belongs to this system.
    ///
    /// It is containment-monotone but, unlike [`TransferSystem::pullback`], is
    /// generally not right adjoint to [`TransferSystem::pushforward`]. The two
    /// operations agree whenever the map preserves meets. For merely monotone
    /// maps this operation need not commute with composition; it does for
    /// lattice homomorphisms.
    ///
    /// # Errors
    ///
    /// Returns [`TransferMapError::CodomainMismatch`] if this system does not
    /// live on the map's codomain, and [`TransferMapError::NotALattice`] if the
    /// map's domain or codomain is not a lattice.
    pub fn generated_inverse_image<M>(&self, map: &M) -> Result<TransferSystem, TransferMapError>
    where
        M: MonotoneMap + ?Sized,
    {
        generated_inverse_image(map, self)
    }
}

/// Returns the map `Tr(L) -> Tr(M)` of containment lattices given by
/// [`TransferSystem::pushforward`] along a monotone map `f: L -> M`.
///
/// `domain` and `codomain` must be lattices of transfer systems on `L` and
/// `M`. When they are the full lattices `Tr(L)` and `Tr(M)`, the returned map
/// is left adjoint to [`pullback_containment_map`]. It inherits the
/// composition caveat of [`TransferSystem::pushforward`].
///
/// # Errors
///
/// Returns [`TransferMapError::DomainMismatch`] or
/// [`TransferMapError::CodomainMismatch`] if `domain` or `codomain` is not on
/// `L` or `M` respectively, [`TransferMapError::NotALattice`] if `L` or `M`
/// is not a lattice, and [`TransferMapError::PushforwardImageMissing`] if
/// some pushforward is not an element of `codomain`.
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

/// Returns the map `Tr(M) -> Tr(L)` of containment lattices given by
/// [`TransferSystem::pullback`] along a monotone map `f: L -> M`.
///
/// `codomain` (a lattice of transfer systems on `M`) is the domain of the
/// returned map, and `domain` (on `L`) is its codomain. When they are the
/// full lattices `Tr(M)` and `Tr(L)`, the returned map is right adjoint to
/// [`pushforward_containment_map`]. It inherits the composition caveat of
/// [`TransferSystem::pullback`].
///
/// # Errors
///
/// Returns [`TransferMapError::DomainMismatch`] or
/// [`TransferMapError::CodomainMismatch`] if `domain` or `codomain` is not on
/// `L` or `M` respectively, [`TransferMapError::NotALattice`] if `L` or `M`
/// is not a lattice, and [`TransferMapError::PullbackImageMissing`] if some
/// pullback is not an element of `domain`.
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

/// Returns the map `Tr(M) -> Tr(L)` of containment lattices given by
/// [`TransferSystem::generated_inverse_image`] along a monotone map
/// `f: L -> M`.
///
/// `codomain` (a lattice of transfer systems on `M`) is the domain of the
/// returned map, and `domain` (on `L`) is its codomain. This operation is
/// always containment-monotone, but unlike [`pullback_containment_map`] it is
/// not generally right adjoint to [`pushforward_containment_map`].
///
/// # Errors
///
/// Returns [`TransferMapError::DomainMismatch`] or
/// [`TransferMapError::CodomainMismatch`] if `domain` or `codomain` is not on
/// `L` or `M` respectively, [`TransferMapError::NotALattice`] if `L` or `M`
/// is not a lattice, and [`TransferMapError::GeneratedInverseImageMissing`] if
/// some generated inverse image is not an element of `domain`.
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

/// Returns the map between composition-closed orders given by
/// [`TransferSystem::pushforward`] along a monotone map `f: L -> M`, if it is
/// monotone.
///
/// `domain` and `codomain` must be the composition-closed orders
/// [`Lattice::transfer_systems_composition_closed`] on `L` and `M`. Every
/// cover of `domain` is checked, so the function succeeds exactly when the
/// pushforward is monotone for these orders.
///
/// # Errors
///
/// Returns [`CompositionMapError::NotCompositionClosedOrder`] if `domain` or
/// `codomain` is not a composition-closed order (for example, if it is
/// [`Lattice::transfer_systems_model_structure_order`]),
/// [`CompositionMapError::Transfer`] with the errors of
/// [`pushforward_containment_map`], and [`CompositionMapError::NotMonotone`]
/// with a witnessing cover and pair of relations if the pushforward is not
/// monotone.
pub fn try_pushforward_composition_map<M>(
    map: &M,
    domain: &TransferPoset,
    codomain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    ensure_composition_closed(domain, codomain)?;
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

/// Returns the map between composition-closed orders given by
/// [`TransferSystem::pullback`] along a monotone map `f: L -> M`, if it is
/// monotone.
///
/// `codomain` and `domain` must be the composition-closed orders
/// [`Lattice::transfer_systems_composition_closed`] on `M` and `L`; `codomain`
/// is the domain of the returned map. Every cover of `codomain` is checked,
/// so the function succeeds exactly when the pullback is monotone for these
/// orders.
///
/// # Errors
///
/// Returns [`CompositionMapError::NotCompositionClosedOrder`] if `codomain`
/// or `domain` is not a composition-closed order (for example, if it is
/// [`Lattice::transfer_systems_model_structure_order`]),
/// [`CompositionMapError::Transfer`] with the errors of
/// [`pullback_containment_map`], and [`CompositionMapError::NotMonotone`]
/// with a witnessing cover and pair of relations if the pullback is not
/// monotone.
pub fn try_pullback_composition_map<M>(
    map: &M,
    codomain: &TransferPoset,
    domain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    ensure_composition_closed(codomain, domain)?;
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

/// Returns the map between composition-closed orders given by
/// [`TransferSystem::generated_inverse_image`] along a monotone map
/// `f: L -> M`, if it is monotone.
///
/// `codomain` and `domain` must be the composition-closed orders
/// [`Lattice::transfer_systems_composition_closed`] on `M` and `L`; `codomain`
/// is the domain of the returned map. Every cover of `codomain` is checked,
/// so the function succeeds exactly when the generated inverse image is
/// monotone for these orders.
///
/// # Errors
///
/// Returns [`CompositionMapError::NotCompositionClosedOrder`] if `codomain`
/// or `domain` is not a composition-closed order (for example, if it is
/// [`Lattice::transfer_systems_model_structure_order`]),
/// [`CompositionMapError::Transfer`] with the errors of
/// [`generated_inverse_image_containment_map`], and
/// [`CompositionMapError::NotMonotone`] with a witnessing cover and pair of
/// relations if the generated inverse image is not monotone.
pub fn try_generated_inverse_image_composition_map<M>(
    map: &M,
    codomain: &TransferPoset,
    domain: &TransferPoset,
) -> Result<PosetMap, CompositionMapError>
where
    M: MonotoneMap + ?Sized,
{
    ensure_composition_closed(codomain, domain)?;
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

/// Checks that both orders are composition-closed orders, the precondition
/// of the `try_*_composition_map` functions.
fn ensure_composition_closed(
    first: &TransferPoset,
    second: &TransferPoset,
) -> Result<(), CompositionMapError> {
    if first.kind() == TransferOrder::CompositionClosed
        && second.kind() == TransferOrder::CompositionClosed
    {
        Ok(())
    } else {
        Err(CompositionMapError::NotCompositionClosedOrder)
    }
}

/// Checks that `images` defines a monotone map between two composition-closed
/// orders, given that it is monotone for containment.
fn validate_composition_images(
    source: &TransferPoset,
    target: &TransferPoset,
    images: &[ElementId],
) -> Result<(), CompositionMapError> {
    for source_cover in source.covers() {
        let lower_image = images[source_cover.from];
        let upper_image = images[source_cover.to];
        if target.leq(lower_image, upper_image) {
            continue;
        }

        // This `expect` is unreachable: both orders are composition-closed
        // (checked by `ensure_composition_closed`), so a source cover is in
        // particular a containment, and every induced operation is
        // containment-monotone. The lower image is therefore contained in the
        // upper one, and in the composition-closed target order they can be
        // unrelated only because the factorization condition fails.
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
