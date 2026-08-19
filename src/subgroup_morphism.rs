//! Maps of subgroup lattices induced by GAP group homomorphisms.
//!
//! If `f: G -> H` is a group homomorphism, direct image sends a subgroup
//! `K <= G` to `f(K) <= H`, while inverse image sends `J <= H` to
//! `f^-1(J) <= G`. Both operations preserve inclusion and hence define
//! monotone maps between subgroup lattices. A
//! [`SubgroupMaps`](crate::subgroup_morphism::SubgroupMaps) value asks GAP to
//! compute both maps once, validates them, and retains their element-id vectors.
//!
//! It also retains shared pointers to the precise
//! [`SubgroupGLattice`](crate::g_lattice::SubgroupGLattice) endpoints. This
//! matters because element ids depend on GAP's concrete subgroup enumeration
//! and are not interchangeable merely because two subgroup lattices happen to
//! be isomorphic.

use crate::g_lattice::SubgroupGLattice;
use crate::group_theory::{self, GapSubgroup, GroupTheoryError};
use crate::morphism::{PosetMap, PosetMapError};
use crate::poset::ElementId;
use gap_sys::GapValue;
use std::fmt;
use std::sync::Arc;

/// The subgroup-lattice maps induced by a concrete GAP group homomorphism.
///
/// Construction calls GAP to compute all subgroup images and inverse images,
/// then caches the resulting element-id vectors as [`PosetMap`]s. Cloning this
/// value shares its endpoint lattices and GAP roots; it does not repeat those
/// computations.
#[derive(Debug, Clone)]
pub struct SubgroupMaps {
    homomorphism: GapValue,
    domain: Arc<SubgroupGLattice>,
    codomain: Arc<SubgroupGLattice>,
    image_map: PosetMap<GapSubgroup, GapSubgroup>,
    preimage_map: PosetMap<GapSubgroup, GapSubgroup>,
    is_injective: bool,
    is_surjective: bool,
}

/// Errors that can occur while constructing subgroup-lattice maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgroupMapError {
    /// An error reported by the GAP interface.
    Gap(String),
    /// The supplied GAP object is not a group homomorphism.
    NotAGroupHomomorphism,
    /// The GAP homomorphism's source is not the concrete domain group.
    SourceMismatch,
    /// The GAP homomorphism's range is not the concrete codomain group.
    RangeMismatch,
    /// GAP computed an image absent from the codomain subgroup enumeration.
    ImageSubgroupNotFound {
        /// The domain subgroup's lattice element id.
        subgroup: ElementId,
    },
    /// GAP computed an inverse image absent from the domain enumeration.
    PreimageSubgroupNotFound {
        /// The codomain subgroup's lattice element id.
        subgroup: ElementId,
    },
    /// Validation of one of the induced monotone maps failed.
    PosetMap(PosetMapError),
}

impl fmt::Display for SubgroupMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gap(error) => write!(f, "{error}"),
            Self::NotAGroupHomomorphism => {
                write!(f, "GAP object is not a group homomorphism")
            }
            Self::SourceMismatch => write!(
                f,
                "group homomorphism source is not the supplied domain group"
            ),
            Self::RangeMismatch => write!(
                f,
                "group homomorphism range is not the supplied codomain group"
            ),
            Self::ImageSubgroupNotFound { subgroup } => write!(
                f,
                "GAP did not find the image of domain subgroup {subgroup} in the codomain subgroup lattice"
            ),
            Self::PreimageSubgroupNotFound { subgroup } => write!(
                f,
                "GAP did not find the inverse image of codomain subgroup {subgroup} in the domain subgroup lattice"
            ),
            Self::PosetMap(error) => write!(f, "induced subgroup map is not monotone: {error}"),
        }
    }
}

impl std::error::Error for SubgroupMapError {}

impl From<PosetMapError> for SubgroupMapError {
    fn from(error: PosetMapError) -> Self {
        Self::PosetMap(error)
    }
}

impl From<GroupTheoryError> for SubgroupMapError {
    fn from(error: GroupTheoryError) -> Self {
        match error {
            GroupTheoryError::Gap(error) => Self::Gap(error),
            GroupTheoryError::NotAGroupHomomorphism => Self::NotAGroupHomomorphism,
            GroupTheoryError::HomomorphismSourceMismatch => Self::SourceMismatch,
            GroupTheoryError::HomomorphismRangeMismatch => Self::RangeMismatch,
            GroupTheoryError::SubgroupImageNotFound { subgroup } => {
                Self::ImageSubgroupNotFound { subgroup }
            }
            GroupTheoryError::SubgroupPreimageNotFound { subgroup } => {
                Self::PreimageSubgroupNotFound { subgroup }
            }
            other => Self::Gap(other.to_string()),
        }
    }
}

impl SubgroupMaps {
    /// Computes the subgroup image and inverse-image maps induced by `homomorphism`.
    ///
    /// GAP's `Source(homomorphism)` and `Range(homomorphism)` must be the exact
    /// group objects underlying `domain` and `codomain`, respectively. This
    /// protects the element ids in the cached maps from being paired with a
    /// different subgroup enumeration or conjugation action.
    pub fn new(
        homomorphism: &GapValue,
        domain: Arc<SubgroupGLattice>,
        codomain: Arc<SubgroupGLattice>,
    ) -> Result<Self, SubgroupMapError> {
        let mut gap = group_theory::global_gap()?;
        let data = group_theory::subgroup_maps_data(
            &mut gap,
            homomorphism,
            group_theory::SubgroupLatticeView {
                group: domain.g_lattice().group(),
                subgroups: domain.subgroups(),
                subgroup_list: domain.subgroup_list(),
            },
            group_theory::SubgroupLatticeView {
                group: codomain.g_lattice().group(),
                subgroups: codomain.subgroups(),
                subgroup_list: codomain.subgroup_list(),
            },
        )?;

        let image_map = PosetMap::between_lattices(
            domain.lattice().as_ref(),
            codomain.lattice().as_ref(),
            data.image_map,
        )?;
        let preimage_map = PosetMap::between_lattices(
            codomain.lattice().as_ref(),
            domain.lattice().as_ref(),
            data.preimage_map,
        )?;

        Ok(Self {
            homomorphism: homomorphism.clone(),
            domain,
            codomain,
            image_map,
            preimage_map,
            is_injective: data.is_injective,
            is_surjective: data.is_surjective,
        })
    }

    /// Returns the rooted GAP group homomorphism.
    pub fn homomorphism(&self) -> &GapValue {
        &self.homomorphism
    }

    /// Returns the precise subgroup G-lattice used as the domain.
    pub fn domain(&self) -> &Arc<SubgroupGLattice> {
        &self.domain
    }

    /// Returns the precise subgroup G-lattice used as the codomain.
    pub fn codomain(&self) -> &Arc<SubgroupGLattice> {
        &self.codomain
    }

    /// Returns the monotone map `K |-> f(K)` from `Sub(G)` to `Sub(H)`.
    pub fn image_map(&self) -> &PosetMap<GapSubgroup, GapSubgroup> {
        &self.image_map
    }

    /// Returns the monotone map `J |-> f^{-1}(J)` from `Sub(H)` to `Sub(G)`.
    pub fn preimage_map(&self) -> &PosetMap<GapSubgroup, GapSubgroup> {
        &self.preimage_map
    }

    /// Returns whether GAP reports the group homomorphism as injective.
    pub fn is_injective(&self) -> bool {
        self.is_injective
    }

    /// Returns whether GAP reports the group homomorphism as surjective.
    pub fn is_surjective(&self) -> bool {
        self.is_surjective
    }

    /// Tests whether `candidate` is the exact stored domain G-lattice.
    pub fn has_domain(&self, candidate: &Arc<SubgroupGLattice>) -> bool {
        Arc::ptr_eq(&self.domain, candidate)
    }

    /// Tests whether `candidate` is the exact stored codomain G-lattice.
    pub fn has_codomain(&self, candidate: &Arc<SubgroupGLattice>) -> bool {
        Arc::ptr_eq(&self.codomain, candidate)
    }
}
