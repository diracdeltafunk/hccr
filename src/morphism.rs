//! Order-preserving maps and lattice homomorphisms.
//!
//! A map is represented by its values on element ids: the entry `map[i]` is the
//! image of the element `i`.  Constructors validate the relevant mathematical
//! axioms before producing a [`crate::morphism::PosetMap`] or [`crate::morphism::LatticeMap`].

use crate::lattice::Lattice;
use crate::poset::{ElementId, Poset};
use std::fmt;

/// Errors that can occur while constructing a monotone map of posets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosetMapError {
    /// The image vector does not have one entry for each element of the domain.
    WrongLength {
        /// The required length, equal to the domain size.
        expected: usize,
        /// The supplied length.
        actual: usize,
    },
    /// An image is not an element of the codomain.
    ImageOutOfBounds {
        /// The domain element whose image is invalid.
        element: ElementId,
        /// The invalid codomain id.
        image: ElementId,
        /// The number of elements in the codomain.
        codomain_len: usize,
    },
    /// The map fails to preserve the order relation.
    NotMonotone {
        /// The lower element in a relation `lower <= upper` of the domain.
        lower: ElementId,
        /// The upper element in a relation `lower <= upper` of the domain.
        upper: ElementId,
        /// The image of `lower`.
        lower_image: ElementId,
        /// The image of `upper`.
        upper_image: ElementId,
    },
    /// Two maps cannot be composed because the codomain of the first is not
    /// the domain of the second.
    NotComposable,
}

impl fmt::Display for PosetMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PosetMapError::WrongLength { expected, actual } => {
                write!(f, "map has length {actual}, expected {expected}")
            }
            PosetMapError::ImageOutOfBounds {
                element,
                image,
                codomain_len,
            } => write!(
                f,
                "image of {element} is {image}, out of bounds for codomain with {codomain_len} elements"
            ),
            PosetMapError::NotMonotone {
                lower,
                upper,
                lower_image,
                upper_image,
            } => write!(
                f,
                "map is not monotone: {lower} <= {upper}, but {lower_image} is not <= {upper_image}"
            ),
            PosetMapError::NotComposable => write!(
                f,
                "cannot compose maps: the codomain of the first is not the domain of the second"
            ),
        }
    }
}

impl std::error::Error for PosetMapError {}

/// Errors that can occur while constructing a lattice homomorphism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeMapError {
    /// The underlying function is not a monotone map of posets.
    Poset(PosetMapError),
    /// The map does not preserve a binary meet.
    DoesNotPreserveMeet {
        /// The first element in the failing meet.
        left: ElementId,
        /// The second element in the failing meet.
        right: ElementId,
    },
    /// The map does not preserve a binary join.
    DoesNotPreserveJoin {
        /// The first element in the failing join.
        left: ElementId,
        /// The second element in the failing join.
        right: ElementId,
    },
    /// The bottom element of the domain is not sent to the bottom element of
    /// the codomain.
    DoesNotPreserveBottom {
        /// The codomain bottom.
        expected: ElementId,
        /// The actual image of the domain bottom.
        actual: ElementId,
    },
    /// The top element of the domain is not sent to the top element of the
    /// codomain.
    DoesNotPreserveTop {
        /// The codomain top.
        expected: ElementId,
        /// The actual image of the domain top.
        actual: ElementId,
    },
}

impl fmt::Display for LatticeMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeMapError::Poset(error) => write!(f, "{error}"),
            LatticeMapError::DoesNotPreserveMeet { left, right } => {
                write!(f, "map does not preserve meet of {left} and {right}")
            }
            LatticeMapError::DoesNotPreserveJoin { left, right } => {
                write!(f, "map does not preserve join of {left} and {right}")
            }
            LatticeMapError::DoesNotPreserveBottom { expected, actual } => {
                write!(f, "map sends bottom to {actual}, expected {expected}")
            }
            LatticeMapError::DoesNotPreserveTop { expected, actual } => {
                write!(f, "map sends top to {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for LatticeMapError {}

impl From<PosetMapError> for LatticeMapError {
    fn from(error: PosetMapError) -> Self {
        Self::Poset(error)
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Read-only access to the data defining a monotone map of finite posets.
///
/// Both [`PosetMap`] and [`LatticeMap`] implement this trait, so algorithms
/// that require only monotonicity can accept either kind of map without
/// discarding the stronger lattice structure. This trait is sealed; construct
/// one of those validated map types rather than implementing it directly.
pub trait MonotoneMap: sealed::Sealed {
    /// Returns the domain's underlying poset.
    fn domain_poset(&self) -> &Poset;

    /// Returns the codomain's underlying poset.
    fn codomain_poset(&self) -> &Poset;

    /// Returns the image vector of the underlying function.
    ///
    /// Entry `i` is the element id to which domain element `i` is sent.
    fn images(&self) -> &[ElementId];

    /// Reports whether this map is guaranteed to preserve every binary meet.
    ///
    /// This is a conservative optimization capability: algorithms may use a
    /// faster path whose correctness depends on meet preservation when this
    /// method returns `true`. Implementations must therefore return `true`
    /// only when they guarantee that property. The default makes no such
    /// guarantee.
    fn is_known_meet_preserving(&self) -> bool {
        false
    }
}

/// A monotone map between finite posets.
///
/// The map is stored by its values on element ids. Cloning a map is cheap
/// apart from copying that vector, since the domain and codomain are shared
/// handles.
#[derive(Clone, PartialEq, Eq)]
pub struct PosetMap {
    domain: Poset,
    codomain: Poset,
    map: Vec<ElementId>,
}

impl PosetMap {
    /// Constructs a monotone map from its image vector.
    ///
    /// The vector must have length equal to the domain size, each image must be
    /// an element id in the codomain, and `x <= y` in the domain must imply
    /// `f(x) <= f(y)` in the codomain. A lattice may be passed wherever a
    /// poset is expected.
    pub fn new(
        domain: &Poset,
        codomain: &Poset,
        images: Vec<ElementId>,
    ) -> Result<Self, PosetMapError> {
        validate_poset_map(domain, codomain, &images)?;
        Ok(Self {
            domain: domain.clone(),
            codomain: codomain.clone(),
            map: images,
        })
    }

    /// Constructs a monotone map from a function on element ids.
    pub fn from_fn<F>(domain: &Poset, codomain: &Poset, f: F) -> Result<Self, PosetMapError>
    where
        F: FnMut(ElementId) -> ElementId,
    {
        Self::new(domain, codomain, domain.ids().map(f).collect())
    }

    /// Returns the identity map of a poset.
    pub fn identity(poset: &Poset) -> Self {
        Self::from_validated(poset.clone(), poset.clone(), poset.ids().collect())
    }

    pub(crate) fn from_validated(domain: Poset, codomain: Poset, map: Vec<ElementId>) -> Self {
        debug_assert!(validate_poset_map(&domain, &codomain, &map).is_ok());
        Self {
            domain,
            codomain,
            map,
        }
    }

    /// Returns the domain poset.
    pub fn domain(&self) -> &Poset {
        &self.domain
    }

    /// Returns the codomain poset.
    pub fn codomain(&self) -> &Poset {
        &self.codomain
    }

    /// Returns the image vector for this map.
    ///
    /// The entry at index `i` is the image of element `i` in the domain.
    pub fn images(&self) -> &[ElementId] {
        &self.map
    }

    /// Applies the map to a domain element.
    ///
    /// Panics if `element` is not an element id of the domain.
    pub fn apply(&self, element: ElementId) -> ElementId {
        self.map[element]
    }

    /// Returns the composite `self ∘ first`: apply `first`, then `self`.
    ///
    /// The codomain of `first` must equal the domain of `self`.
    pub fn compose(&self, first: &impl MonotoneMap) -> Result<PosetMap, PosetMapError> {
        if first.codomain_poset() != &self.domain {
            return Err(PosetMapError::NotComposable);
        }
        Ok(Self::from_validated(
            first.domain_poset().clone(),
            self.codomain.clone(),
            first.images().iter().map(|&x| self.map[x]).collect(),
        ))
    }
}

impl sealed::Sealed for PosetMap {}

impl MonotoneMap for PosetMap {
    fn domain_poset(&self) -> &Poset {
        &self.domain
    }

    fn codomain_poset(&self) -> &Poset {
        &self.codomain
    }

    fn images(&self) -> &[ElementId] {
        &self.map
    }
}

/// A lattice homomorphism between finite lattices.
///
/// This is a function preserving bottom, top, binary meets, and binary joins.
/// In finite lattices such a function is automatically monotone, but the
/// constructor also checks monotonicity for clearer diagnostics.
#[derive(Clone, PartialEq, Eq)]
pub struct LatticeMap {
    domain: Lattice,
    codomain: Lattice,
    map: Vec<ElementId>,
}

impl LatticeMap {
    /// Constructs a lattice homomorphism from its image vector.
    ///
    /// The vector must define a monotone map of the underlying posets and must
    /// preserve bottom, top, all binary meets, and all binary joins.
    pub fn new(
        domain: &Lattice,
        codomain: &Lattice,
        images: Vec<ElementId>,
    ) -> Result<Self, LatticeMapError> {
        validate_lattice_map(domain, codomain, &images)?;
        Ok(Self {
            domain: domain.clone(),
            codomain: codomain.clone(),
            map: images,
        })
    }

    /// Constructs a lattice homomorphism from a function on element ids.
    pub fn from_fn<F>(domain: &Lattice, codomain: &Lattice, f: F) -> Result<Self, LatticeMapError>
    where
        F: FnMut(ElementId) -> ElementId,
    {
        Self::new(domain, codomain, domain.ids().map(f).collect())
    }

    /// Returns the identity homomorphism of a lattice.
    pub fn identity(lattice: &Lattice) -> Self {
        Self::from_validated(lattice.clone(), lattice.clone(), lattice.ids().collect())
    }

    pub(crate) fn from_validated(domain: Lattice, codomain: Lattice, map: Vec<ElementId>) -> Self {
        debug_assert!(validate_lattice_map(&domain, &codomain, &map).is_ok());
        Self {
            domain,
            codomain,
            map,
        }
    }

    /// Returns the domain lattice.
    pub fn domain(&self) -> &Lattice {
        &self.domain
    }

    /// Returns the codomain lattice.
    pub fn codomain(&self) -> &Lattice {
        &self.codomain
    }

    /// Returns the image vector for this homomorphism.
    ///
    /// The entry at index `i` is the image of element `i` in the domain.
    pub fn images(&self) -> &[ElementId] {
        &self.map
    }

    /// Applies the homomorphism to a domain element.
    ///
    /// Panics if `element` is not an element id of the domain.
    pub fn apply(&self, element: ElementId) -> ElementId {
        self.map[element]
    }

    /// Returns the composite `self ∘ first`: apply `first`, then `self`.
    ///
    /// The codomain of `first` must equal the domain of `self`.
    pub fn compose(&self, first: &LatticeMap) -> Result<LatticeMap, PosetMapError> {
        if first.codomain != self.domain {
            return Err(PosetMapError::NotComposable);
        }
        Ok(Self::from_validated(
            first.domain.clone(),
            self.codomain.clone(),
            first.map.iter().map(|&x| self.map[x]).collect(),
        ))
    }

    /// Forgets the lattice structure and returns the underlying monotone map.
    pub fn as_poset_map(&self) -> PosetMap {
        PosetMap::from_validated(
            self.domain.as_poset().clone(),
            self.codomain.as_poset().clone(),
            self.map.clone(),
        )
    }
}

impl sealed::Sealed for LatticeMap {}

impl MonotoneMap for LatticeMap {
    fn domain_poset(&self) -> &Poset {
        self.domain.as_poset()
    }

    fn codomain_poset(&self) -> &Poset {
        self.codomain.as_poset()
    }

    fn images(&self) -> &[ElementId] {
        &self.map
    }

    fn is_known_meet_preserving(&self) -> bool {
        true
    }
}

impl From<LatticeMap> for PosetMap {
    fn from(map: LatticeMap) -> Self {
        PosetMap::from_validated(map.domain.into(), map.codomain.into(), map.map)
    }
}

fn fmt_map(
    name: &str,
    domain: &Poset,
    codomain: &Poset,
    map: &[ElementId],
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    write!(f, "{name} {{")?;
    for (x, &y) in map.iter().enumerate() {
        if x > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{} |-> {}", domain.label(x), codomain.label(y))?;
    }
    f.write_str("}")
}

impl fmt::Display for PosetMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_map("PosetMap", &self.domain, &self.codomain, &self.map, f)
    }
}

impl fmt::Debug for PosetMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for LatticeMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_map("LatticeMap", &self.domain, &self.codomain, &self.map, f)
    }
}

impl fmt::Debug for LatticeMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn validate_lattice_map(
    domain: &Lattice,
    codomain: &Lattice,
    map: &[ElementId],
) -> Result<(), LatticeMapError> {
    validate_poset_map(domain, codomain, map)?;

    let mapped_bottom = map[domain.bottom()];
    if mapped_bottom != codomain.bottom() {
        return Err(LatticeMapError::DoesNotPreserveBottom {
            expected: codomain.bottom(),
            actual: mapped_bottom,
        });
    }

    let mapped_top = map[domain.top()];
    if mapped_top != codomain.top() {
        return Err(LatticeMapError::DoesNotPreserveTop {
            expected: codomain.top(),
            actual: mapped_top,
        });
    }

    // Meet and join are commutative, and the diagonal laws are automatic.
    for i in 0..domain.size() {
        for j in (i + 1)..domain.size() {
            if map[domain.meet(i, j)] != codomain.meet(map[i], map[j]) {
                return Err(LatticeMapError::DoesNotPreserveMeet { left: i, right: j });
            }
            if map[domain.join(i, j)] != codomain.join(map[i], map[j]) {
                return Err(LatticeMapError::DoesNotPreserveJoin { left: i, right: j });
            }
        }
    }
    Ok(())
}

fn validate_poset_map(
    domain: &Poset,
    codomain: &Poset,
    map: &[ElementId],
) -> Result<(), PosetMapError> {
    if map.len() != domain.size() {
        return Err(PosetMapError::WrongLength {
            expected: domain.size(),
            actual: map.len(),
        });
    }
    for (element, &image) in map.iter().enumerate() {
        if image >= codomain.size() {
            return Err(PosetMapError::ImageOutOfBounds {
                element,
                image,
                codomain_len: codomain.size(),
            });
        }
    }
    for edge in domain.proper_relations_iter() {
        let lower_image = map[edge.from];
        let upper_image = map[edge.to];
        if !codomain.leq(lower_image, upper_image) {
            return Err(PosetMapError::NotMonotone {
                lower: edge.from,
                upper: edge.to,
                lower_image,
                upper_image,
            });
        }
    }
    Ok(())
}
