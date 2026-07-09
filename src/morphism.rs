//! Order-preserving maps and lattice homomorphisms.
//!
//! A map is represented by its values on element ids: the entry `map[i]` is the
//! image of the element `i`.  Constructors validate the relevant mathematical
//! axioms before producing a [`crate::morphism::PosetMap`] or [`crate::morphism::LatticeMap`].

use crate::lattice::Lattice;
use crate::poset::{ElementId, Poset};
use std::fmt;
use std::sync::Arc;

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

/// A monotone map between finite posets.
///
/// The domain and codomain are reference-counted so maps can be returned
/// together with constructions such as products and coproducts.
#[derive(Debug, Clone)]
pub struct PosetMap<A, B> {
    domain: Arc<Poset<A>>,
    codomain: Arc<Poset<B>>,
    map: Vec<ElementId>,
}

impl<A, B> PosetMap<A, B> {
    /// Constructs a monotone map from its image vector.
    ///
    /// The vector must have length equal to the domain size, each image must be
    /// an element id in the codomain, and `x <= y` in the domain must imply
    /// `f(x) <= f(y)` in the codomain.
    pub fn new(
        domain: Arc<Poset<A>>,
        codomain: Arc<Poset<B>>,
        map: Vec<ElementId>,
    ) -> Result<Self, PosetMapError> {
        validate_poset_map(domain.as_ref(), codomain.as_ref(), &map)?;
        Ok(Self {
            domain,
            codomain,
            map,
        })
    }

    /// Returns the domain poset.
    pub fn domain(&self) -> &Arc<Poset<A>> {
        &self.domain
    }

    /// Returns the codomain poset.
    pub fn codomain(&self) -> &Arc<Poset<B>> {
        &self.codomain
    }

    /// Returns the image vector for this map.
    ///
    /// The entry at index `i` is the image of element `i` in the domain.
    pub fn map(&self) -> &[ElementId] {
        &self.map
    }

    /// Applies the map to a domain element.
    ///
    /// Returns `None` when the supplied element id is outside the domain.
    pub fn apply(&self, element: ElementId) -> Option<ElementId> {
        self.map.get(element).copied()
    }

    pub(crate) fn from_validated(
        domain: Arc<Poset<A>>,
        codomain: Arc<Poset<B>>,
        map: Vec<ElementId>,
    ) -> Self {
        debug_assert!(validate_poset_map(domain.as_ref(), codomain.as_ref(), &map).is_ok());
        Self {
            domain,
            codomain,
            map,
        }
    }
}

/// A lattice homomorphism between finite lattices.
///
/// This is a function preserving bottom, top, binary meets, and binary joins.
/// In finite lattices such a function is automatically monotone, but the
/// constructor also checks monotonicity for clearer diagnostics.
#[derive(Debug, Clone)]
pub struct LatticeMap<A, B> {
    domain: Arc<Lattice<A>>,
    codomain: Arc<Lattice<B>>,
    map: Vec<ElementId>,
}

impl<A, B> LatticeMap<A, B> {
    /// Constructs a lattice homomorphism from its image vector.
    ///
    /// The vector must define a monotone map of the underlying posets and must
    /// preserve bottom, top, all binary meets, and all binary joins.
    pub fn new(
        domain: Arc<Lattice<A>>,
        codomain: Arc<Lattice<B>>,
        map: Vec<ElementId>,
    ) -> Result<Self, LatticeMapError> {
        validate_poset_map(domain.as_poset(), codomain.as_poset(), &map)?;

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

        for i in 0..domain.size() {
            for j in 0..domain.size() {
                let meet_image = map[domain.meet_id(i, j)];
                let image_meet = codomain.meet_id(map[i], map[j]);
                if meet_image != image_meet {
                    return Err(LatticeMapError::DoesNotPreserveMeet { left: i, right: j });
                }

                let join_image = map[domain.join_id(i, j)];
                let image_join = codomain.join_id(map[i], map[j]);
                if join_image != image_join {
                    return Err(LatticeMapError::DoesNotPreserveJoin { left: i, right: j });
                }
            }
        }

        Ok(Self {
            domain,
            codomain,
            map,
        })
    }

    /// Returns the domain lattice.
    pub fn domain(&self) -> &Arc<Lattice<A>> {
        &self.domain
    }

    /// Returns the codomain lattice.
    pub fn codomain(&self) -> &Arc<Lattice<B>> {
        &self.codomain
    }

    /// Returns the image vector for this homomorphism.
    ///
    /// The entry at index `i` is the image of element `i` in the domain.
    pub fn map(&self) -> &[ElementId] {
        &self.map
    }

    /// Applies the homomorphism to a domain element.
    ///
    /// Returns `None` when the supplied element id is outside the domain.
    pub fn apply(&self, element: ElementId) -> Option<ElementId> {
        self.map.get(element).copied()
    }

    /// Forgets the lattice structure and returns the underlying monotone map.
    pub fn as_poset_map(&self) -> PosetMap<A, B>
    where
        A: Clone,
        B: Clone,
    {
        PosetMap::from_validated(
            Arc::new(self.domain.as_poset().clone()),
            Arc::new(self.codomain.as_poset().clone()),
            self.map.clone(),
        )
    }
}

fn validate_poset_map<A, B>(
    domain: &Poset<A>,
    codomain: &Poset<B>,
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
