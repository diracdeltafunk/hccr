use crate::lattice::Lattice;
use crate::poset::{ElementId, Poset};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosetMapError {
    WrongLength {
        expected: usize,
        actual: usize,
    },
    ImageOutOfBounds {
        element: ElementId,
        image: ElementId,
        codomain_len: usize,
    },
    NotMonotone {
        lower: ElementId,
        upper: ElementId,
        lower_image: ElementId,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeMapError {
    Poset(PosetMapError),
    DoesNotPreserveMeet {
        left: ElementId,
        right: ElementId,
    },
    DoesNotPreserveJoin {
        left: ElementId,
        right: ElementId,
    },
    DoesNotPreserveBottom {
        expected: ElementId,
        actual: ElementId,
    },
    DoesNotPreserveTop {
        expected: ElementId,
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

#[derive(Debug, Clone)]
pub struct PosetMap<A, B> {
    domain: Arc<Poset<A>>,
    codomain: Arc<Poset<B>>,
    map: Vec<ElementId>,
}

impl<A, B> PosetMap<A, B> {
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

    pub fn domain(&self) -> &Arc<Poset<A>> {
        &self.domain
    }

    pub fn codomain(&self) -> &Arc<Poset<B>> {
        &self.codomain
    }

    pub fn map(&self) -> &[ElementId] {
        &self.map
    }

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

#[derive(Debug, Clone)]
pub struct LatticeMap<A, B> {
    domain: Arc<Lattice<A>>,
    codomain: Arc<Lattice<B>>,
    map: Vec<ElementId>,
}

impl<A, B> LatticeMap<A, B> {
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

    pub fn domain(&self) -> &Arc<Lattice<A>> {
        &self.domain
    }

    pub fn codomain(&self) -> &Arc<Lattice<B>> {
        &self.codomain
    }

    pub fn map(&self) -> &[ElementId] {
        &self.map
    }

    pub fn apply(&self, element: ElementId) -> Option<ElementId> {
        self.map.get(element).copied()
    }

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
