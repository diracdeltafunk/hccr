use crate::lattice::Lattice;
use crate::poset::{ElementId, Poset};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
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

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::WrongLength { expected, actual } => {
                write!(f, "map has length {actual}, expected {expected}")
            }
            MapError::ImageOutOfBounds {
                element,
                image,
                codomain_len,
            } => write!(
                f,
                "image of {element} is {image}, out of bounds for codomain with {codomain_len} elements"
            ),
            MapError::NotMonotone {
                lower,
                upper,
                lower_image,
                upper_image,
            } => write!(
                f,
                "map is not monotone: {lower} <= {upper}, but {lower_image} is not <= {upper_image}"
            ),
            MapError::DoesNotPreserveMeet { left, right } => {
                write!(f, "map does not preserve meet of {left} and {right}")
            }
            MapError::DoesNotPreserveJoin { left, right } => {
                write!(f, "map does not preserve join of {left} and {right}")
            }
            MapError::DoesNotPreserveBottom { expected, actual } => {
                write!(f, "map sends bottom to {actual}, expected {expected}")
            }
            MapError::DoesNotPreserveTop { expected, actual } => {
                write!(f, "map sends top to {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for MapError {}

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
    ) -> Result<Self, MapError> {
        if map.len() != domain.len() {
            return Err(MapError::WrongLength {
                expected: domain.len(),
                actual: map.len(),
            });
        }
        for (element, &image) in map.iter().enumerate() {
            if image >= codomain.len() {
                return Err(MapError::ImageOutOfBounds {
                    element,
                    image,
                    codomain_len: codomain.len(),
                });
            }
        }
        for edge in domain.all_relations_iter() {
            let lower_image = map[edge.from];
            let upper_image = map[edge.to];
            if !codomain.leq(lower_image, upper_image) {
                return Err(MapError::NotMonotone {
                    lower: edge.from,
                    upper: edge.to,
                    lower_image,
                    upper_image,
                });
            }
        }
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

    pub fn apply(&self, element: ElementId) -> ElementId {
        self.map[element]
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
    ) -> Result<Self, MapError> {
        if map.len() != domain.len() {
            return Err(MapError::WrongLength {
                expected: domain.len(),
                actual: map.len(),
            });
        }
        for (element, &image) in map.iter().enumerate() {
            if image >= codomain.len() {
                return Err(MapError::ImageOutOfBounds {
                    element,
                    image,
                    codomain_len: codomain.len(),
                });
            }
        }
        for edge in domain.as_poset().all_relations_iter() {
            let lower_image = map[edge.from];
            let upper_image = map[edge.to];
            if !codomain.leq(lower_image, upper_image) {
                return Err(MapError::NotMonotone {
                    lower: edge.from,
                    upper: edge.to,
                    lower_image,
                    upper_image,
                });
            }
        }

        let mapped_bottom = map[domain.bottom()];
        if mapped_bottom != codomain.bottom() {
            return Err(MapError::DoesNotPreserveBottom {
                expected: codomain.bottom(),
                actual: mapped_bottom,
            });
        }

        let mapped_top = map[domain.top()];
        if mapped_top != codomain.top() {
            return Err(MapError::DoesNotPreserveTop {
                expected: codomain.top(),
                actual: mapped_top,
            });
        }

        for i in 0..domain.len() {
            for j in 0..domain.len() {
                let meet_image = map[domain.meet_id(i, j)];
                let image_meet = codomain.meet_id(map[i], map[j]);
                if meet_image != image_meet {
                    return Err(MapError::DoesNotPreserveMeet { left: i, right: j });
                }

                let join_image = map[domain.join_id(i, j)];
                let image_join = codomain.join_id(map[i], map[j]);
                if join_image != image_join {
                    return Err(MapError::DoesNotPreserveJoin { left: i, right: j });
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

    pub fn apply(&self, element: ElementId) -> ElementId {
        self.map[element]
    }

    pub fn as_poset_map(&self) -> Result<PosetMap<A, B>, MapError>
    where
        A: Clone,
        B: Clone,
    {
        PosetMap::new(
            Arc::new(self.domain.as_poset().clone()),
            Arc::new(self.codomain.as_poset().clone()),
            self.map.clone(),
        )
    }
}
