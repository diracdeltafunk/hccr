use std::collections::HashMap;
use std::fmt::Debug;

// A CongLattice encodes the data of a finite lattice, together with an
// equivalence relation on its elements which is a congruence relation in the
// sense of universal algebra. The data structure also has some hash tables to
// cache restriction / meet / join / etc. operations.
pub struct CongLattice {
    pub num_elements: usize,
    poset_relation: Vec<Vec<bool>>,
    equivalence_relation: Vec<Vec<bool>>,
    meet_cache: HashMap<(usize, usize), usize>,
    join_cache: HashMap<(usize, usize), usize>,
    restriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
}

impl CongLattice {
    // Construct a total order with trivial equivalence relation.
    pub fn total_order(num_elements: usize) -> Self {
        assert!(num_elements > 0, "Number of elements must be positive.");
        let mut poset_relation = vec![vec![false; num_elements]; num_elements];
        for i in 0..num_elements {
            for j in i..num_elements {
                poset_relation[i][j] = true;
            }
        }
        let mut equivalence_relation = vec![vec![false; num_elements]; num_elements];
        for i in 0..num_elements {
            equivalence_relation[i][i] = true;
        }
        CongLattice {
            num_elements,
            poset_relation,
            equivalence_relation,
            meet_cache: HashMap::new(),
            join_cache: HashMap::new(),
            restriction_cache: HashMap::new(),
        }
    }
    // Additional methods for manipulating the lattice go here.
}

impl Debug for CongLattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CongLattice with {} elements", self.num_elements)
    }
}
