use binary_matrix::BinaryMatrix64;
use std::collections::HashMap;

mod cong_lattice {
    // A CongLattice encodes the data of a finite lattice, together with an
    // equivalence relation on its elements which is a congruence relation in the
    // sense of universal algebra. The data structure also has some hash tables to
    // cache restriction / meet / join / etc. operations.
    pub struct CongLattice {
        pub num_elements: usize,
        poset_relation: BinaryMatrix64,
        equivalence_relation: BinaryMatrix64,
        meet_cache: HashMap<(usize, usize), usize>,
        join_cache: HashMap<(usize, usize), usize>,
        restriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
    }

    impl CongLattice {
        // Construct a total order with trivial equivalence relation.
        pub fn total_order(num_elements: usize) -> Self {
            assert!(num_elements > 0, "Number of elements must be positive.");
            let mut poset_relation = BinaryMatrix64::zero(num_elements, num_elements);
            for i in 0..num_elements {
                for j in i..num_elements {
                    poset_relation[(i, j)] = 1;
                }
            }
            CongLattice {
                num_elements,
                poset_relation,
                equivalence_relation: BinaryMatrix64::identity(num_elements),
                meet_cache: HashMap::new(),
                join_cache: HashMap::new(),
                restriction_cache: HashMap::new(),
            }
        }
        // Additional methods for manipulating the lattice go here.
    }
}
