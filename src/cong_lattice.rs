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
    pub fn leq(&self, a: usize, b: usize) -> bool {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        self.poset_relation[a][b]
    }
    pub fn meet(&mut self, a: usize, b: usize) -> usize {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        if let Some(&result) = self.meet_cache.get(&(a, b)) {
            return result;
        }
        // Find the greatest lower bound (meet) of a and b.
        let mut meet = None;
        for i in 0..self.num_elements {
            if self.leq(i, a) && self.leq(i, b) {
                if meet.is_none() || self.leq(meet.unwrap(), i) {
                    meet = Some(i);
                }
            }
        }
        let result = meet.expect("Meet not found");
        self.meet_cache.insert((a, b), result);
        result
    }
    pub fn join(&mut self, a: usize, b: usize) -> usize {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        if let Some(&result) = self.join_cache.get(&(a, b)) {
            return result;
        }
        // Find the least upper bound (join) of a and b.
        let mut join = None;
        for i in 0..self.num_elements {
            if self.leq(a, i) && self.leq(b, i) {
                if join.is_none() || self.leq(i, join.unwrap()) {
                    join = Some(i);
                }
            }
        }
        let result = join.expect("Join not found");
        self.join_cache.insert((a, b), result);
        result
    }
    // Returns the sublattice spanned by `elements` and a vector v such that
    // v[i] is the index of the i-th element of the sublattice in the original
    // lattice.
    pub fn sublattice(self, elements: Vec<usize>) -> (Self, Vec<usize>) {
        unimplemented!("sublattice method is not yet implemented");
    }
}

impl Debug for CongLattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CongLattice with {} elements", self.num_elements)
    }
}

// Test meet and join
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meet() {
        let mut lattice = CongLattice::total_order(5);
        // Check that meet(a,b) == a iff a <= b
        for i in 0..5 {
            for j in i..5 {
                assert_eq!(lattice.meet(i, j), i);
                assert_eq!(lattice.meet(j, i), i);
            }
        }
    }

    #[test]
    fn test_join() {
        let mut lattice = CongLattice::total_order(5);
        // Check that join(a,b) == b iff a <= b
        for i in 0..5 {
            for j in i..5 {
                assert_eq!(lattice.join(i, j), j);
                assert_eq!(lattice.join(j, i), j);
            }
        }
    }
}
