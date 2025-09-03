use std::collections::HashMap;
use std::fmt::Debug;

/// Indexes into a lower-triangular matrix
/// e.g. for n = 4 we have
/// 0 1 2 3
/// 4 5 6 _
/// 7 8 _ _
/// 9 _ _ _
/// so 2 -> (0,2) and 7 -> (2,0).
fn idx_to_pair(n: usize, idx: usize) -> (usize, usize) {
    assert!(
        n + 1 < (usize::MAX as f64).sqrt() as usize && 2 * idx < n * (n + 1),
        "Out of bounds!"
    );
    let r = (0.5 + (n as f64) - ((0.5 + (n as f64)).powi(2) - 2.0 * (idx as f64)).sqrt()) as usize;
    (r, idx - (1 + 2 * n - r) * r / 2)
}

// Gets index into a lower-triangular matrix from a pair (r, c)
// e.g. for n = 4 we have
// 0 1 2 3
// 4 5 6 _
// 7 8 _ _
// 9 _ _ _
// so (0,2) -> 2 and (2,1) -> 8.
fn pair_to_idx(n: usize, (r, c): (usize, usize)) -> usize {
    assert!(r < n && c < n, "Indices out of bounds.");
    (1 + 2 * n - r) * r / 2 + c
}

// A CongLattice encodes the data of a finite lattice, together with an
// equivalence relation on its elements which is a congruence relation on the
// underlying poset in the sense of universal algebra. The data structure also
// has some hash tables to cache (co)restriction / meet / join / etc.
pub struct CongLattice {
    pub num_elements: usize,
    poset_relation: Vec<Vec<bool>>,
    // TODO Actually, we should be storing an equivalence relation on edges...
    //      Not sure exactly what axioms this equivalence relation needs to
    //      satisfy. One condition must be that the induced equivalence relation
    //      on the elements (i.e. identity edges) is a poset congruence relation
    equivalence_relation: Vec<Vec<bool>>,
    meet_cache: HashMap<(usize, usize), usize>,
    join_cache: HashMap<(usize, usize), usize>,
    restriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
    corestriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
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
            corestriction_cache: HashMap::new(),
        }
    }
    pub fn leq(&self, a: usize, b: usize) -> bool {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        self.poset_relation[a][b]
    }
    pub fn equiv(&self, a: usize, b: usize) -> bool {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        self.equivalence_relation[a][b]
    }
    // Check if [a] <= [b] in the quotient poset
    pub fn preceq(&self, a: usize, b: usize) -> bool {
        assert!(
            a < self.num_elements && b < self.num_elements,
            "Indices out of bounds."
        );
        for i in 0..self.num_elements {
            if self.equivalence_relation[a][i] && self.poset_relation[i][b] {
                return true;
            }
        }
        false
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
        let mut meet: Option<usize> = None;
        for i in 0..self.num_elements {
            if self.poset_relation[i][a] && self.poset_relation[i][b] {
                if meet.is_none() || self.poset_relation[meet.unwrap()][i] {
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
        let mut join: Option<usize> = None;
        for i in 0..self.num_elements {
            if self.poset_relation[a][i] && self.poset_relation[b][i] {
                if join.is_none() || self.poset_relation[i][join.unwrap()] {
                    join = Some(i);
                }
            }
        }
        let result = join.expect("Join not found");
        self.join_cache.insert((a, b), result);
        result
    }
    // Given an edge a -> b in the lattice, returns the list of all edges
    // meet(x,a) -> a
    // for x <= b
    pub fn restriction(&mut self, a: usize, b: usize) -> Vec<(usize, usize)> {
        assert!(
            a <= b,
            "First argument must be less than or equal to second."
        );
        assert!(b < self.num_elements, "Indices out of bounds.");
        if let Some(result) = self.restriction_cache.get(&(a, b)) {
            return result.clone();
        }
        let mut restriction = Vec::new();
        for i in 0..self.num_elements {
            if self.poset_relation[i][b] {
                restriction.push((self.meet(i, a), a));
            }
        }
        self.restriction_cache.insert((a, b), restriction.clone());
        restriction
    }
    // Given an edge a -> b in the lattice, returns the list of all edges
    // b -> join(x,b)
    // for x >= a
    pub fn corestriction(&mut self, a: usize, b: usize) -> Vec<(usize, usize)> {
        assert!(
            a <= b,
            "First argument must be less than or equal to second."
        );
        assert!(b < self.num_elements, "Indices out of bounds.");
        if let Some(result) = self.corestriction_cache.get(&(a, b)) {
            return result.clone();
        }
        let mut corestriction = Vec::new();
        for i in 0..self.num_elements {
            if self.poset_relation[a][i] {
                corestriction.push((b, self.join(i, b)));
            }
        }
        self.corestriction_cache
            .insert((a, b), corestriction.clone());
        corestriction
    }
    // Returns the equivalence class of the input element
    pub fn equivalence_class(&self, a: usize) -> Vec<usize> {
        assert!(a < self.num_elements, "Index out of bounds.");
        let mut class = Vec::new();
        for i in 0..self.num_elements {
            if self.equivalence_relation[a][i] {
                class.push(i);
            }
        }
        class
    }
    // Takes a subset of elements and returns the union of their equivalence classes
    pub fn equivalence_closure(&self, elements: &[usize]) -> Vec<usize> {
        let mut closure = Vec::new();
        for &elem in elements {
            for i in self.equivalence_class(elem) {
                closure.push(i);
            }
        }
        closure.sort_unstable();
        closure.dedup();
        closure
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
    fn test_idx_to_pair() {
        assert_eq!(idx_to_pair(4, 0), (0, 0));
        assert_eq!(idx_to_pair(4, 1), (0, 1));
        assert_eq!(idx_to_pair(4, 2), (0, 2));
        assert_eq!(idx_to_pair(4, 3), (0, 3));
        assert_eq!(idx_to_pair(4, 4), (1, 0));
        assert_eq!(idx_to_pair(4, 5), (1, 1));
        assert_eq!(idx_to_pair(4, 6), (1, 2));
        assert_eq!(idx_to_pair(4, 7), (2, 0));
        assert_eq!(idx_to_pair(4, 8), (2, 1));
        assert_eq!(idx_to_pair(4, 9), (3, 0));
    }

    #[test]
    fn test_pair_to_idx() {
        assert_eq!(pair_to_idx(4, (0, 0)), 0);
        assert_eq!(pair_to_idx(4, (0, 1)), 1);
        assert_eq!(pair_to_idx(4, (0, 2)), 2);
        assert_eq!(pair_to_idx(4, (0, 3)), 3);
        assert_eq!(pair_to_idx(4, (1, 0)), 4);
        assert_eq!(pair_to_idx(4, (1, 1)), 5);
        assert_eq!(pair_to_idx(4, (1, 2)), 6);
        assert_eq!(pair_to_idx(4, (2, 0)), 7);
        assert_eq!(pair_to_idx(4, (2, 1)), 8);
        assert_eq!(pair_to_idx(4, (3, 0)), 9);
    }

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
