use std::collections::{HashMap, HashSet};
// use std::fmt::Debug;
// use bitvec::prelude::*;

pub trait IsPoset {
    fn num_elements(&self) -> usize;
    fn leq(&self, a: usize, b: usize) -> bool;
    fn upset(&self, a: usize) -> HashSet<usize> {
        let mut upset = HashSet::new();
        for i in 0..self.num_elements() {
            if self.leq(a, i) {
                upset.insert(i);
            }
        }
        upset
    }
    fn downset(&self, a: usize) -> HashSet<usize> {
        let mut downset = HashSet::new();
        for i in 0..self.num_elements() {
            if self.leq(i, a) {
                downset.insert(i);
            }
        }
        downset
    }
    fn maybe_meet(&self, a: usize, b: usize) -> Option<usize> {
        (0..self.num_elements())
            .map(|i| self.downset(i))
            .position(|s| {
                s == self
                    .downset(a)
                    .intersection(&self.downset(b))
                    .copied()
                    .collect()
            })
    }
    fn maybe_join(&self, a: usize, b: usize) -> Option<usize> {
        (0..self.num_elements())
            .map(|i| self.upset(i))
            .position(|s| {
                s == self
                    .upset(a)
                    .intersection(&self.upset(b))
                    .copied()
                    .collect()
            })
    }
}

pub trait IsCongPoset: IsPoset {
    fn preceq(&self, a: usize, b: usize) -> bool;
    fn simeq(&self, a: usize, b: usize) -> bool {
        self.preceq(a, b) && self.preceq(b, a)
    }
    fn equivalence_class_element(&self, a: usize) -> HashSet<usize> {
        let mut class = HashSet::new();
        for i in 0..self.num_elements() {
            if self.simeq(a, i) {
                class.insert(i);
            }
        }
        class
    }
}

pub trait IsLattice: IsPoset {
    fn meet(&self, a: usize, b: usize) -> usize {
        self.maybe_meet(a, b).expect("Meet not found")
    }
    fn join(&self, a: usize, b: usize) -> usize {
        self.maybe_join(a, b).expect("Join not found")
    }
    fn restriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        let mut restriction = HashSet::new();
        for x in self.downset(b) {
            restriction.insert((self.meet(a, x), x));
        }
        restriction
    }
    fn corestriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        let mut corestriction = HashSet::new();
        for x in self.upset(a) {
            corestriction.insert((x, self.join(b, x)));
        }
        corestriction
    }
}

pub trait IsCongLattice: IsLattice + IsCongPoset {}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Poset<A = String> {
    pub elements: Vec<A>,
    partial_order: Vec<Vec<bool>>,
    partial_order_transposed: Vec<Vec<bool>>, // Must be the transpose of partial_order
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CongPoset<B: IsPoset> {
    pub underlying_poset: B,
    equivalence_relation: HashMap<(usize, usize), HashSet<(usize, usize)>>, // An equivalence relation on edges
    preceq: Vec<Vec<bool>>, // a \preceq b iff (a,a) there is some a' such that (a,a) ~ (a',a') and a' \leq b
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Lattice<B: IsPoset> {
    pub underlying_poset: B,
    // We precompute all meets, joins, restrictions, and corestrictions
    meets: HashMap<(usize, usize), usize>,
    joins: HashMap<(usize, usize), usize>,
    restrictions: HashMap<(usize, usize), HashSet<(usize, usize)>>,
    corestrictions: HashMap<(usize, usize), HashSet<(usize, usize)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CongLattice<B: IsCongPoset + IsLattice>(B); // Access the congposet via .0

impl<A> IsPoset for Poset<A> {
    fn num_elements(&self) -> usize {
        self.elements.len()
    }
    fn leq(&self, a: usize, b: usize) -> bool {
        self.partial_order[a][b]
    }
    fn upset(&self, a: usize) -> HashSet<usize> {
        self.partial_order[a]
            .iter()
            .enumerate()
            .filter_map(|(i, &leq)| if leq { Some(i) } else { None })
            .collect()
    }
    fn downset(&self, a: usize) -> HashSet<usize> {
        self.partial_order_transposed[a]
            .iter()
            .enumerate()
            .filter_map(|(i, &leq)| if leq { Some(i) } else { None })
            .collect()
    }
}

impl<B: IsPoset> IsPoset for CongPoset<B> {
    fn num_elements(&self) -> usize {
        self.underlying_poset.num_elements()
    }
    fn leq(&self, a: usize, b: usize) -> bool {
        self.underlying_poset.leq(a, b)
    }
    fn upset(&self, a: usize) -> HashSet<usize> {
        self.underlying_poset.upset(a)
    }
    fn downset(&self, a: usize) -> HashSet<usize> {
        self.underlying_poset.downset(a)
    }
}

impl<B: IsPoset> IsPoset for Lattice<B> {
    fn num_elements(&self) -> usize {
        self.underlying_poset.num_elements()
    }
    fn leq(&self, a: usize, b: usize) -> bool {
        self.underlying_poset.leq(a, b)
    }
    fn upset(&self, a: usize) -> HashSet<usize> {
        self.underlying_poset.upset(a)
    }
    fn downset(&self, a: usize) -> HashSet<usize> {
        self.underlying_poset.downset(a)
    }
}

impl<B: IsCongPoset + IsLattice> IsPoset for CongLattice<B> {
    fn num_elements(&self) -> usize {
        self.0.num_elements()
    }
    fn leq(&self, a: usize, b: usize) -> bool {
        self.0.leq(a, b)
    }
    fn upset(&self, a: usize) -> HashSet<usize> {
        self.0.upset(a)
    }
    fn downset(&self, a: usize) -> HashSet<usize> {
        self.0.downset(a)
    }
}

impl<B: IsPoset> IsCongPoset for CongPoset<B> {
    fn preceq(&self, a: usize, b: usize) -> bool {
        self.preceq[a][b]
    }
}

impl<B: IsCongPoset + IsLattice> IsCongPoset for CongLattice<B> {
    fn preceq(&self, a: usize, b: usize) -> bool {
        self.0.preceq(a, b)
    }
    fn simeq(&self, a: usize, b: usize) -> bool {
        self.0.simeq(a, b)
    }
    fn equivalence_class_element(&self, a: usize) -> HashSet<usize> {
        self.0.equivalence_class_element(a)
    }
}

impl<B: IsPoset> IsLattice for Lattice<B> {
    fn meet(&self, a: usize, b: usize) -> usize {
        *self.meets.get(&(a, b)).expect("Meet not found")
    }
    fn join(&self, a: usize, b: usize) -> usize {
        *self.joins.get(&(a, b)).expect("Join not found")
    }
    fn restriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.restrictions
            .get(&(a, b))
            .expect("Restriction not found")
            .clone()
    }
    fn corestriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.corestrictions
            .get(&(a, b))
            .expect("Corestriction not found")
            .clone()
    }
}

impl<B: IsCongPoset + IsLattice> IsLattice for CongLattice<B> {
    fn meet(&self, a: usize, b: usize) -> usize {
        self.0.meet(a, b)
    }
    fn join(&self, a: usize, b: usize) -> usize {
        self.0.join(a, b)
    }
    fn restriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.0.restriction(a, b)
    }
    fn corestriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.0.corestriction(a, b)
    }
}

impl<A> Poset<A> {
    pub fn new(elements: Vec<A>, relation: Vec<Vec<bool>>) -> Self {
        let n = elements.len();
        assert_eq!(relation.len(), n, "Relation must be square matrix.");
        for row in &relation {
            assert_eq!(row.len(), n, "Relation must be square matrix.");
        }
        // Check reflexivity
        for i in 0..n {
            assert!(relation[i][i], "Relation must be reflexive.");
        }
        // Check antisymmetry
        for i in 0..n {
            for j in 0..n {
                if relation[i][j] && relation[j][i] {
                    assert_eq!(i, j, "Relation must be antisymmetric.");
                }
            }
        }
        // Check transitivity
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if relation[i][j] && relation[j][k] {
                        assert!(relation[i][k], "Relation must be transitive.");
                    }
                }
            }
        }
        Poset::new_unsafe(elements, relation)
    }
    fn new_unsafe(elements: Vec<A>, relation: Vec<Vec<bool>>) -> Self {
        let n = elements.len();
        let partial_order_transposed = (0..n)
            .map(|i| (0..n).map(|j| relation[j][i]).collect())
            .collect();
        Poset {
            elements,
            partial_order: relation,
            partial_order_transposed,
        }
    }
    /// Produces the poset generated by the given edges. The input edges
    /// should be given as pairs of indices into the `elements` vector.
    pub fn from_edges(elements: Vec<A>, edges: &[(usize, usize)]) -> Self {
        let n = elements.len();
        let mut relation = vec![vec![false; n]; n];
        for &(a, b) in edges {
            assert!(a < n && b < n, "Edge indices out of bounds.");
            relation[a][b] = true;
        }
        // Compute transitive closure using Floyd-Warshall
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if relation[i][k] && relation[k][j] {
                        relation[i][j] = true;
                    }
                }
            }
        }
        // Check antisymmetry
        for i in 0..n {
            for j in 0..n {
                if relation[i][j] && relation[j][i] {
                    assert_eq!(i, j, "Relation must be antisymmetric.");
                }
            }
        }
        // Add reflexivity
        for i in 0..n {
            relation[i][i] = true;
        }
        Poset::new_unsafe(elements, relation)
    }
}

impl<B: IsPoset> Lattice<B> {
    pub fn from_poset(poset: B) -> Self {
        let n = poset.num_elements();
        let mut meets = HashMap::new();
        let mut joins = HashMap::new();
        let mut restrictions = HashMap::new();
        let mut corestrictions = HashMap::new();
        for a in 0..n {
            for b in poset.upset(a) {
                restrictions.insert((a, b), HashSet::new());
                corestrictions.insert((a, b), HashSet::new());
            }
        }
        for a in 0..n {
            for b in 0..n {
                let meet = poset.maybe_meet(a, b).expect(&format!(
                    "Poset is not a lattice! No meet for {} and {}",
                    a, b
                ));
                let join = poset.maybe_join(a, b).expect(&format!(
                    "Poset is not a lattice! No join for {} and {}",
                    a, b
                ));
                meets.insert((a, b), meet);
                joins.insert((a, b), join);
                for &c in poset.upset(a).intersection(&poset.upset(b)) {
                    restrictions.get_mut(&(a, c)).unwrap().insert((meet, b));
                    restrictions.get_mut(&(b, c)).unwrap().insert((meet, a));
                }
                for &c in poset.downset(a).intersection(&poset.downset(b)) {
                    corestrictions.get_mut(&(c, b)).unwrap().insert((a, join));
                    corestrictions.get_mut(&(c, a)).unwrap().insert((b, join));
                }
            }
        }
        Lattice {
            underlying_poset: poset,
            meets,
            joins,
            restrictions,
            corestrictions,
        }
    }
}

impl<B: IsPoset> CongPoset<B> {
    pub fn discrete_from_poset(poset: B) -> Self {
        let n = poset.num_elements();
        let mut equivalence_relation = HashMap::new();
        let mut preceq = vec![vec![false; n]; n];
        for a in 0..n {
            for b in poset.upset(a) {
                preceq[a][b] = true;
                equivalence_relation.insert((a, b), HashSet::from([(a, b)]));
            }
        }
        CongPoset {
            underlying_poset: poset,
            equivalence_relation,
            preceq,
        }
    }
}

// // A CongLattice encodes the data of a finite lattice, together with an
// // equivalence relation on its elements which is a congruence relation on the
// // underlying poset in the sense of universal algebra. The data structure also
// // has some hash tables to cache (co)restriction / meet / join / etc.
// pub struct CongLattice {
//     pub num_elements: usize,
//     poset_relation: Vec<Vec<bool>>,
//     // TODO Actually, we should be storing an equivalence relation on edges...
//     //      Not sure exactly what axioms this equivalence relation needs to
//     //      satisfy. One condition must be that the induced equivalence relation
//     //      on the elements (i.e. identity edges) is a poset congruence relation
//     equivalence_relation: Vec<Vec<bool>>,
//     meet_cache: HashMap<(usize, usize), usize>,
//     join_cache: HashMap<(usize, usize), usize>,
//     restriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
//     corestriction_cache: HashMap<(usize, usize), Vec<(usize, usize)>>,
// }

// impl CongLattice {
//     // Construct a total order with trivial equivalence relation.
//     pub fn total_order(num_elements: usize) -> Self {
//         assert!(num_elements > 0, "Number of elements must be positive.");
//         let mut poset_relation = vec![vec![false; num_elements]; num_elements];
//         for i in 0..num_elements {
//             for j in i..num_elements {
//                 poset_relation[i][j] = true;
//             }
//         }
//         let mut equivalence_relation = vec![vec![false; num_elements]; num_elements];
//         for i in 0..num_elements {
//             equivalence_relation[i][i] = true;
//         }
//         CongLattice {
//             num_elements,
//             poset_relation,
//             equivalence_relation,
//             meet_cache: HashMap::new(),
//             join_cache: HashMap::new(),
//             restriction_cache: HashMap::new(),
//             corestriction_cache: HashMap::new(),
//         }
//     }
//     pub fn leq(&self, a: usize, b: usize) -> bool {
//         assert!(
//             a < self.num_elements && b < self.num_elements,
//             "Indices out of bounds."
//         );
//         self.poset_relation[a][b]
//     }
//     pub fn equiv(&self, a: usize, b: usize) -> bool {
//         assert!(
//             a < self.num_elements && b < self.num_elements,
//             "Indices out of bounds."
//         );
//         self.equivalence_relation[a][b]
//     }
//     // Check if [a] <= [b] in the quotient poset
//     pub fn preceq(&self, a: usize, b: usize) -> bool {
//         assert!(
//             a < self.num_elements && b < self.num_elements,
//             "Indices out of bounds."
//         );
//         for i in 0..self.num_elements {
//             if self.equivalence_relation[a][i] && self.poset_relation[i][b] {
//                 return true;
//             }
//         }
//         false
//     }
//     pub fn meet(&mut self, a: usize, b: usize) -> usize {
//         assert!(
//             a < self.num_elements && b < self.num_elements,
//             "Indices out of bounds."
//         );
//         if let Some(&result) = self.meet_cache.get(&(a, b)) {
//             return result;
//         }
//         // Find the greatest lower bound (meet) of a and b.
//         let mut meet: Option<usize> = None;
//         for i in 0..self.num_elements {
//             if self.poset_relation[i][a] && self.poset_relation[i][b] {
//                 if meet.is_none() || self.poset_relation[meet.unwrap()][i] {
//                     meet = Some(i);
//                 }
//             }
//         }
//         let result = meet.expect("Meet not found");
//         self.meet_cache.insert((a, b), result);
//         result
//     }
//     pub fn join(&mut self, a: usize, b: usize) -> usize {
//         assert!(
//             a < self.num_elements && b < self.num_elements,
//             "Indices out of bounds."
//         );
//         if let Some(&result) = self.join_cache.get(&(a, b)) {
//             return result;
//         }
//         // Find the least upper bound (join) of a and b.
//         let mut join: Option<usize> = None;
//         for i in 0..self.num_elements {
//             if self.poset_relation[a][i] && self.poset_relation[b][i] {
//                 if join.is_none() || self.poset_relation[i][join.unwrap()] {
//                     join = Some(i);
//                 }
//             }
//         }
//         let result = join.expect("Join not found");
//         self.join_cache.insert((a, b), result);
//         result
//     }
//     // Given an edge a -> b in the lattice, returns the list of all edges
//     // meet(x,a) -> a
//     // for x <= b
//     pub fn restriction(&mut self, a: usize, b: usize) -> Vec<(usize, usize)> {
//         assert!(
//             a <= b,
//             "First argument must be less than or equal to second."
//         );
//         assert!(b < self.num_elements, "Indices out of bounds.");
//         if let Some(result) = self.restriction_cache.get(&(a, b)) {
//             return result.clone();
//         }
//         let mut restriction = Vec::new();
//         for i in 0..self.num_elements {
//             if self.poset_relation[i][b] {
//                 restriction.push((self.meet(i, a), a));
//             }
//         }
//         self.restriction_cache.insert((a, b), restriction.clone());
//         restriction
//     }
//     // Given an edge a -> b in the lattice, returns the list of all edges
//     // b -> join(x,b)
//     // for x >= a
//     pub fn corestriction(&mut self, a: usize, b: usize) -> Vec<(usize, usize)> {
//         assert!(
//             a <= b,
//             "First argument must be less than or equal to second."
//         );
//         assert!(b < self.num_elements, "Indices out of bounds.");
//         if let Some(result) = self.corestriction_cache.get(&(a, b)) {
//             return result.clone();
//         }
//         let mut corestriction = Vec::new();
//         for i in 0..self.num_elements {
//             if self.poset_relation[a][i] {
//                 corestriction.push((b, self.join(i, b)));
//             }
//         }
//         self.corestriction_cache
//             .insert((a, b), corestriction.clone());
//         corestriction
//     }
//     // Returns the equivalence class of the input element
//     pub fn equivalence_class(&self, a: usize) -> Vec<usize> {
//         assert!(a < self.num_elements, "Index out of bounds.");
//         let mut class = Vec::new();
//         for i in 0..self.num_elements {
//             if self.equivalence_relation[a][i] {
//                 class.push(i);
//             }
//         }
//         class
//     }
//     // Takes a subset of elements and returns the union of their equivalence classes
//     pub fn equivalence_closure(&self, elements: &[usize]) -> Vec<usize> {
//         let mut closure = Vec::new();
//         for &elem in elements {
//             for i in self.equivalence_class(elem) {
//                 closure.push(i);
//             }
//         }
//         closure.sort_unstable();
//         closure.dedup();
//         closure
//     }
// }

// impl Debug for CongLattice {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "CongLattice with {} elements", self.num_elements)
//     }
// }

// // Test meet and join
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_idx_to_pair() {
//         assert_eq!(idx_to_pair(4, 0), (0, 0));
//         assert_eq!(idx_to_pair(4, 1), (0, 1));
//         assert_eq!(idx_to_pair(4, 2), (0, 2));
//         assert_eq!(idx_to_pair(4, 3), (0, 3));
//         assert_eq!(idx_to_pair(4, 4), (1, 0));
//         assert_eq!(idx_to_pair(4, 5), (1, 1));
//         assert_eq!(idx_to_pair(4, 6), (1, 2));
//         assert_eq!(idx_to_pair(4, 7), (2, 0));
//         assert_eq!(idx_to_pair(4, 8), (2, 1));
//         assert_eq!(idx_to_pair(4, 9), (3, 0));
//     }

//     #[test]
//     fn test_pair_to_idx() {
//         assert_eq!(pair_to_idx(4, (0, 0)), 0);
//         assert_eq!(pair_to_idx(4, (0, 1)), 1);
//         assert_eq!(pair_to_idx(4, (0, 2)), 2);
//         assert_eq!(pair_to_idx(4, (0, 3)), 3);
//         assert_eq!(pair_to_idx(4, (1, 0)), 4);
//         assert_eq!(pair_to_idx(4, (1, 1)), 5);
//         assert_eq!(pair_to_idx(4, (1, 2)), 6);
//         assert_eq!(pair_to_idx(4, (2, 0)), 7);
//         assert_eq!(pair_to_idx(4, (2, 1)), 8);
//         assert_eq!(pair_to_idx(4, (3, 0)), 9);
//     }

//     #[test]
//     fn test_meet() {
//         let mut lattice = CongLattice::total_order(5);
//         // Check that meet(a,b) == a iff a <= b
//         for i in 0..5 {
//             for j in i..5 {
//                 assert_eq!(lattice.meet(i, j), i);
//                 assert_eq!(lattice.meet(j, i), i);
//             }
//         }
//     }

//     #[test]
//     fn test_join() {
//         let mut lattice = CongLattice::total_order(5);
//         // Check that join(a,b) == b iff a <= b
//         for i in 0..5 {
//             for j in i..5 {
//                 assert_eq!(lattice.join(i, j), j);
//                 assert_eq!(lattice.join(j, i), j);
//             }
//         }
//     }
// }
