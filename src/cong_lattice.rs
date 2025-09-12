use std::collections::{HashMap, HashSet};

use fcars::{FormalConcept, FormalContext};
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

impl<B: IsLattice> IsLattice for CongPoset<B> {
    fn meet(&self, a: usize, b: usize) -> usize {
        self.underlying_poset.meet(a, b)
    }
    fn join(&self, a: usize, b: usize) -> usize {
        self.underlying_poset.join(a, b)
    }
    fn restriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.underlying_poset.restriction(a, b)
    }
    fn corestriction(&self, a: usize, b: usize) -> HashSet<(usize, usize)> {
        self.underlying_poset.corestriction(a, b)
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
        Self {
            underlying_poset: poset,
            equivalence_relation,
            preceq,
        }
    }
    pub fn from_poset(
        poset: B,
        equivalence_relation: HashMap<(usize, usize), HashSet<(usize, usize)>>,
    ) -> Self {
        let n = poset.num_elements();
        let mut preceq = vec![vec![false; n]; n];
        for a in 0..n {
            for b in 0..n {
                preceq[a][b] = equivalence_relation.get(&(a, a)).map_or(false, |class| {
                    class.iter().any(|&(a_prime, _)| poset.leq(a_prime, b))
                });
            }
        }
        Self {
            underlying_poset: poset,
            equivalence_relation,
            preceq,
        }
    }
}

impl<A> Poset<A> {
    pub fn is_bottom(&self, x: usize) -> bool {
        for a in 0..self.num_elements() {
            if !self.leq(x, a) {
                return false;
            }
        }
        true
    }
    pub fn is_top(&self, x: usize) -> bool {
        for a in 0..self.num_elements() {
            if !self.leq(a, x) {
                return false;
            }
        }
        true
    }
}

impl<B: IsCongPoset + IsLattice> CongLattice<B> {
    pub fn transfer_context(&self) -> fcars::FormalContext {
        todo!()
    }
}

struct ConceptLattice<A, B> {
    concepts: Vec<FormalConcept<A, B>>,
}

impl<A, B> IsPoset for ConceptLattice<A, B> {
    fn num_elements(&self) -> usize {
        self.concepts.len()
    }
    fn leq(&self, a: usize, b: usize) -> bool {
        self.concepts[a].extent <= self.concepts[b].extent
    }
}

impl<A, B> IsLattice for ConceptLattice<A, B> {
    fn meet(&self, a: usize, b: usize) -> usize {
        self.concepts
            .iter()
            .position(|c| {
                c.extent == (self.concepts[a].extent.clone() & self.concepts[b].extent.clone())
            })
            .expect("Meet not found")
    }
    fn join(&self, a: usize, b: usize) -> usize {
        self.concepts
            .iter()
            .position(|c| {
                c.intent == (self.concepts[a].intent.clone() & self.concepts[b].intent.clone())
            })
            .expect("Join not found")
    }
}
