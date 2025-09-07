use crate::fca::FormalConcept;
use crate::fca::FormalContext;
use rayon::iter::walk_tree;
use rayon::prelude::*;

impl<A: Clone + Send + Sync, B: Clone + Send + Sync> FormalConcept<A, B> {
    fn cbo_children(&self, y: usize) -> Vec<(Self, usize)> {
        let mut result = vec![];
        for j in self.intent.iter_zeros().filter(|&j| j >= y) {
            let c = self.extent.clone() & self.context.get_attribute_extent(j);
            let d = self.context.induce_r(&c);
            if self.intent[0..j] == d[0..j] {
                result.push((
                    Self {
                        context: self.context.clone(),
                        extent: c,
                        intent: d,
                    },
                    j + 1,
                ));
            }
        }
        result
    }
}

impl<A: Clone + Send + Sync, B: Clone + Send + Sync> FormalContext<A, B> {
    pub fn all_concepts_par_iter(&self) -> impl ParallelIterator<Item = FormalConcept<A, B>> {
        walk_tree((self.max_concept(), 0), |(x, y)| x.cbo_children(*y)).map(|(c, _)| c)
    }
    pub fn all_concepts(&self) -> Vec<FormalConcept<A, B>> {
        self.all_concepts_par_iter().collect()
    }
    pub fn num_concepts(&self) -> usize {
        self.all_concepts_par_iter().count()
    }
}
