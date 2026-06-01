use crate::poset::*;
use fcars::{FormalContext, RawFormalConcept};

pub type TransferSystem = RawFormalConcept;

pub type TransferContext = FormalContext<Edge, Edge>;

pub struct TransferPoset<A> {
    pub underlying_poset: Poset<A>,
    pub transfer_context: TransferContext,
    pub transfer_poset: Poset<TransferSystem>,
}

impl<A: Send + Sync> Poset<A> {
    pub fn transfer_poset_composition_closed(&self) -> Poset<TransferSystem> {
        let context = self.transfer_context();
        let concepts = context.all_concepts_raw();
        Poset::from_vec_by(concepts, |t1, t2| {
            if !(t1 <= t2) {
                return false;
            }
            let mut r1: EdgeSet = t1.extent.iter_ones().map(|i| context.objects[i]).collect();
            for n in 0..self.elements.len() {
                r1.insert((n, n));
            }
            let r2: EdgeSet = t2.extent.iter_ones().map(|i| context.objects[i]).collect();
            // No need to add identity edges since we only need the left-lifting class of r2
            let l2 = self.llc(r2);
            let comp = self.compose(r1, l2.collect());
            self.composition_closed(comp)
        })
    }
    pub fn transfer_context(&self) -> TransferContext {
        let irr: Vec<_> = self.proper_relations_iter().collect();
        let matrix = irr
            .iter()
            .map(|&(a, b)| {
                irr.iter()
                    .map(|&(c, d)| self.leq(d, a) || !self.leq(d, b) || !self.leq(c, a))
                    .collect()
            })
            .collect();
        FormalContext::new(irr.clone(), irr, matrix)
    }
    pub fn transfer_poset(&self) -> Poset<TransferSystem> {
        Poset::from_context(self.transfer_context())
    }
}
