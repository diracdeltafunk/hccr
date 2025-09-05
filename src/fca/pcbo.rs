use crate::fca::FormalConcept;
use crate::fca::FormalContext;
use bitvec::prelude::*;
use std::sync::Arc;
const RECURSION_LEVEL: usize = 10;

fn parallel_generate_from(concept: &FormalConcept, y: usize, l: usize) {
    if l == RECURSION_LEVEL {
        // put (concept, y) in queue
        return;
    }
    // assert!(y < RECURSION_LEVEL);
    // YIELD concept
    let num_objs = concept.context.objects.len();
    let num_attrs = concept.context.attributes.len();
    if y >= num_attrs || concept.intent.all() {
        return;
    }
    // assert!(!concept.intent[y]);
    for j in y..num_attrs {
        if !concept.intent[j] {
            let mut one_hot = BitVec::repeat(false, num_objs);
            one_hot.set(j, true);
            let c = concept.extent.clone() & concept.context.induce_l(&one_hot);
            let d = concept.context.induce_r(&c);
            if concept.intent[0..j] == d[0..j] {
                // YIELD parallel_generate_from(
                //     &FormalConcept {
                //         context: concept.context.clone(),
                //         extent: c,
                //         intent: d,
                //     },
                //     j + 1,
                //     l + 1
                // );
            }
        }
    }
    if l == 1 {
        // Do normal generate_from
    }
}

pub fn all_concepts(context: FormalContext) -> Vec<FormalConcept> {
    let init_concept = FormalConcept {
        extent: BitVec::repeat(true, context.objects.len()),
        intent: context.induce_r(&BitVec::repeat(true, context.objects.len())),
        context: Arc::new(context),
    };
    parallel_generate_from(&init_concept, 0, 1);
    todo!()
}
