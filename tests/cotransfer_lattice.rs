use fcars::FormalConcept;
use hccr::cotransfer_lattice::CotransferSystem;
use hccr::lattice::Lattice;
use hccr::poset::{Edge, EdgeSet};
use std::collections::HashSet;
use std::error::Error;

fn representative_lattices() -> Vec<Lattice> {
    vec![
        Lattice::chain(0),
        Lattice::chain(3),
        Lattice::boolean(2),
        Lattice::from_covers(
            ["0", "a", "b", "c", "1"],
            [("0", "a"), ("a", "c"), ("c", "1"), ("0", "b"), ("b", "1")],
        )
        .unwrap(),
    ]
}

fn assert_cotransfer_axioms(system: &CotransferSystem) {
    let lattice = system.lattice();
    let arrows = system.relations();

    for &first in &arrows {
        for &second in &arrows {
            if first.to == second.from {
                assert!(
                    arrows.contains(&Edge::new(first.from, second.to)),
                    "cotransfer system was not transitive"
                );
            }
        }
    }

    for &edge in &arrows {
        for z in lattice.ids() {
            if lattice.leq(edge.from, z) {
                assert!(
                    arrows.contains(&Edge::new(z, lattice.join(edge.to, z))),
                    "cotransfer system was not closed under pushout"
                );
            }
        }
    }
}

#[test]
fn cotransfer_generation_enumeration_and_opposite_conversion_obey_their_laws()
-> Result<(), Box<dyn Error>> {
    for lattice in representative_lattices() {
        let containment = lattice.cotransfer_systems();
        let opposite_count = lattice.opposite().transfer_system_count();

        assert_eq!(containment.size(), opposite_count);
        assert_eq!(lattice.cotransfer_system_count(), opposite_count);
        for system in containment.systems() {
            assert_cotransfer_axioms(system);

            let opposite = system.opposite_transfer_system();
            assert_eq!(*opposite.lattice(), lattice.opposite());
            for edge in system.relations() {
                assert!(opposite.contains_relation(edge.reversed()));
            }
            let round_trip = lattice.cotransfer_system_from_opposite(&opposite)?;
            assert_eq!(&round_trip, system);
        }
    }
    Ok(())
}

#[test]
fn lifting_classes_give_an_order_reversing_bijection() {
    for lattice in representative_lattices() {
        let transfers = lattice.transfer_systems();

        for right in transfers.systems() {
            let left = right.left_lifting_cotransfer();
            assert_eq!(left.right_lifting_transfer(), *right);
        }

        for left in lattice.cotransfer_systems().systems() {
            let right = left.right_lifting_transfer();
            assert_eq!(right.left_lifting_cotransfer(), *left);
        }

        for lower in transfers.systems() {
            for upper in transfers.systems() {
                if lower <= upper {
                    assert!(upper.left_lifting_cotransfer() <= lower.left_lifting_cotransfer());
                }
            }
        }
    }
}

#[test]
fn generated_cotransfer_system_is_the_least_pushout_closed_system() {
    let lattice = Lattice::chain(2);
    let generated = lattice
        .cotransfer_system_generated_by([Edge::new(0, 2)])
        .unwrap();
    assert_eq!(
        generated.proper_relations(),
        [Edge::new(0, 2), Edge::new(1, 2)].into_iter().collect()
    );
}

#[test]
fn opposite_lattice_interchanges_meets_and_joins() {
    for lattice in representative_lattices() {
        let opposite = lattice.opposite();
        assert_eq!(opposite.bottom(), lattice.top());
        for left in lattice.ids() {
            for right in lattice.ids() {
                assert_eq!(opposite.meet(left, right), lattice.join(left, right));
                assert_eq!(opposite.join(left, right), lattice.meet(left, right));
                assert_eq!(opposite.leq(left, right), lattice.leq(right, left));
            }
        }
        assert_eq!(opposite.opposite(), lattice);
    }
}

/// A minimal generating set generates the system, and no proper subset
/// does; its size is the generator complexity.
#[test]
fn cotransfer_minimal_generating_sets_generate_irredundantly() -> Result<(), Box<dyn Error>> {
    for lattice in representative_lattices() {
        let cotransfer = lattice.cotransfer_systems();
        for system in cotransfer.systems() {
            let basis = system.minimal_generating_set();
            assert!(system.is_generated_by(basis.iter().copied())?);
            assert!(system.is_minimal_generating_set(basis.iter().copied())?);
            assert_eq!(basis.len(), system.generator_complexity());
            assert_eq!(system.minimum_generating_set(), basis);
        }
        assert_eq!(
            &lattice.trivial_cotransfer_system(),
            cotransfer.system(cotransfer.bottom())
        );
        assert_eq!(
            &lattice.complete_cotransfer_system(),
            cotransfer.system(cotransfer.top())
        );
    }
    Ok(())
}

fn context_test_lattices() -> Vec<Lattice> {
    let mut lattices = (1..=4).map(Lattice::chain).collect::<Vec<_>>();
    lattices.push(Lattice::boolean(2));
    lattices.push(Lattice::boolean(3));
    // The pentagon N5 and the diamond M3.
    lattices.push(
        Lattice::from_covers([0, 1, 2, 3, 4], [(0, 1), (1, 3), (3, 4), (0, 2), (2, 4)]).unwrap(),
    );
    lattices.push(
        Lattice::from_covers(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
        )
        .unwrap(),
    );
    lattices
}

fn concept_pairs(
    concepts: &[FormalConcept<Edge, Edge>],
    swap: bool,
) -> HashSet<(EdgeSet, EdgeSet)> {
    concepts
        .iter()
        .map(|concept| {
            let extent = concept.extent_names_iter().copied().collect::<EdgeSet>();
            let intent = concept.intent_names_iter().copied().collect::<EdgeSet>();
            if swap {
                (intent, extent)
            } else {
                (extent, intent)
            }
        })
        .collect()
}

#[test]
fn cotransfer_context_closed_extents_are_exactly_the_cotransfer_systems() {
    for lattice in context_test_lattices() {
        let context = lattice.cotransfer_context();
        let proper = lattice.proper_relations();
        assert_eq!(context.objects.iter().copied().collect::<EdgeSet>(), proper);
        assert_eq!(
            context.attributes.iter().copied().collect::<EdgeSet>(),
            proper
        );

        let concepts = context.all_concepts();
        let extents = concepts
            .iter()
            .map(|concept| concept.extent_names_iter().copied().collect::<EdgeSet>())
            .collect::<HashSet<_>>();
        assert_eq!(extents.len(), concepts.len());
        let all_systems = lattice.cotransfer_systems();
        let systems = all_systems
            .systems()
            .iter()
            .map(|system| system.proper_relations())
            .collect::<HashSet<_>>();
        assert_eq!(extents, systems);

        // Each extent is the left lifting class of the intent, and the intent
        // the right lifting class of the extent.
        for (extent, intent) in concept_pairs(&concepts, false) {
            let proper_part = |set: EdgeSet| {
                set.into_iter()
                    .filter(|edge| !edge.is_identity())
                    .collect::<EdgeSet>()
            };
            assert_eq!(extent, proper_part(lattice.left_lifting_class(&intent)));
            assert_eq!(intent, proper_part(lattice.right_lifting_class(&extent)));
        }

        // Exchanging objects and attributes turns the transfer context into
        // the cotransfer context, so the concepts correspond with extent and
        // intent swapped.
        let transfer_concepts = lattice.transfer_context().all_concepts();
        assert_eq!(
            concept_pairs(&concepts, false),
            concept_pairs(&transfer_concepts, true)
        );

        let concept_lattice = Lattice::from_formal_context(&context).unwrap();
        assert_eq!(concept_lattice.size(), all_systems.size());
    }
}
