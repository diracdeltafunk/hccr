use hccr::lattice::Lattice;
use hccr::model_structure::{ModelStructure, ModelStructureError};
use hccr::poset::{EdgeSet, compose};
use hccr::transfer_lattice::TransferSystem;

fn bottom_and_top(lattice: &Lattice) -> (TransferSystem, TransferSystem) {
    (
        lattice.trivial_transfer_system(),
        lattice.complete_transfer_system(),
    )
}

fn ambient_edges(lattice: &Lattice) -> EdgeSet {
    lattice.all_relations_iter().collect()
}

fn check_model_category_laws(lattice: Lattice) {
    let tr = lattice.transfer_systems();
    let systems = tr.systems();

    for acyclic_fibrations in systems {
        for fibrations in systems {
            let Ok(model) = ModelStructure::new(acyclic_fibrations.clone(), fibrations.clone())
            else {
                continue;
            };

            let recovered_acyclic_fibrations = model.cofibrations().right_lifting_transfer();
            let recovered_fibrations = model.acyclic_cofibrations().right_lifting_transfer();
            assert!(recovered_acyclic_fibrations == *model.acyclic_fibrations());
            assert!(recovered_fibrations == *model.fibrations());

            let ambient = ambient_edges(&lattice);
            let cofibrations = model.cofibrations().edges(true);
            let acyclic_cofibrations = model.acyclic_cofibrations().edges(true);
            let fibrations = model.fibrations().edges(true);
            let acyclic_fibrations = model.acyclic_fibrations().edges(true);

            assert_eq!(cofibrations, lattice.llc(&acyclic_fibrations));
            assert_eq!(acyclic_cofibrations, lattice.llc(&fibrations));
            assert_eq!(lattice.rlc(&cofibrations), acyclic_fibrations);
            assert_eq!(lattice.rlc(&acyclic_cofibrations), fibrations);
            assert!(lattice.two_out_of_three(model.weak_equivalences()));

            assert_eq!(&fibrations & model.weak_equivalences(), acyclic_fibrations);
            assert_eq!(
                &cofibrations & model.weak_equivalences(),
                acyclic_cofibrations
            );

            for relation in ambient {
                let cof_afib = model
                    .factor_as_cofibration_then_acyclic_fibration(relation)
                    .expect("the first weak factorization system factors every arrow");
                assert_eq!(cof_afib.first.from, relation.from);
                assert_eq!(cof_afib.first.to, cof_afib.second.from);
                assert_eq!(cof_afib.second.to, relation.to);
                assert!(model.contains_cofibration(cof_afib.first));
                assert!(model.contains_acyclic_fibration(cof_afib.second));

                let acof_fib = model
                    .factor_as_acyclic_cofibration_then_fibration(relation)
                    .expect("the second weak factorization system factors every arrow");
                assert_eq!(acof_fib.first.from, relation.from);
                assert_eq!(acof_fib.first.to, acof_fib.second.from);
                assert_eq!(acof_fib.second.to, relation.to);
                assert!(model.contains_acyclic_cofibration(acof_fib.first));
                assert!(model.contains_fibration(acof_fib.second));
            }
        }
    }
}

#[test]
fn every_constructed_model_structure_satisfies_the_model_category_laws() {
    for top in 0..=3 {
        check_model_category_laws(Lattice::chain(top));
    }
    check_model_category_laws(Lattice::boolean(2));
}

#[test]
fn constructor_recognizes_exactly_the_intervals_in_the_model_structure_order() {
    let lattice = Lattice::chain(3);
    let tr = lattice.transfer_systems();
    let order = lattice.transfer_systems_model_structure_order();

    for acyclic_fibrations in tr.systems() {
        for fibrations in tr.systems() {
            let lower = order
                .id_of(acyclic_fibrations)
                .expect("every transfer system occurs in the model-structure order");
            let upper = order
                .id_of(fibrations)
                .expect("every transfer system occurs in the model-structure order");
            assert_eq!(
                ModelStructure::new(acyclic_fibrations.clone(), fibrations.clone()).is_ok(),
                order.leq(lower, upper),
                "classification disagrees for transfer systems {lower} and {upper}"
            );
        }
    }

    let model_structures = lattice.model_structures();
    assert_eq!(model_structures.len(), order.all_relations_iter().count());
}

#[test]
fn fibrant_and_cofibrant_queries_are_endpoint_predicates() {
    let lattice = Lattice::chain(3);
    let (bottom, top) = bottom_and_top(&lattice);

    let discrete = ModelStructure::new(bottom, top)
        .expect("identity weak equivalences define the discrete model structure");
    assert_eq!(
        discrete.cofibrations().lattice(),
        discrete.acyclic_cofibrations().lattice()
    );

    for object in lattice.ids() {
        assert_eq!(
            discrete.is_fibrant(object),
            discrete.contains_fibration((object, lattice.top()).into())
        );
        assert_eq!(
            discrete.is_cofibrant(object),
            discrete.contains_cofibration((lattice.bottom(), object).into())
        );
        assert_eq!(
            discrete.is_bifibrant(object),
            discrete.is_fibrant(object) && discrete.is_cofibrant(object)
        );
    }

    assert_eq!(discrete.fibrant_objects(), vec![0, 1, 2, 3]);
    assert_eq!(discrete.cofibrant_objects(), vec![0, 1, 2, 3]);
    assert_eq!(discrete.bifibrant_objects(), vec![0, 1, 2, 3]);
    assert!(!discrete.is_fibrant(4));
    assert!(!discrete.is_cofibrant(4));
}

#[test]
fn invalid_intervals_return_mathematical_witnesses() {
    let lattice = Lattice::chain(2);
    let (bottom, top) = bottom_and_top(&lattice);

    let containment_error = ModelStructure::new(top, bottom)
        .expect_err("acyclic fibrations must be contained in fibrations");
    let ModelStructureError::AcyclicFibrationsNotContained { relation } = containment_error else {
        panic!("expected a failed containment witness");
    };
    assert!(relation.from < relation.to);

    let tr = lattice.transfer_systems();
    let systems = tr.systems();
    let ((acyclic_fibrations, fibrations), failure) = systems
        .iter()
        .flat_map(|acyclic_fibrations| {
            systems
                .iter()
                .map(move |fibrations| (acyclic_fibrations, fibrations))
        })
        .filter(|(acyclic_fibrations, fibrations)| acyclic_fibrations <= fibrations)
        .find_map(|(acyclic_fibrations, fibrations)| {
            match ModelStructure::new(acyclic_fibrations.clone(), fibrations.clone()) {
                Err(ModelStructureError::WeakEquivalencesFailTwoOutOfThree { witness }) => {
                    Some(((acyclic_fibrations, fibrations), witness))
                }
                _ => None,
            }
        })
        .expect("the three-element chain has a premodel structure failing 2-out-of-3");

    let proposed_weak_equivalences = compose(
        &acyclic_fibrations.edges(true),
        &lattice.llc(&fibrations.edges(true)),
    );
    let membership = [failure.first, failure.second, failure.composite]
        .map(|edge| proposed_weak_equivalences.contains(&edge));
    assert_eq!(membership.into_iter().filter(|&present| present).count(), 2);
    assert!(!proposed_weak_equivalences.contains(&failure.missing));
    assert_eq!(failure.first.to, failure.second.from);
    assert_eq!(failure.composite.from, failure.first.from);
    assert_eq!(failure.composite.to, failure.second.to);
}

#[test]
fn systems_on_different_lattices_cannot_be_mixed() {
    let first = Lattice::chain(2);
    let relabelled = first.relabelled(|_, label| ("x", label)).unwrap();
    let (acyclic_fibrations, _) = bottom_and_top(&first);
    let (_, fibrations) = bottom_and_top(&relabelled);

    assert!(matches!(
        ModelStructure::new(acyclic_fibrations.clone(), fibrations),
        Err(ModelStructureError::DifferentLattices)
    ));

    // Independently constructed but equal lattices are interchangeable.
    let (_, same_fibrations) = bottom_and_top(&Lattice::chain(2));
    assert!(ModelStructure::new(acyclic_fibrations, same_fibrations).is_ok());
}
