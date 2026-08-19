use hccr::lattice::Lattice;
use hccr::model_structure::{ModelStructure, ModelStructureError};
use hccr::poset::{Edge, EdgeSet, compose};
use hccr::transfer_lattice::{TransferSystem, TransferUniverse};
use std::sync::Arc;

fn chain_universe(top: usize) -> Arc<TransferUniverse<usize>> {
    Arc::new(Lattice::chain(top).expect("a finite chain is a lattice")).transfer_universe()
}

fn bottom_and_top<A>(
    universe: &Arc<TransferUniverse<A>>,
) -> (TransferSystem<A>, TransferSystem<A>) {
    let bottom = universe
        .generated_by(std::iter::empty::<Edge>())
        .expect("the empty set generates a transfer system");
    let top = universe
        .generated_by(universe.proper_edges().iter().copied())
        .expect("all ambient relations generate a transfer system");
    (bottom, top)
}

fn ambient_edges<A>(universe: &TransferUniverse<A>) -> EdgeSet {
    universe.lattice().as_poset().all_relations_iter().collect()
}

fn check_model_category_laws<A>(universe: Arc<TransferUniverse<A>>) {
    let systems = universe.transfer_systems();

    for acyclic_fibrations in &systems {
        for fibrations in &systems {
            let Ok(model) = ModelStructure::new(acyclic_fibrations.clone(), fibrations.clone())
            else {
                continue;
            };

            let recovered_acyclic_fibrations = model
                .cofibrations()
                .right_lifting_transfer()
                .expect("the right lifting class is a transfer system");
            let recovered_fibrations = model
                .acyclic_cofibrations()
                .right_lifting_transfer()
                .expect("the right lifting class is a transfer system");
            assert!(recovered_acyclic_fibrations == *model.acyclic_fibrations());
            assert!(recovered_fibrations == *model.fibrations());

            let ambient = ambient_edges(&universe);
            let cofibrations = model.cofibrations().edges(true);
            let acyclic_cofibrations = model.acyclic_cofibrations().edges(true);
            let fibrations = model.fibrations().edges(true);
            let acyclic_fibrations = model.acyclic_fibrations().edges(true);

            assert_eq!(
                cofibrations,
                universe.lattice().as_poset().llc(&acyclic_fibrations)
            );
            assert_eq!(
                acyclic_cofibrations,
                universe.lattice().as_poset().llc(&fibrations)
            );
            assert_eq!(
                universe.lattice().as_poset().rlc(&cofibrations),
                acyclic_fibrations
            );
            assert_eq!(
                universe.lattice().as_poset().rlc(&acyclic_cofibrations),
                fibrations
            );
            assert!(
                universe
                    .lattice()
                    .as_poset()
                    .two_out_of_three(model.weak_equivalences())
            );

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
        check_model_category_laws(chain_universe(top));
    }
    check_model_category_laws(
        Arc::new(Lattice::boolean(2).expect("the Boolean lattice exists")).transfer_universe(),
    );
}

#[test]
fn constructor_recognizes_exactly_the_intervals_in_the_model_structure_order() {
    let universe = chain_universe(3);
    let systems = universe.transfer_systems();
    let order = universe
        .model_structure_order()
        .expect("the model-structure relation is a partial order");

    for acyclic_fibrations in &systems {
        for fibrations in &systems {
            let lower = order
                .raw_poset()
                .elements()
                .iter()
                .position(|raw| raw == acyclic_fibrations.raw())
                .expect("every transfer system occurs in the model-structure order");
            let upper = order
                .raw_poset()
                .elements()
                .iter()
                .position(|raw| raw == fibrations.raw())
                .expect("every transfer system occurs in the model-structure order");
            assert_eq!(
                ModelStructure::new(acyclic_fibrations.clone(), fibrations.clone()).is_ok(),
                order.raw_poset().leq(lower, upper),
                "classification disagrees for transfer systems {lower} and {upper}"
            );
        }
    }
}

#[test]
fn fibrant_and_cofibrant_queries_are_endpoint_predicates() {
    let universe = chain_universe(3);
    let (bottom, top) = bottom_and_top(&universe);

    let discrete = ModelStructure::new(bottom, top)
        .expect("identity weak equivalences define the discrete model structure");
    assert!(Arc::ptr_eq(
        discrete.cofibrations().universe(),
        discrete.acyclic_cofibrations().universe()
    ));

    for object in 0..universe.lattice().size() {
        assert_eq!(
            discrete.is_fibrant(object),
            discrete.contains_fibration(Edge::new(object, universe.lattice().top()))
        );
        assert_eq!(
            discrete.is_cofibrant(object),
            discrete.contains_cofibration(Edge::new(universe.lattice().bottom(), object))
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
    let universe = chain_universe(2);
    let (bottom, top) = bottom_and_top(&universe);

    let containment_error = ModelStructure::new(top, bottom)
        .expect_err("acyclic fibrations must be contained in fibrations");
    let ModelStructureError::AcyclicFibrationsNotContained { relation } = containment_error else {
        panic!("expected a failed containment witness");
    };
    assert!(relation.from < relation.to);

    let systems = universe.transfer_systems();
    let ((acyclic_fibrations, fibrations), failure) = systems
        .iter()
        .flat_map(|acyclic_fibrations| {
            systems
                .iter()
                .map(move |fibrations| (acyclic_fibrations, fibrations))
        })
        .filter(|(acyclic_fibrations, fibrations)| {
            universe.proper_edges().iter().all(|&edge| {
                !acyclic_fibrations.contains_relation(edge) || fibrations.contains_relation(edge)
            })
        })
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
        &universe.lattice().as_poset().llc(&fibrations.edges(true)),
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
fn systems_from_distinct_universes_cannot_be_mixed() {
    let first = chain_universe(2);
    let second = chain_universe(2);
    let (acyclic_fibrations, _) = bottom_and_top(&first);
    let (_, fibrations) = bottom_and_top(&second);

    assert!(matches!(
        ModelStructure::new(acyclic_fibrations, fibrations),
        Err(ModelStructureError::DifferentUniverses)
    ));
}
