use hccr::lattice::Lattice;
use hccr::poset::{Edge, EdgeSet};
use hccr::transfer_lattice::CompatibilityFailure;
use std::collections::HashSet;

#[test]
fn saturated_closure_is_the_least_saturated_extension() {
    let tr = Lattice::boolean(2).transfer_systems();
    let systems = tr.systems();

    for system in systems {
        let closure = system.saturated_closure();
        assert!(closure.is_saturated());
        assert!(system <= &closure);
        assert_eq!(closure.saturated_closure(), closure);

        for saturated_extension in systems
            .iter()
            .filter(|candidate| candidate.is_saturated() && system <= candidate)
        {
            assert!(&closure <= saturated_extension);
        }
    }
}

#[test]
fn cosaturated_coclosure_is_the_greatest_cosaturated_subsystem() {
    let tr = Lattice::boolean(2).transfer_systems();
    let systems = tr.systems();

    for system in systems {
        let coclosure = system.cosaturated_coclosure();
        assert!(coclosure.is_cosaturated());
        assert!(coclosure.is_disklike());
        assert!(&coclosure <= system);
        assert_eq!(coclosure.cosaturated_coclosure(), coclosure);

        for cosaturated_subsystem in systems
            .iter()
            .filter(|candidate| candidate.is_cosaturated() && candidate <= &system)
        {
            assert!(cosaturated_subsystem <= &coclosure);
        }
    }
}

#[test]
fn saturated_systems_form_the_expected_containment_lattice() {
    let lattice = Lattice::boolean(2).saturated_transfer_systems();

    // The Boolean square is the subgroup lattice of C_pq, which has seven
    // saturated transfer systems.
    assert_eq!(lattice.size(), 7);
    assert!(lattice.systems().iter().all(|system| system.is_saturated()));

    for left in lattice.ids() {
        for right in lattice.ids() {
            assert!(lattice.system(lattice.meet(left, right)).is_saturated());
            assert!(lattice.system(lattice.join(left, right)).is_saturated());
        }
    }
}

#[test]
fn generated_bases_are_irredundant_and_minimum() {
    let lattice = Lattice::boolean(2);
    let proper_edges = lattice.proper_relations().into_iter().collect::<Vec<_>>();

    for system in lattice.transfer_systems().systems() {
        let basis = system.minimal_generating_set();
        assert_eq!(basis, system.minimum_generating_set());
        assert_eq!(basis.len(), system.generator_complexity());
        assert!(system.is_generated_by(basis.iter().copied()).unwrap());
        assert!(
            system
                .is_minimal_generating_set(basis.iter().copied())
                .unwrap()
        );

        let mut minimum_size = usize::MAX;
        for mask in 0usize..(1usize << proper_edges.len()) {
            let candidate = proper_edges
                .iter()
                .enumerate()
                .filter_map(|(bit, &edge)| ((mask >> bit) & 1 == 1).then_some(edge))
                .collect::<Vec<_>>();
            if system.is_generated_by(candidate).unwrap() {
                minimum_size = minimum_size.min(mask.count_ones() as usize);
            }
        }
        assert_eq!(system.generator_complexity(), minimum_size);
    }
}

#[test]
fn lattice_complexity_and_width_have_their_defining_values() {
    let lattice = Lattice::boolean(2);
    let defining_complexity = lattice
        .transfer_systems()
        .systems()
        .iter()
        .map(|system| system.generator_complexity())
        .max()
        .unwrap();
    let complete = lattice
        .transfer_system_generated_by(lattice.proper_relations())
        .unwrap();

    assert_eq!(complete, lattice.complete_transfer_system());
    assert_eq!(lattice.transfer_system_complexity(), defining_complexity);
    assert_eq!(
        lattice.transfer_system_width(),
        complete.generator_complexity()
    );
}

#[test]
fn compatibility_satisfies_the_standard_general_laws() {
    let lattice = Lattice::boolean(2);
    let tr = lattice.transfer_systems();
    let systems = tr.systems();
    let complete = lattice.complete_transfer_system();
    let trivial = lattice
        .transfer_system_generated_by(std::iter::empty::<Edge>())
        .unwrap();
    assert_eq!(trivial, lattice.trivial_transfer_system());

    for system in systems {
        assert!(complete.is_compatible_with(system));
        assert!(system.is_compatible_with(&trivial));
        assert_eq!(system.is_compatible_with(system), system.is_saturated());
    }

    let enumerated = lattice.compatible_transfer_system_pairs();
    let defining_count = systems
        .iter()
        .flat_map(|additive| {
            systems
                .iter()
                .filter(move |multiplicative| additive.is_compatible_with(multiplicative))
        })
        .count();
    assert_eq!(enumerated.len(), defining_count);

    assert!(matches!(
        trivial.compatibility_failure(&complete),
        Some(CompatibilityFailure::MultiplicativeNotAdditive { .. })
    ));
    assert!(
        systems
            .iter()
            .any(|additive| systems.iter().any(|multiplicative| {
                matches!(
                    additive.compatibility_failure(multiplicative),
                    Some(CompatibilityFailure::Distributivity { .. })
                )
            }))
    );

    let other_trivial = Lattice::chain(3).trivial_transfer_system();
    assert_eq!(
        trivial.compatibility_failure(&other_trivial),
        Some(CompatibilityFailure::DifferentLattices)
    );
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

#[test]
fn transfer_context_closed_extents_are_exactly_the_transfer_systems() {
    for lattice in context_test_lattices() {
        let context = lattice.transfer_context();
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
        let all_systems = lattice.transfer_systems();
        let systems = all_systems
            .systems()
            .iter()
            .map(|system| system.proper_relations())
            .collect::<HashSet<_>>();
        assert_eq!(extents, systems);

        // The intent of each concept is the left lifting class of its extent.
        for concept in &concepts {
            let extent = concept.extent_names_iter().copied().collect::<EdgeSet>();
            let intent = concept.intent_names_iter().copied().collect::<EdgeSet>();
            let left_lifting = lattice
                .left_lifting_class(&extent)
                .into_iter()
                .filter(|edge| !edge.is_identity())
                .collect::<EdgeSet>();
            assert_eq!(intent, left_lifting);
        }

        let concept_lattice = Lattice::from_formal_context(&context).unwrap();
        assert_eq!(concept_lattice.size(), all_systems.size());
    }
}

/// Sizes of all inclusion-minimal generating sets of `system`, found by
/// brute force over subsets of its proper relations, using generation as
/// the only primitive.
fn minimal_generating_set_sizes(
    lattice: &Lattice,
    system: &hccr::transfer_lattice::TransferSystem,
) -> HashSet<usize> {
    let relations = system.proper_relations().into_iter().collect::<Vec<_>>();
    let count = relations.len();
    let generates = (0u32..1 << count)
        .map(|mask| {
            let subset = (0..count)
                .filter(|&i| mask >> i & 1 == 1)
                .map(|i| relations[i]);
            lattice.transfer_system_generated_by(subset).unwrap() == *system
        })
        .collect::<Vec<_>>();
    (0u32..1 << count)
        .filter(|&mask| {
            generates[mask as usize]
                && (0..count)
                    .filter(|&i| mask >> i & 1 == 1)
                    .all(|i| !generates[(mask & !(1 << i)) as usize])
        })
        .map(|mask| mask.count_ones() as usize)
        .collect()
}

/// All inclusion-minimal generating sets of a transfer system have the same
/// size, the generator complexity (Adamyk et al., arXiv:2506.11159,
/// Corollary 2.14, whose proof applies to any finite lattice). The width is
/// the number of meet-irreducible elements (Proposition 4.5 there), and the
/// complexities of `Sub(C_{p^n q}) = [n] x [1]` are `3k + 1` for `n = 2k`
/// and `3k + 2` for `n = 2k + 1` (Corollary 6.6 there).
#[test]
fn minimal_generating_sets_have_a_common_size_counted_by_known_invariants() {
    let pentagon =
        Lattice::from_covers([0, 1, 2, 3, 4], [(0, 1), (1, 3), (3, 4), (0, 2), (2, 4)]).unwrap();
    let m3 = Lattice::horizontal_join([Lattice::chain(2), Lattice::chain(2), Lattice::chain(2)])
        .unwrap();
    let hexagon = Lattice::horizontal_join([Lattice::chain(3), Lattice::chain(3)]).unwrap();
    let lattices = [
        Lattice::chain(3),
        Lattice::boolean(2),
        Lattice::product([Lattice::chain(2), Lattice::chain(1)]),
        pentagon.clone(),
        pentagon.opposite(),
        m3,
        hexagon,
    ];
    for lattice in &lattices {
        for system in lattice.transfer_systems().systems() {
            let sizes = minimal_generating_set_sizes(lattice, system);
            assert_eq!(
                sizes,
                HashSet::from([system.generator_complexity()]),
                "{system}"
            );
            assert_eq!(
                system.minimum_generating_set().len(),
                system.generator_complexity()
            );
        }
        assert_eq!(
            lattice.transfer_system_width(),
            lattice.meet_irreducibles().len(),
            "{lattice}"
        );
    }

    for lattice in [
        Lattice::boolean(3),
        Lattice::product([pentagon.clone(), Lattice::chain(1)]),
    ] {
        assert_eq!(
            lattice.transfer_system_width(),
            lattice.meet_irreducibles().len(),
            "{lattice}"
        );
    }
    for (n, complexity) in [(1, 2), (2, 4), (3, 5)] {
        let grid = Lattice::product([Lattice::chain(n), Lattice::chain(1)]);
        assert_eq!(grid.transfer_system_complexity(), complexity);
        assert_eq!(grid.transfer_system_width(), n + 1);
    }
}
