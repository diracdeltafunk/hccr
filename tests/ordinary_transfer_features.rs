use hccr::lattice::Lattice;
use hccr::poset::Edge;
use hccr::transfer_lattice::CompatibilityFailure;

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
    let proper_edges = lattice.proper_relations_iter().collect::<Vec<_>>();

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
        .transfer_system_generated_by(lattice.proper_relations_iter())
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

#[test]
fn transfer_systems_on_equal_lattices_are_interchangeable() {
    let first = Lattice::boolean(2);
    let second = Lattice::boolean(2);
    let left = first.transfer_systems();
    for system in second.transfer_systems().systems() {
        assert!(left.id_of(system).is_some());
    }
}
