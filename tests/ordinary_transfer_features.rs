use hccr::lattice::Lattice;
use hccr::poset::Edge;
use hccr::transfer_lattice::CompatibilityFailure;
use std::sync::Arc;

fn is_raw_subset(left: &bitvec::vec::BitVec, right: &bitvec::vec::BitVec) -> bool {
    left.iter_ones().all(|bit| right[bit])
}

#[test]
fn saturated_closure_is_the_least_saturated_extension() {
    let universe = Arc::new(Lattice::boolean(2).unwrap()).transfer_universe();
    let systems = universe.transfer_systems();

    for system in &systems {
        let closure = system.saturated_closure();
        assert!(closure.is_saturated());
        assert!(is_raw_subset(system.raw().arrows(), closure.raw().arrows()));
        assert_eq!(closure.saturated_closure(), closure);

        for saturated_extension in systems.iter().filter(|candidate| {
            candidate.is_saturated()
                && is_raw_subset(system.raw().arrows(), candidate.raw().arrows())
        }) {
            assert!(is_raw_subset(
                closure.raw().arrows(),
                saturated_extension.raw().arrows()
            ));
        }
    }
}

#[test]
fn cosaturated_coclosure_is_the_greatest_cosaturated_subsystem() {
    let universe = Arc::new(Lattice::boolean(2).unwrap()).transfer_universe();
    let systems = universe.transfer_systems();

    for system in &systems {
        let coclosure = system.cosaturated_coclosure();
        assert!(coclosure.is_cosaturated());
        assert!(coclosure.is_disklike());
        assert!(is_raw_subset(
            coclosure.raw().arrows(),
            system.raw().arrows()
        ));
        assert_eq!(coclosure.cosaturated_coclosure(), coclosure);

        for cosaturated_subsystem in systems.iter().filter(|candidate| {
            candidate.is_cosaturated()
                && is_raw_subset(candidate.raw().arrows(), system.raw().arrows())
        }) {
            assert!(is_raw_subset(
                cosaturated_subsystem.raw().arrows(),
                coclosure.raw().arrows()
            ));
        }
    }
}

#[test]
fn saturated_systems_form_the_expected_containment_lattice() {
    let universe = Arc::new(Lattice::boolean(2).unwrap()).transfer_universe();
    let saturated = universe.saturated_transfer_systems();
    let lattice = universe.saturated_containment_lattice().unwrap();

    // The Boolean square is the subgroup lattice of C_pq, which has seven
    // saturated transfer systems.
    assert_eq!(saturated.len(), 7);
    assert_eq!(lattice.size(), saturated.len());
    assert!(lattice.systems().all(|system| system.is_saturated()));

    for left in 0..lattice.size() {
        for right in 0..lattice.size() {
            assert!(
                lattice
                    .system(lattice.meet_id(left, right))
                    .unwrap()
                    .is_saturated()
            );
            assert!(
                lattice
                    .system(lattice.join_id(left, right))
                    .unwrap()
                    .is_saturated()
            );
        }
    }
}

#[test]
fn generated_bases_are_irredundant_and_minimum() {
    let universe = Arc::new(Lattice::boolean(2).unwrap()).transfer_universe();
    let proper_edges = universe.proper_edges();

    for system in universe.transfer_systems() {
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
    let lattice = Arc::new(Lattice::boolean(2).unwrap());
    let universe = Arc::clone(&lattice).transfer_universe();
    let systems = universe.transfer_systems();
    let defining_complexity = systems
        .iter()
        .map(|system| system.generator_complexity())
        .max()
        .unwrap();
    let complete = universe
        .generated_by(universe.proper_edges().iter().copied())
        .unwrap();

    assert_eq!(universe.complexity(), defining_complexity);
    assert_eq!(universe.width(), complete.generator_complexity());
    assert_eq!(
        Arc::clone(&lattice).transfer_system_complexity(),
        defining_complexity
    );
    assert_eq!(lattice.transfer_system_width(), universe.width());
}

#[test]
fn compatibility_satisfies_the_standard_general_laws() {
    let lattice = Arc::new(Lattice::boolean(2).unwrap());
    let universe = Arc::clone(&lattice).transfer_universe();
    let systems = universe.transfer_systems();
    let complete = universe
        .generated_by(universe.proper_edges().iter().copied())
        .unwrap();
    let trivial = universe.generated_by(std::iter::empty::<Edge>()).unwrap();

    for system in &systems {
        assert!(complete.is_compatible_with(system));
        assert!(system.is_compatible_with(&trivial));
        assert_eq!(system.is_compatible_with(system), system.is_saturated());
    }

    let enumerated = universe.compatible_pairs();
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

    let other_trivial = lattice
        .transfer_universe()
        .generated_by(std::iter::empty::<Edge>())
        .unwrap();
    assert_eq!(
        trivial.compatibility_failure(&other_trivial),
        Some(CompatibilityFailure::DifferentUniverses)
    );
}
