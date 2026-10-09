#![cfg(feature = "groups")]

use hccr::g_lattice::{GLattice, GTransferSystem};
use hccr::lattice::Lattice;
use std::error::Error;

#[test]
fn equivariant_saturation_cosaturation_and_bases_obey_their_universal_properties()
-> Result<(), Box<dyn Error>> {
    let g_lattice = swapping_diamond()?;
    let tr = g_lattice.transfer_systems();
    let systems = tr.systems();

    for system in systems {
        let underlying = system.underlying_transfer_system();
        let saturated = system.saturated_closure();
        assert_eq!(system.is_saturated(), underlying.is_saturated());
        assert!(saturated.is_saturated());
        assert!(contained(system, &saturated));
        assert_eq!(saturated.saturated_closure(), saturated);
        assert_eq!(
            saturated.relations(false),
            underlying.saturated_closure().edges(false)
        );
        for candidate in systems
            .iter()
            .filter(|candidate| candidate.is_saturated() && contained(system, candidate))
        {
            assert!(contained(&saturated, candidate));
        }

        let cosaturated = system.cosaturated_coclosure();
        assert_eq!(system.is_cosaturated(), underlying.is_cosaturated());
        assert!(cosaturated.is_cosaturated());
        assert!(cosaturated.is_disklike());
        assert!(contained(&cosaturated, system));
        assert_eq!(cosaturated.cosaturated_coclosure(), cosaturated);
        assert_eq!(
            cosaturated.relations(false),
            underlying.cosaturated_coclosure().edges(false)
        );
        for candidate in systems
            .iter()
            .filter(|candidate| candidate.is_cosaturated() && contained(candidate, system))
        {
            assert!(contained(candidate, &cosaturated));
        }

        let basis = system.minimal_generating_set();
        let generators = basis
            .iter()
            .map(|label| label.canonical_representative())
            .collect::<Vec<_>>();
        assert!(system.is_generated_by(generators.iter().copied())?);
        assert!(system.is_minimal_generating_set(generators.iter().copied())?);
        assert_eq!(system.minimum_generating_set(), basis);
        assert_eq!(system.generator_complexity(), basis.len());
        for removed in 0..generators.len() {
            let smaller = generators
                .iter()
                .enumerate()
                .filter_map(|(id, &edge)| (id != removed).then_some(edge));
            assert!(!system.is_generated_by(smaller)?);
        }

        let orbit_labels = g_lattice.non_identity_relation_orbit_labels();
        let mut minimum_size = usize::MAX;
        for mask in 0usize..(1usize << orbit_labels.len()) {
            let candidate = orbit_labels.iter().enumerate().filter_map(|(bit, label)| {
                ((mask >> bit) & 1 == 1).then_some(label.canonical_representative())
            });
            if system.is_generated_by(candidate)? {
                minimum_size = minimum_size.min(mask.count_ones() as usize);
            }
        }
        assert_eq!(system.generator_complexity(), minimum_size);

        // Blumberg--Hill compatibility specializes to saturation on the diagonal.
        assert_eq!(system.is_compatible_with(system), system.is_saturated());
        assert_eq!(
            system.is_compatible_with(system),
            underlying.is_compatible_with(&underlying)
        );
    }

    let complexity = systems
        .iter()
        .map(GTransferSystem::generator_complexity)
        .max()
        .unwrap_or(0);
    assert_eq!(g_lattice.transfer_system_complexity(), complexity);

    let complete = tr.system(tr.top());
    assert_eq!(complete, &g_lattice.complete_transfer_system());
    assert_eq!(
        g_lattice.transfer_system_width(),
        complete.generator_complexity()
    );
    Ok(())
}

#[test]
fn saturated_g_transfer_systems_form_the_claimed_containment_lattice_and_pairs_enumerate()
-> Result<(), Box<dyn Error>> {
    let g_lattice = swapping_diamond()?;
    let lattice = g_lattice.saturated_transfer_systems();
    let tr = g_lattice.transfer_systems();
    let all = tr.systems();

    assert_eq!(
        lattice.size(),
        all.iter().filter(|system| system.is_saturated()).count()
    );
    assert!(lattice.systems().iter().all(|system| system.is_saturated()));
    for left in lattice.ids() {
        for right in lattice.ids() {
            assert!(lattice.system(lattice.meet(left, right)).is_saturated());
            assert!(lattice.system(lattice.join(left, right)).is_saturated());
        }
    }

    for additive in all {
        for multiplicative in all {
            assert_eq!(
                additive.is_compatible_with(multiplicative),
                additive
                    .underlying_transfer_system()
                    .is_compatible_with(&multiplicative.underlying_transfer_system())
            );
        }
    }
    let expected_pair_count = all
        .iter()
        .flat_map(|additive| {
            all.iter()
                .filter(move |multiplicative| additive.is_compatible_with(multiplicative))
        })
        .count();
    let pairs = g_lattice.compatible_transfer_system_pairs();
    assert_eq!(pairs.len(), expected_pair_count);
    assert!(pairs.iter().all(|(additive, multiplicative)| {
        additive.compatibility_failure(multiplicative).is_none()
    }));
    Ok(())
}

fn contained(lower: &GTransferSystem, upper: &GTransferSystem) -> bool {
    let by_relations = lower.relations(false).is_subset(&upper.relations(false));
    assert_eq!(by_relations, lower <= upper);
    by_relations
}

fn swapping_diamond() -> Result<GLattice, Box<dyn Error>> {
    let lattice = Lattice::from_covers(
        ["bottom", "left", "right", "top"],
        [
            ("bottom", "left"),
            ("bottom", "right"),
            ("left", "top"),
            ("right", "top"),
        ],
    )?;
    let group = gap_sys::eval("Group((1,2));")?;
    Ok(GLattice::from_generator_images(
        &lattice,
        &group,
        vec![vec![0, 2, 1, 3]],
    )?)
}
