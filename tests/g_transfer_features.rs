#![cfg(feature = "groups")]

use hccr::g_lattice::GLattice;
use hccr::g_transfer_lattice::{GTransferSystem, GTransferSystemError};
use hccr::lattice::Lattice;
use hccr::poset::Edge;
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
            saturated.proper_relations(),
            underlying.saturated_closure().proper_relations()
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
            cosaturated.proper_relations(),
            underlying.cosaturated_coclosure().proper_relations()
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

#[test]
fn minimal_generating_set_checks_reject_invalid_relations_even_with_identities()
-> Result<(), Box<dyn Error>> {
    let g_lattice = swapping_diamond()?;
    let size = g_lattice.size();
    let top = g_lattice.top();
    let left = g_lattice.id("left")?;
    let right = g_lattice.id("right")?;
    let system = g_lattice.complete_transfer_system();
    let cosystem = g_lattice.complete_cotransfer_system();

    // An out-of-range identity edge is invalid input, not a redundant generator.
    let out_of_range = Edge::new(size, size);
    assert_eq!(
        system.is_minimal_generating_set([out_of_range]),
        Err(GTransferSystemError::EdgeOutOfBounds {
            edge: out_of_range,
            lattice_size: size,
        })
    );
    assert_eq!(
        system.is_minimal_generating_set([out_of_range]).is_err(),
        system.is_generated_by([out_of_range]).is_err()
    );
    assert_eq!(
        system.is_minimal_generating_set([out_of_range]).is_err(),
        cosystem.is_minimal_generating_set([out_of_range]).is_err()
    );

    // An invalid generator is reported even when an identity is also supplied.
    let identity = Edge::new(top, top);
    let non_relation = Edge::new(left, right);
    assert_eq!(
        system.is_minimal_generating_set([identity, non_relation]),
        Err(GTransferSystemError::NotLatticeRelation { edge: non_relation })
    );
    assert_eq!(
        system.is_minimal_generating_set([identity, Edge::new(top, size)]),
        Err(GTransferSystemError::EdgeOutOfBounds {
            edge: Edge::new(top, size),
            lattice_size: size,
        })
    );

    // A valid identity still makes a generating collection redundant.
    let mut generators = system
        .minimal_generating_set()
        .iter()
        .map(|label| label.canonical_representative())
        .collect::<Vec<_>>();
    assert!(system.is_minimal_generating_set(generators.iter().copied())?);
    generators.push(identity);
    assert!(!system.is_minimal_generating_set(generators)?);
    Ok(())
}

fn contained(lower: &GTransferSystem, upper: &GTransferSystem) -> bool {
    let by_relations = lower
        .proper_relations()
        .is_subset(&upper.proper_relations());
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
