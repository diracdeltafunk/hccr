#![cfg(feature = "groups")]

use hccr::g_cotransfer_lattice::GCotransferSystem;
use hccr::g_lattice::GLattice;
use hccr::g_transfer_lattice::GTransferSystem;
use hccr::lattice::Lattice;
use hccr::poset::Edge;
use std::collections::HashSet;
use std::error::Error;

fn diamond() -> Lattice {
    Lattice::from_covers(
        ["bottom", "left", "right", "top"],
        [
            ("bottom", "left"),
            ("bottom", "right"),
            ("left", "top"),
            ("right", "top"),
        ],
    )
    .unwrap()
}

fn assert_cotransfer_axioms(system: &GCotransferSystem) {
    let lattice = system.lattice();
    let arrows = system.relations();
    for &first in &arrows {
        for &second in &arrows {
            if first.to == second.from {
                assert!(arrows.contains(&Edge::new(first.from, second.to)));
            }
        }
    }
    for &edge in &arrows {
        for z in lattice.ids() {
            if lattice.leq(edge.from, z) {
                assert!(arrows.contains(&Edge::new(z, lattice.join(edge.to, z))));
            }
        }
    }
}

/// The sets of relations of all G-transfer systems on the opposite lattice,
/// computed independently from a G-lattice built on `L^op`.
fn opposite_transfer_relation_sets(
    lattice: &Lattice,
    group: &gap_sys::GapValue,
    generator_images: Vec<Vec<usize>>,
) -> Result<HashSet<Vec<Edge>>, Box<dyn Error>> {
    let opposite = GLattice::from_generator_images(&lattice.opposite(), group, generator_images)?;
    Ok(opposite
        .transfer_systems()
        .systems()
        .iter()
        .map(sorted)
        .collect())
}

fn sorted(system: &GTransferSystem) -> Vec<Edge> {
    let mut relations = system.proper_relations().into_iter().collect::<Vec<_>>();
    relations.sort_unstable();
    relations
}

#[test]
fn g_cotransfers_are_exactly_invariant_cotransfers_and_form_a_containment_lattice()
-> Result<(), Box<dyn Error>> {
    let group = gap_sys::eval("Group((1,2));")?;
    let g_lattice = GLattice::from_generator_images(&diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    let containment = g_lattice.cotransfer_systems();

    assert_eq!(containment.size(), g_lattice.cotransfer_system_count());
    assert_eq!(containment.size(), g_lattice.transfer_system_count());
    let expected_opposites =
        opposite_transfer_relation_sets(&diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    assert_eq!(expected_opposites.len(), containment.size());

    for system in &containment {
        assert_cotransfer_axioms(system);
        assert_eq!(
            system.proper_relations(),
            system.underlying_cotransfer_system().proper_relations()
        );
        for relation in system.proper_relations() {
            let orbit = g_lattice.relation_orbit(relation).unwrap();
            assert!(
                orbit
                    .relations()
                    .iter()
                    .all(|&translate| system.contains_relation(translate))
            );
        }

        let opposite = system.opposite_transfer_system();
        assert!(opposite.is_generated_by(opposite.proper_relations())?);
        for relation in system.relations() {
            assert!(opposite.contains_relation(relation.reversed()));
        }
        assert!(expected_opposites.contains(&sorted(&opposite)));
        assert_eq!(
            &g_lattice.cotransfer_system_from_opposite(&opposite)?,
            system
        );
        assert_eq!(
            g_lattice
                .cotransfer_system_from_opposite(&opposite)?
                .opposite_transfer_system(),
            opposite
        );
    }
    Ok(())
}

#[test]
fn opposite_isomorphism_agrees_on_a_non_self_dual_presentation() -> Result<(), Box<dyn Error>> {
    // N5 in these element coordinates is not unchanged when its order matrix
    // is transposed, unlike the symmetric diamond above.
    let n5 = Lattice::from_covers([0, 1, 2, 3, 4], [(0, 1), (1, 3), (3, 4), (0, 2), (2, 4)])?;
    let group = gap_sys::eval("TrivialGroup();")?;
    let g_lattice = GLattice::from_generator_images(&n5, &group, vec![])?;
    let cotransfers = g_lattice.cotransfer_systems();
    let expected_opposites = opposite_transfer_relation_sets(&n5, &group, vec![])?;

    assert_eq!(cotransfers.size(), expected_opposites.len());
    assert_eq!(cotransfers.size(), n5.cotransfer_system_count());
    for cotransfer in &cotransfers {
        let opposite = cotransfer.opposite_transfer_system();
        assert!(expected_opposites.contains(&sorted(&opposite)));
        let round_trip = g_lattice.cotransfer_system_from_opposite(&opposite)?;
        assert_eq!(&round_trip, cotransfer);
        assert_eq!(round_trip.opposite_transfer_system(), opposite);
    }

    // A G-transfer system on an unrelated G-lattice is rejected.
    let unrelated = GLattice::from_generator_images(&n5.opposite(), &group, vec![])?;
    assert!(
        g_lattice
            .cotransfer_system_from_opposite(&unrelated.trivial_transfer_system())
            .is_err()
    );
    Ok(())
}

#[test]
fn equivariant_lifting_classes_are_mutually_inverse_and_reverse_containment()
-> Result<(), Box<dyn Error>> {
    let group = gap_sys::eval("Group((1,2));")?;
    let g_lattice = GLattice::from_generator_images(&diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    let transfers = g_lattice.transfer_systems();

    for right in &transfers {
        let left = right.left_lifting_cotransfer();
        assert_eq!(left.right_lifting_transfer(), *right);
    }
    for left in &g_lattice.cotransfer_systems() {
        let right = left.right_lifting_transfer();
        assert_eq!(right.left_lifting_cotransfer(), *left);
    }
    for lower in &transfers {
        for upper in &transfers {
            if lower <= upper {
                assert!(upper.left_lifting_cotransfer() <= lower.left_lifting_cotransfer());
            }
        }
    }
    Ok(())
}

/// A minimal set of orbit generators generates the G-cotransfer system, no
/// proper subset does, and the trivial and complete systems are the bottom
/// and top of the containment lattice.
#[test]
fn g_cotransfer_minimal_generating_sets_generate_irredundantly() -> Result<(), Box<dyn Error>> {
    let group = gap_sys::eval("Group((1,2));")?;
    let g_lattice = GLattice::from_generator_images(&diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    let containment = g_lattice.cotransfer_systems();
    for system in containment.systems() {
        let basis: Vec<Edge> = system
            .minimal_generating_set()
            .iter()
            .map(|label| label.canonical_representative())
            .collect();
        assert!(system.is_generated_by(basis.iter().copied())?);
        assert!(system.is_minimal_generating_set(basis.iter().copied())?);
        assert_eq!(basis.len(), system.generator_complexity());
    }
    assert_eq!(
        &g_lattice.trivial_cotransfer_system(),
        containment.system(containment.bottom())
    );
    assert_eq!(
        &g_lattice.complete_cotransfer_system(),
        containment.system(containment.top())
    );
    Ok(())
}
