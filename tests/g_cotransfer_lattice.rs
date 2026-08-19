#![cfg(feature = "groups")]

use hccr::g_cotransfer_lattice::{GCotransferSystem, GCotransferUniverse};
use hccr::g_lattice::GLattice;
use hccr::lattice::Lattice;
use hccr::poset::{Edge, Poset};
use std::error::Error;
use std::sync::Arc;

fn diamond() -> Arc<Lattice<&'static str>> {
    Arc::new(
        Lattice::new(
            Poset::from_edges(
                vec!["bottom", "left", "right", "top"],
                [
                    Edge::new(0, 1),
                    Edge::new(0, 2),
                    Edge::new(1, 3),
                    Edge::new(2, 3),
                ],
            )
            .unwrap(),
        )
        .unwrap(),
    )
}

fn assert_cotransfer_axioms<A>(system: &GCotransferSystem<A>) {
    let lattice = system.lattice();
    let arrows = system.relations(true);
    for &first in &arrows {
        for &second in &arrows {
            if first.to == second.from {
                assert!(arrows.contains(&Edge::new(first.from, second.to)));
            }
        }
    }
    for &edge in &arrows {
        for z in 0..lattice.size() {
            if lattice.leq(edge.from, z) {
                assert!(arrows.contains(&Edge::new(z, lattice.join_id(edge.to, z))));
            }
        }
    }
}

#[test]
fn g_cotransfers_are_exactly_invariant_cotransfers_and_form_a_containment_lattice()
-> Result<(), Box<dyn Error>> {
    let group = gap_sys::eval("Group((1,2));")?;
    let g_lattice = GLattice::from_generator_images(diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    let universe = g_lattice.cotransfer_universe();
    let systems = universe.cotransfer_systems();
    let containment = universe.containment_lattice()?;

    assert_eq!(containment.size(), systems.len());
    assert_eq!(
        systems.len(),
        g_lattice.transfer_universe().transfer_systems().len()
    );
    for system in systems {
        assert_cotransfer_axioms(&system);
        assert_eq!(
            system.relations(false),
            system.underlying_cotransfer_system().edges(false)
        );
        for relation in system.relations(false) {
            let orbit = g_lattice.relation_orbit(relation).unwrap();
            assert!(
                orbit
                    .relations()
                    .iter()
                    .all(|&translate| system.contains_relation(translate))
            );
        }

        let opposite = system.opposite_transfer_system();
        universe
            .opposite_transfer_universe()
            .try_from_raw(opposite.raw().clone())?;
        for relation in system.relations(true) {
            assert!(opposite.contains_relation(Edge::new(relation.to, relation.from)));
        }
        assert_eq!(universe.from_opposite_transfer_system(&opposite)?, system);
    }
    for opposite in universe.opposite_transfer_universe().transfer_systems() {
        let cotransfer = universe.from_opposite_transfer_system(&opposite)?;
        assert_eq!(cotransfer.opposite_transfer_system(), opposite);
    }
    Ok(())
}

#[test]
fn opposite_isomorphism_agrees_on_a_non_self_dual_presentation() -> Result<(), Box<dyn Error>> {
    // N5 in these element coordinates is not unchanged when its order matrix
    // is transposed, unlike the symmetric diamond above.
    let n5 = Arc::new(Lattice::new(Poset::from_edges(
        (0..5).collect(),
        [
            Edge::new(0, 1),
            Edge::new(1, 3),
            Edge::new(3, 4),
            Edge::new(0, 2),
            Edge::new(2, 4),
        ],
    )?)?);
    let group = gap_sys::eval("TrivialGroup();")?;
    let g_lattice = GLattice::from_generator_images(n5, &group, vec![])?;
    let universe = g_lattice.cotransfer_universe();
    let cotransfers = universe.cotransfer_systems();
    let opposite_transfers = universe.opposite_transfer_universe().transfer_systems();

    assert_eq!(cotransfers.len(), opposite_transfers.len());
    for cotransfer in &cotransfers {
        universe
            .opposite_transfer_universe()
            .try_from_raw(cotransfer.opposite_transfer_system().raw().clone())?;
    }
    for opposite in opposite_transfers {
        let cotransfer = universe.from_opposite_transfer_system(&opposite)?;
        assert_eq!(cotransfer.opposite_transfer_system(), opposite);
    }
    Ok(())
}

#[test]
fn equivariant_lifting_classes_are_mutually_inverse_and_reverse_containment()
-> Result<(), Box<dyn Error>> {
    let group = gap_sys::eval("Group((1,2));")?;
    let g_lattice = GLattice::from_generator_images(diamond(), &group, vec![vec![0, 2, 1, 3]])?;
    let transfer_universe = g_lattice.transfer_universe();
    let cotransfer_universe = Arc::new(GCotransferUniverse::new(Arc::clone(&transfer_universe)));
    let transfers = transfer_universe.transfer_systems();

    for right in &transfers {
        let left = cotransfer_universe.left_lifting_of(right)?;
        assert_eq!(left.right_lifting_transfer()?, *right);
    }
    for left in cotransfer_universe.cotransfer_systems() {
        let right = left.right_lifting_transfer()?;
        assert_eq!(cotransfer_universe.left_lifting_of(&right)?, left);
    }
    for lower in &transfers {
        for upper in &transfers {
            if lower.relations(false).is_subset(&upper.relations(false)) {
                let lower_left = cotransfer_universe.left_lifting_of(lower)?;
                let upper_left = cotransfer_universe.left_lifting_of(upper)?;
                assert!(
                    upper_left
                        .relations(false)
                        .is_subset(&lower_left.relations(false))
                );
            }
        }
    }
    Ok(())
}
