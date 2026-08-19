use hccr::cotransfer_lattice::{CotransferSystem, opposite_lattice};
use hccr::lattice::Lattice;
use hccr::poset::{Edge, Poset};
use std::error::Error;
use std::sync::Arc;

fn representative_lattices() -> Vec<Arc<Lattice<usize>>> {
    vec![
        Arc::new(Lattice::chain(0).unwrap()),
        Arc::new(Lattice::chain(3).unwrap()),
        Arc::new(Lattice::boolean(2).unwrap()),
        Arc::new(
            Lattice::new(
                Poset::from_edges(
                    (0..5).collect(),
                    [
                        Edge::new(0, 1),
                        Edge::new(1, 3),
                        Edge::new(3, 4),
                        Edge::new(0, 2),
                        Edge::new(2, 4),
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
        ),
    ]
}

fn assert_cotransfer_axioms<A>(system: &CotransferSystem<A>) {
    let lattice = system.lattice();
    let arrows = system.edges(true);

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
        for z in 0..lattice.size() {
            if lattice.leq(edge.from, z) {
                assert!(
                    arrows.contains(&Edge::new(z, lattice.join_id(edge.to, z))),
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
        let universe = Arc::clone(&lattice).cotransfer_universe();
        let systems = universe.cotransfer_systems();
        let containment = universe.containment_lattice()?;
        let opposite_count = universe
            .opposite_transfer_universe()
            .transfer_systems()
            .len();

        assert_eq!(systems.len(), opposite_count);
        assert_eq!(containment.size(), systems.len());
        for system in systems {
            assert_cotransfer_axioms(&system);

            let opposite = system.opposite_transfer_system();
            for edge in system.edges(true) {
                assert!(opposite.contains_relation(Edge::new(edge.to, edge.from)));
            }
            let round_trip = universe.from_opposite_transfer_system(&opposite)?;
            assert_eq!(round_trip, system);
        }
    }
    Ok(())
}

#[test]
fn lifting_classes_give_an_order_reversing_bijection() -> Result<(), Box<dyn Error>> {
    for lattice in representative_lattices() {
        let transfer_universe = Arc::clone(&lattice).transfer_universe();
        let cotransfer_universe = Arc::new(
            hccr::cotransfer_lattice::CotransferUniverse::from_transfer_universe(Arc::clone(
                &transfer_universe,
            )),
        );
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
                if lower.edges(false).is_subset(&upper.edges(false)) {
                    let lower_left = cotransfer_universe.left_lifting_of(lower)?;
                    let upper_left = cotransfer_universe.left_lifting_of(upper)?;
                    assert!(upper_left.edges(false).is_subset(&lower_left.edges(false)));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn generated_cotransfer_system_is_the_least_pushout_closed_system() {
    let lattice = Arc::new(Lattice::chain(2).unwrap());
    let generated = lattice
        .cotransfer_universe()
        .generated_by([Edge::new(0, 2)])
        .unwrap();
    assert_eq!(
        generated.edges(false),
        [Edge::new(0, 2), Edge::new(1, 2)].into_iter().collect()
    );
}

#[test]
fn opposite_lattice_interchanges_meets_and_joins() {
    for lattice in representative_lattices() {
        let opposite = opposite_lattice(lattice.as_ref());
        for left in 0..lattice.size() {
            for right in 0..lattice.size() {
                assert_eq!(opposite.meet_id(left, right), lattice.join_id(left, right));
                assert_eq!(opposite.join_id(left, right), lattice.meet_id(left, right));
                assert_eq!(opposite.leq(left, right), lattice.leq(right, left));
            }
        }
    }
}
