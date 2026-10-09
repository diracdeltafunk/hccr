use hccr::cotransfer_lattice::CotransferSystem;
use hccr::lattice::Lattice;
use hccr::poset::Edge;
use std::error::Error;

fn representative_lattices() -> Vec<Lattice> {
    vec![
        Lattice::chain(0),
        Lattice::chain(3),
        Lattice::boolean(2),
        Lattice::from_covers(
            ["0", "a", "b", "c", "1"],
            [("0", "a"), ("a", "c"), ("c", "1"), ("0", "b"), ("b", "1")],
        )
        .unwrap(),
    ]
}

fn assert_cotransfer_axioms(system: &CotransferSystem) {
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
        for z in lattice.ids() {
            if lattice.leq(edge.from, z) {
                assert!(
                    arrows.contains(&Edge::new(z, lattice.join(edge.to, z))),
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
        let containment = lattice.cotransfer_systems();
        let opposite_count = lattice.opposite().transfer_system_count();

        assert_eq!(containment.size(), opposite_count);
        assert_eq!(lattice.cotransfer_system_count(), opposite_count);
        for system in containment.systems() {
            assert_cotransfer_axioms(system);

            let opposite = system.opposite_transfer_system();
            assert_eq!(*opposite.lattice(), lattice.opposite());
            for edge in system.edges(true) {
                assert!(opposite.contains_relation(edge.reversed()));
            }
            let round_trip = lattice.cotransfer_system_from_opposite(&opposite)?;
            assert_eq!(&round_trip, system);
        }
    }
    Ok(())
}

#[test]
fn lifting_classes_give_an_order_reversing_bijection() {
    for lattice in representative_lattices() {
        let transfers = lattice.transfer_systems();

        for right in transfers.systems() {
            let left = right.left_lifting_cotransfer();
            assert_eq!(left.right_lifting_transfer(), *right);
        }

        for left in lattice.cotransfer_systems().systems() {
            let right = left.right_lifting_transfer();
            assert_eq!(right.left_lifting_cotransfer(), *left);
        }

        for lower in transfers.systems() {
            for upper in transfers.systems() {
                if lower <= upper {
                    assert!(upper.left_lifting_cotransfer() <= lower.left_lifting_cotransfer());
                }
            }
        }
    }
}

#[test]
fn generated_cotransfer_system_is_the_least_pushout_closed_system() {
    let lattice = Lattice::chain(2);
    let generated = lattice
        .cotransfer_system_generated_by([Edge::new(0, 2)])
        .unwrap();
    assert_eq!(
        generated.edges(false),
        [Edge::new(0, 2), Edge::new(1, 2)].into_iter().collect()
    );
}

#[test]
fn opposite_lattice_interchanges_meets_and_joins() {
    for lattice in representative_lattices() {
        let opposite = lattice.opposite();
        assert_eq!(opposite.bottom(), lattice.top());
        for left in lattice.ids() {
            for right in lattice.ids() {
                assert_eq!(opposite.meet(left, right), lattice.join(left, right));
                assert_eq!(opposite.join(left, right), lattice.meet(left, right));
                assert_eq!(opposite.leq(left, right), lattice.leq(right, left));
            }
        }
        assert_eq!(opposite.opposite(), lattice);
    }
}
