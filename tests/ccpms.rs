use hccr::lattice::{Lattice, horizontal_join};
use std::sync::Arc;

#[test]
fn composition_closed_orders_on_three_chain_joins_are_lattices() {
    for a in 1..=3 {
        for b in a..=3 {
            for c in b..=3 {
                let ab = horizontal_join(
                    Arc::new(Lattice::chain(a).unwrap()),
                    Arc::new(Lattice::chain(b).unwrap()),
                )
                .unwrap();
                let abc = horizontal_join(ab.lattice, Arc::new(Lattice::chain(c).unwrap()))
                    .unwrap()
                    .lattice;
                let composition = Arc::clone(&abc)
                    .transfer_systems_composition_closed()
                    .unwrap();

                assert!(
                    composition.raw_poset().is_lattice(),
                    "CC([{}]*[{}]*[{}]) should be a lattice",
                    a,
                    b,
                    c
                );
            }
        }
    }
}

#[test]
fn composition_closed_orders_on_non_chain_joins_are_not_lattices() {
    let product = Lattice::product(
        Arc::new(Lattice::chain(1).unwrap()),
        Arc::new(Lattice::chain(2).unwrap()),
    )
    .unwrap()
    .lattice
    .relabelled(|element| format!("{element:?}"));

    for (name, lattice) in [
        (
            "B3",
            Lattice::boolean(3)
                .unwrap()
                .relabelled(|element| format!("{element:b}")),
        ),
        ("[1] x [2]", product),
    ] {
        assert!(
            !lattice.is_fusion_of_total_orders(),
            "{name} should not be a horizontal join of chains"
        );

        let lattice = Arc::new(lattice);
        let composition = Arc::clone(&lattice)
            .transfer_systems_composition_closed()
            .unwrap();

        assert!(
            !composition.raw_poset().is_lattice(),
            "CC({name}) should not be a lattice"
        );
    }
}
