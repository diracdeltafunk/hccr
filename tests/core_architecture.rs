use hccr::lattice::Lattice;
use hccr::poset::{self, Edge, Poset};
use hccr::tikz::{
    ToTikz, TransferSystemTikzOptions, transfer_system_lattice_to_tikz,
    transfer_system_order_to_tikz,
};
use std::sync::Arc;

/// Verifies that `Lattice`'s precomputed meet/join tables agree with the
/// independent `Poset::meet` and `Poset::join` methods for every pair of
/// elements.
fn check_meets_and_joins_agree<A: Clone + std::fmt::Debug>(poset: Poset<A>) {
    let lattice = Lattice::new(poset).expect("expected a valid lattice");
    let poset = lattice.as_poset();
    for i in 0..poset.size() {
        for j in 0..poset.size() {
            let expected_meet = poset.meet(i, j).expect("should have meet");
            let expected_join = poset.join(i, j).expect("should have join");
            assert_eq!(
                lattice.meet_id(i, j),
                expected_meet,
                "meet({i}, {j}): lattice returned {}, poset returned {expected_meet}",
                lattice.meet_id(i, j),
            );
            assert_eq!(
                lattice.join_id(i, j),
                expected_join,
                "join({i}, {j}): lattice returned {}, poset returned {expected_join}",
                lattice.join_id(i, j),
            );
        }
    }
}

#[test]
fn lattice_meet_join_agree_with_poset_methods() {
    // Single element.
    check_meets_and_joins_agree(Poset::from_edges(vec![0i32], []).unwrap());

    // Chain: 0 ≤ 1 ≤ 2 ≤ 3.
    check_meets_and_joins_agree(
        Poset::from_edges(
            vec![0i32, 1, 2, 3],
            [Edge::new(0, 1), Edge::new(1, 2), Edge::new(2, 3)],
        )
        .unwrap(),
    );

    // Boolean lattice B₂: bottom ≤ a, b ≤ top.
    check_meets_and_joins_agree(
        Poset::from_edges(
            vec![0i32, 1, 2, 3],
            [
                Edge::new(0, 1),
                Edge::new(0, 2),
                Edge::new(1, 3),
                Edge::new(2, 3),
            ],
        )
        .unwrap(),
    );

    // M₃: bottom ≤ a, b, c ≤ top (three incomparable middle elements).
    check_meets_and_joins_agree(
        Poset::from_edges(
            vec![0i32, 1, 2, 3, 4],
            [
                Edge::new(0, 1),
                Edge::new(0, 2),
                Edge::new(0, 3),
                Edge::new(1, 4),
                Edge::new(2, 4),
                Edge::new(3, 4),
            ],
        )
        .unwrap(),
    );

    // N₅ (pentagon): 0 < a < c < 1, 0 < b < 1, with a and b incomparable.
    // This non-modular lattice exercises asymmetric meet/join patterns.
    check_meets_and_joins_agree(
        Poset::from_edges(
            vec![0i32, 1, 2, 3, 4],
            [
                Edge::new(0, 1),
                Edge::new(1, 3),
                Edge::new(3, 4),
                Edge::new(0, 2),
                Edge::new(2, 4),
            ],
        )
        .unwrap(),
    );
}

#[test]
fn non_lattice_poset_is_rejected() {
    // Four elements a, b, c, d with a < c, a < d, b < c, b < d,
    // but c and d incomparable: join(a, b) has two minimal upper bounds
    // with no least one.
    let poset = Poset::from_edges(
        vec![0i32, 1, 2, 3],
        [
            Edge::new(0, 2),
            Edge::new(0, 3),
            Edge::new(1, 2),
            Edge::new(1, 3),
        ],
    )
    .unwrap();
    assert!(Lattice::new(poset).is_err());
}

#[test]
fn poset_product_and_disjoint_union_constructors_work() {
    let left = Arc::new(Poset::chain(1).unwrap());
    let right = Arc::new(Poset::chain(2).unwrap());

    let product = poset::product(Arc::clone(&left), Arc::clone(&right)).unwrap();
    assert_eq!(product.poset.size(), 6);
    assert!(product.poset.leq(0, 5));
    assert!(product.poset.leq(1, 4));
    assert!(!product.poset.leq(2, 3));
    assert_eq!(product.left_projection.map(), &[0, 0, 0, 1, 1, 1]);
    assert_eq!(product.right_projection.map(), &[0, 1, 2, 0, 1, 2]);

    let coproduct = poset::disjoint_union(Arc::clone(&left), Arc::clone(&right)).unwrap();
    assert_eq!(coproduct.poset.size(), 5);
    assert!(coproduct.poset.leq(0, 1));
    assert!(coproduct.poset.leq(2, 4));
    assert!(!coproduct.poset.leq(0, 2));
    assert!(!coproduct.poset.leq(2, 0));
    assert_eq!(coproduct.left.map(), &[0, 1]);
    assert_eq!(coproduct.right.map(), &[2, 3, 4]);
}

#[test]
fn transfer_system_orders_are_available_from_lattices() {
    let diamond = Arc::new(
        Lattice::new(
            Poset::from_edges(
                vec![0i32, 1, 2, 3],
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
    );

    let containment = Arc::clone(&diamond)
        .transfer_systems_containment()
        .expect("diamond should have a transfer-system containment lattice");
    let composition = Arc::clone(&diamond)
        .transfer_systems_composition_closed()
        .expect("diamond should have a transfer-system composition-closed order");

    assert!(containment.size() > 0);
    assert_eq!(containment.size(), composition.size());

    let _ = transfer_system_lattice_to_tikz(&containment);
    let _ = transfer_system_order_to_tikz(&composition);
    let _ = containment.to_tikz();
    let _ = composition.to_tikz_with(&TransferSystemTikzOptions::default());
}

#[test]
fn transfer_universe_can_be_reused_across_orders() {
    let diamond = Arc::new(
        Lattice::new(
            Poset::from_edges(
                vec![0i32, 1, 2, 3],
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
    );

    let universe = Arc::clone(&diamond).transfer_universe();
    let containment = universe
        .containment_lattice()
        .expect("diamond should have a transfer-system containment lattice");
    let composition = universe
        .composition_closed_order()
        .expect("diamond should have a transfer-system composition-closed order");

    assert!(Arc::ptr_eq(containment.universe(), &universe));
    assert!(Arc::ptr_eq(composition.universe(), &universe));
    assert_eq!(containment.size(), composition.size());
    assert_eq!(containment.to_system_lattice().size(), containment.size());
    assert_eq!(composition.to_system_poset().size(), composition.size());
    assert_eq!(
        containment
            .system(containment.bottom())
            .expect("bottom transfer system should exist")
            .edges(false)
            .len(),
        0
    );
}
