use hccr::lattice::Lattice;
use hccr::poset::{self, Edge, EdgeSet, Poset, compose, composition_closed};
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
    for source in 0..product.poset.size() {
        for target in 0..product.poset.size() {
            let &(source_left, source_right) = product.poset.element(source).unwrap();
            let &(target_left, target_right) = product.poset.element(target).unwrap();
            assert_eq!(
                product.poset.leq(source, target),
                left.leq(source_left, target_left) && right.leq(source_right, target_right),
            );
        }
    }
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

fn lifting_condition(poset: &Poset<usize>, left: Edge, right: Edge) -> bool {
    !poset.leq(left.from, right.from)
        || !poset.leq(left.to, right.to)
        || poset.leq(left.to, right.from)
}

/// Checks the wordwise lifting-class algorithms against their defining
/// universal properties for every class of arrows in several small posets.
#[test]
fn lifting_classes_satisfy_their_defining_conditions() {
    let posets = [
        Poset::chain(2).unwrap(),
        Poset::from_edges(
            (0..4).collect(),
            [
                Edge::new(0, 1),
                Edge::new(0, 2),
                Edge::new(1, 3),
                Edge::new(2, 3),
            ],
        )
        .unwrap(),
    ];

    for poset in posets {
        let relations = poset.all_relations_iter().collect::<Vec<_>>();
        for class_bits in 0usize..(1usize << relations.len()) {
            let arrows = relations
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(id, edge)| ((class_bits >> id) & 1 == 1).then_some(edge))
                .collect::<EdgeSet>();

            let expected_left = relations
                .iter()
                .copied()
                .filter(|&left| {
                    arrows
                        .iter()
                        .all(|&right| lifting_condition(&poset, left, right))
                })
                .collect::<EdgeSet>();
            let expected_right = relations
                .iter()
                .copied()
                .filter(|&right| {
                    arrows
                        .iter()
                        .all(|&left| lifting_condition(&poset, left, right))
                })
                .collect::<EdgeSet>();

            assert_eq!(poset.llc(&arrows), expected_left);
            assert_eq!(poset.rlc(&arrows), expected_right);
        }
    }
}

/// Checks composition against its definition for every class of arrows on
/// three objects. Closure is equivalent to containing every such composite.
#[test]
fn relation_composition_and_closure_satisfy_their_definitions() {
    let relations = (0..3)
        .flat_map(|from| (0..3).map(move |to| Edge::new(from, to)))
        .collect::<Vec<_>>();

    for class_bits in 0usize..(1usize << relations.len()) {
        let class = relations
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(id, edge)| ((class_bits >> id) & 1 == 1).then_some(edge))
            .collect::<EdgeSet>();
        let expected = class
            .iter()
            .flat_map(|edge1| {
                class.iter().filter_map(move |edge2| {
                    (edge2.to == edge1.from).then_some(Edge::new(edge2.from, edge1.to))
                })
            })
            .collect::<EdgeSet>();

        assert_eq!(compose(&class, &class), expected);
        assert_eq!(composition_closed(&class), expected.is_subset(&class));
    }
}
