use hccr::lattice::Lattice;
use hccr::poset::{Edge, EdgeSet, Poset, compose, composition_closed};

/// Verifies that `Lattice`'s precomputed meet/join tables agree with the
/// independent `Poset::meet` and `Poset::join` methods for every pair of
/// elements.
fn check_meets_and_joins_agree(poset: Poset) {
    let lattice = Lattice::new(poset).expect("expected a valid lattice");
    let poset = lattice.as_poset();
    for i in 0..poset.size() {
        for j in 0..poset.size() {
            let expected_meet = poset.try_meet(i, j).expect("should have meet");
            let expected_join = poset.try_join(i, j).expect("should have join");
            assert_eq!(
                lattice.meet(i, j),
                expected_meet,
                "meet({i}, {j}): lattice returned {}, poset returned {expected_meet}",
                lattice.meet(i, j),
            );
            assert_eq!(
                lattice.join(i, j),
                expected_join,
                "join({i}, {j}): lattice returned {}, poset returned {expected_join}",
                lattice.join(i, j),
            );
        }
    }
}

#[test]
fn lattice_meet_join_agree_with_poset_methods() {
    // Single element.
    check_meets_and_joins_agree(Poset::from_edges(vec![0i32], [] as [Edge; 0]).unwrap());

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
    let left = Poset::chain(1);
    let right = Poset::chain(2);

    let (product, projections) = Poset::product_with_projections([&left, &right]);
    assert_eq!(product.size(), 6);
    for source in product.ids() {
        for target in product.ids() {
            let (source_left, source_right): (usize, usize) =
                product.label(source).try_into().unwrap();
            let (target_left, target_right): (usize, usize) =
                product.label(target).try_into().unwrap();
            assert_eq!(
                product.leq(source, target),
                left.leq(source_left, target_left) && right.leq(source_right, target_right),
            );
        }
    }
    assert_eq!(projections[0].images(), &[0, 0, 0, 1, 1, 1]);
    assert_eq!(projections[1].images(), &[0, 1, 2, 0, 1, 2]);

    let (coproduct, inclusions) = Poset::disjoint_union_with_inclusions([&left, &right]);
    assert_eq!(coproduct.size(), 5);
    assert!(coproduct.leq(0, 1));
    assert!(coproduct.leq(2, 4));
    assert!(!coproduct.leq(0, 2));
    assert!(!coproduct.leq(2, 0));
    assert_eq!(inclusions[0].images(), &[0, 1]);
    assert_eq!(inclusions[1].images(), &[2, 3, 4]);
    assert_eq!(coproduct.id((1, 2)).unwrap(), 4);
}

fn lifting_condition(poset: &Poset, left: Edge, right: Edge) -> bool {
    !poset.leq(left.from, right.from)
        || !poset.leq(left.to, right.to)
        || poset.leq(left.to, right.from)
}

/// Checks the wordwise lifting-class algorithms against their defining
/// universal properties for every class of arrows in several small posets.
#[test]
fn lifting_classes_satisfy_their_defining_conditions() {
    let posets = [
        Poset::chain(2),
        Poset::from_edges(
            0..4,
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

fn direct_two_out_of_three(poset: &Poset, class: &EdgeSet) -> bool {
    poset.all_relations_iter().all(|first| {
        poset
            .all_relations_iter()
            .filter(|second| first.to == second.from)
            .all(|second| {
                let composite = Edge::new(first.from, second.to);
                [first, second, composite]
                    .into_iter()
                    .filter(|edge| class.contains(edge))
                    .count()
                    != 2
            })
    })
}

/// Checks 2-out-of-3 against its defining condition for every relation class
/// on representative finite posets.
#[test]
fn two_out_of_three_satisfies_its_defining_condition() {
    let posets = [
        Poset::chain(2),
        Poset::from_edges(
            0..4,
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
            let class = relations
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(id, edge)| ((class_bits >> id) & 1 == 1).then_some(edge))
                .collect::<EdgeSet>();
            assert_eq!(
                poset.two_out_of_three(&class),
                direct_two_out_of_three(&poset, &class),
            );
        }
    }

    assert!(!Poset::chain(1).two_out_of_three(&EdgeSet::from([Edge::new(1, 0)])));
}

#[test]
fn constructions_have_the_expected_order_structure() {
    use hccr::prelude::{chain, horizontal_join, product};

    // The product order and meet are componentwise.
    let left = chain(1);
    let right = chain(2);
    let (square, projections) = Lattice::product_with_projections([&left, &right]);
    assert_eq!(square.size(), 6);
    for x in square.ids() {
        for y in square.ids() {
            let (p, q) = (&projections[0], &projections[1]);
            assert_eq!(
                square.leq(x, y),
                left.leq(p.apply(x), p.apply(y)) && right.leq(q.apply(x), q.apply(y))
            );
            assert_eq!(
                p.apply(square.meet(x, y)),
                left.meet(p.apply(x), p.apply(y))
            );
            assert_eq!(
                q.apply(square.join(x, y)),
                right.join(q.apply(x), q.apply(y))
            );
        }
    }

    // The horizontal join glues the factors along their bottoms and tops and
    // makes elements of different factors incomparable otherwise.
    let c3 = chain(3);
    let (fusion, inclusions) = Lattice::horizontal_join_with_inclusions([&c3, &c3]).unwrap();
    assert_eq!(fusion.size(), 6);
    assert!(fusion.is_fusion_of_total_orders());
    for inclusion in &inclusions {
        assert_eq!(inclusion.apply(c3.bottom()), fusion.bottom());
        assert_eq!(inclusion.apply(c3.top()), fusion.top());
    }
    for x in 1..3 {
        for y in 1..3 {
            let (a, b) = (inclusions[0].apply(x), inclusions[1].apply(y));
            assert!(!fusion.leq(a, b) && !fusion.leq(b, a));
        }
    }

    // The empty horizontal join is the two-element chain, its unit.
    let unit = horizontal_join([] as [Lattice; 0]).unwrap();
    assert_eq!(unit.size(), 2);
    assert!(unit.is_total_order());
    assert_eq!(
        horizontal_join([&c3, &unit])
            .unwrap()
            .transfer_system_count(),
        c3.transfer_system_count()
    );
    assert!(horizontal_join([chain(0)]).is_err());
    assert_eq!(product([chain(1), chain(1)]).transfer_system_count(), 10);
}
