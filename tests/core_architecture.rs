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
        let relations = poset.relations().into_iter().collect::<Vec<_>>();
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

            assert_eq!(poset.left_lifting_class(&arrows), expected_left);
            assert_eq!(poset.right_lifting_class(&arrows), expected_right);
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
    poset.relations().into_iter().all(|first| {
        poset
            .relations()
            .into_iter()
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
        let relations = poset.relations().into_iter().collect::<Vec<_>>();
        for class_bits in 0usize..(1usize << relations.len()) {
            let class = relations
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(id, edge)| ((class_bits >> id) & 1 == 1).then_some(edge))
                .collect::<EdgeSet>();
            assert_eq!(
                poset.satisfies_two_out_of_three(&class),
                direct_two_out_of_three(&poset, &class),
            );
        }
    }

    assert!(!Poset::chain(1).satisfies_two_out_of_three(&EdgeSet::from([Edge::new(1, 0)])));
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
    let (joined, inclusions) = Lattice::horizontal_join_with_inclusions([&c3, &c3]).unwrap();
    assert_eq!(joined.size(), 6);
    assert!(joined.is_horizontal_join_of_total_orders());
    for inclusion in &inclusions {
        assert_eq!(inclusion.apply(c3.bottom()), joined.bottom());
        assert_eq!(inclusion.apply(c3.top()), joined.top());
    }
    for x in 1..3 {
        for y in 1..3 {
            let (a, b) = (inclusions[0].apply(x), inclusions[1].apply(y));
            assert!(!joined.leq(a, b) && !joined.leq(b, a));
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

/// The transitive closure of a set of relations `x < y`.
fn transitive_closure(relations: &EdgeSet) -> EdgeSet {
    let mut closure = relations.clone();
    loop {
        let composites: Vec<Edge> = closure
            .iter()
            .flat_map(|a| {
                closure
                    .iter()
                    .filter(move |b| b.from == a.to)
                    .map(move |b| Edge::new(a.from, b.to))
            })
            .collect();
        let before = closure.len();
        closure.extend(composites);
        if closure.len() == before {
            return closure;
        }
    }
}

/// Checks that `covers` are the covers of the partial order whose
/// non-identity relations are `proper`: they generate it under composition,
/// and none of them is a composite of two relations of it.
fn check_covers(proper: &EdgeSet, covers: &EdgeSet) {
    assert!(covers.is_subset(proper));
    assert_eq!(&transitive_closure(covers), proper);
    for cover in covers {
        assert!(
            !proper
                .iter()
                .any(|a| { a.from == cover.from && proper.contains(&Edge::new(a.to, cover.to)) })
        );
    }
}

#[test]
fn covers_generate_the_order_and_are_not_composites() {
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )
    .unwrap();
    for poset in [
        pentagon.as_poset().clone(),
        Lattice::boolean(3).as_poset().clone(),
        Poset::disjoint_union([Poset::chain(2), Poset::antichain(2)]),
    ] {
        let identities: EdgeSet = poset.ids().map(|id| Edge::new(id, id)).collect();
        let proper = poset.proper_relations();
        assert_eq!(
            poset.relations(),
            proper.union(&identities).copied().collect::<EdgeSet>()
        );
        check_covers(&proper, &poset.covers());
    }

    // A transfer system is a partial order on the lattice, and its covers
    // are taken in that order, not in the lattice.
    for lattice in [pentagon, Lattice::chain(3)] {
        for system in lattice.transfer_systems().systems() {
            check_covers(&system.proper_relations(), &system.covers());
        }
        for system in lattice.cotransfer_systems().systems() {
            check_covers(&system.proper_relations(), &system.covers());
        }
    }
}

/// The element-level queries agree with their definitions in terms of `<=`.
#[test]
fn element_queries_agree_with_the_order() {
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )
    .unwrap();
    for poset in [
        pentagon.as_poset().clone(),
        Lattice::boolean(3).as_poset().clone(),
        Poset::disjoint_union([Poset::chain(2), Poset::antichain(2)]),
    ] {
        let ids = || poset.ids();
        for x in ids() {
            let below: Vec<_> = ids().filter(|&y| poset.leq(y, x)).collect();
            let above: Vec<_> = ids().filter(|&y| poset.leq(x, y)).collect();
            assert_eq!(poset.down_set(x), below);
            assert_eq!(poset.up_set(x), above);
            let covered: Vec<_> = ids()
                .filter(|&y| poset.lt(y, x) && !ids().any(|z| poset.lt(y, z) && poset.lt(z, x)))
                .collect();
            let covering: Vec<_> = ids()
                .filter(|&y| poset.lt(x, y) && !ids().any(|z| poset.lt(x, z) && poset.lt(z, y)))
                .collect();
            assert_eq!(poset.lower_covers(x), covered);
            assert_eq!(poset.upper_covers(x), covering);
            for y in ids() {
                assert_eq!(poset.lt(x, y), x != y && poset.leq(x, y));
                assert_eq!(poset.comparable(x, y), poset.leq(x, y) || poset.leq(y, x));
                let between: Vec<_> = ids()
                    .filter(|&z| poset.leq(x, z) && poset.leq(z, y))
                    .collect();
                assert_eq!(poset.interval(x, y), between);
            }
            assert_eq!(poset.try_id(poset.label(x).clone()), Some(x));
        }
        assert_eq!(poset.try_id("not an element"), None);
    }
}

/// Equal values have equal hashes, so posets, lattices and transfer systems
/// can be used as set members and map keys.
#[test]
fn equal_values_hash_equally() {
    use std::collections::HashSet;
    let lattice = Lattice::boolean(2);
    let first = lattice.transfer_systems();
    let second = Lattice::boolean(2).transfer_systems();
    let systems: HashSet<_> = first.systems().iter().cloned().collect();
    assert_eq!(systems.len(), first.size());
    for system in second.systems() {
        assert!(systems.contains(system));
    }
    let posets: HashSet<Poset> = [lattice.as_poset().clone(), Lattice::boolean(2).into_poset()]
        .into_iter()
        .collect();
    assert_eq!(posets.len(), 1);
}

/// The concept lattice of a lattice's standard context is that lattice: the
/// map sending `x` to the concept (join-irreducibles below `x`,
/// meet-irreducibles above `x`) is an order isomorphism.
#[test]
fn the_standard_context_recovers_the_lattice() {
    use hccr::label::Label;
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )
    .unwrap();
    for lattice in [
        Lattice::chain(0),
        Lattice::chain(3),
        Lattice::boolean(3),
        pentagon,
        Lattice::chain(3).transfer_systems().as_lattice().clone(),
    ] {
        let concepts = Lattice::from_formal_context(&lattice.standard_context()).unwrap();
        assert_eq!(concepts.size(), lattice.size());
        let joins = lattice.join_irreducibles();
        let meets = lattice.meet_irreducibles();
        let image = |x: usize| {
            let label = Label::tuple([
                Label::set(
                    joins
                        .iter()
                        .filter(|&&j| lattice.leq(j, x))
                        .map(|&j| lattice.label(j).clone()),
                ),
                Label::set(
                    meets
                        .iter()
                        .filter(|&&m| lattice.leq(x, m))
                        .map(|&m| lattice.label(m).clone()),
                ),
            ]);
            concepts.id(label).unwrap()
        };
        for x in lattice.ids() {
            for y in lattice.ids() {
                assert_eq!(concepts.leq(image(x), image(y)), lattice.leq(x, y));
            }
        }
    }
}

fn hash_of<T: std::hash::Hash>(value: &T) -> u64 {
    use std::hash::{BuildHasher, RandomState};
    // One hasher state for the whole test, so that hashes are comparable.
    thread_local!(static STATE: RandomState = RandomState::new());
    STATE.with(|state| state.hash_one(value))
}

/// Posets, lattices and transfer systems compute some structure lazily, the
/// first time it is needed. Filling those caches must not change a value's
/// hash or equality, or a hash set holding the value would stop finding it.
#[test]
fn hashes_are_unchanged_by_lazily_computed_structure() {
    use std::collections::HashSet;
    let lattice = Lattice::boolean(2);
    let poset = lattice.as_poset().clone();
    let system = lattice.transfer_systems().system(1).clone();

    let before = (hash_of(&poset), hash_of(&lattice), hash_of(&system));
    let posets: HashSet<Poset> = HashSet::from([poset.clone()]);
    let lattices: HashSet<Lattice> = HashSet::from([lattice.clone()]);
    let systems: HashSet<_> = HashSet::from([system.clone()]);

    // Fill every cache: the lattice of transfer systems (already built
    // above), the lattice of cotransfer systems, and the model structures.
    let _ = lattice.cotransfer_systems();
    let _ = lattice.model_structures();
    let _ = poset.covers();

    assert_eq!(
        before,
        (hash_of(&poset), hash_of(&lattice), hash_of(&system))
    );
    assert!(posets.contains(&poset));
    assert!(lattices.contains(&lattice));
    assert!(systems.contains(&system));
}

/// A lattice is a horizontal join of total orders exactly when comparability
/// is an equivalence relation on its elements other than bottom and top, and
/// exactly when any two incomparable elements have meet bottom and join top.
/// `is_horizontal_join_of_total_orders` must agree with both
/// characterizations, whatever the element ids, including on lattices with
/// more than 64 elements.
#[test]
fn horizontal_join_of_total_orders_matches_definition() {
    fn comparability_is_an_equivalence_on_the_interior(lattice: &Lattice) -> bool {
        let interior = lattice
            .ids()
            .filter(|&x| x != lattice.bottom() && x != lattice.top())
            .collect::<Vec<_>>();
        interior.iter().all(|&x| {
            interior.iter().all(|&y| {
                interior.iter().all(|&z| {
                    !(lattice.comparable(x, y) && lattice.comparable(y, z))
                        || lattice.comparable(x, z)
                })
            })
        })
    }

    fn incomparables_meet_at_bottom_and_join_at_top(lattice: &Lattice) -> bool {
        lattice.ids().all(|x| {
            lattice.ids().all(|y| {
                lattice.comparable(x, y)
                    || (lattice.meet(x, y) == lattice.bottom()
                        && lattice.join(x, y) == lattice.top())
            })
        })
    }

    /// The same lattice with its element ids permuted.
    fn scrambled(lattice: &Lattice) -> Lattice {
        let mut order = lattice.ids().collect::<Vec<_>>();
        order.sort_by_key(|&x| (x % 3, std::cmp::Reverse(x)));
        Lattice::from_covers(
            order.iter().map(|&x| lattice.label(x).clone()),
            lattice.covers().iter().map(|edge| {
                (
                    lattice.label(edge.from).clone(),
                    lattice.label(edge.to).clone(),
                )
            }),
        )
        .unwrap()
    }

    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )
    .unwrap();
    // Incomparable b and c have meet bottom but join d, not top.
    let joined_below_top = Lattice::from_covers(
        ["bot", "b", "c", "d", "top"],
        [
            ("bot", "b"),
            ("bot", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "top"),
        ],
    )
    .unwrap();
    let large =
        Lattice::horizontal_join([Lattice::chain(40), Lattice::chain(30), Lattice::chain(5)])
            .unwrap();
    let large_with_cube =
        Lattice::horizontal_join([Lattice::chain(40), Lattice::boolean(3), Lattice::chain(30)])
            .unwrap();

    let examples = [
        Lattice::chain(0),
        Lattice::chain(1),
        Lattice::chain(4),
        Lattice::boolean(2),
        Lattice::horizontal_join([Lattice::chain(3), Lattice::chain(1), Lattice::chain(2)])
            .unwrap(),
        Lattice::horizontal_join(std::iter::empty::<Lattice>()).unwrap(),
        pentagon,
        Lattice::horizontal_join([Lattice::chain(2), Lattice::boolean(2)]).unwrap(),
        large.clone(),
    ];
    let non_examples = [
        joined_below_top,
        Lattice::boolean(3),
        Lattice::product([Lattice::chain(2), Lattice::chain(1)]),
        Lattice::horizontal_join([Lattice::chain(2), Lattice::boolean(3)]).unwrap(),
        large_with_cube,
        Lattice::product([large, Lattice::chain(1)]),
    ];
    for (lattices, expected) in [(&examples[..], true), (&non_examples[..], false)] {
        for lattice in lattices {
            for lattice in [lattice.clone(), lattice.opposite(), scrambled(lattice)] {
                assert_eq!(
                    comparability_is_an_equivalence_on_the_interior(&lattice),
                    expected,
                    "{lattice}"
                );
                assert_eq!(
                    incomparables_meet_at_bottom_and_join_at_top(&lattice),
                    expected,
                    "{lattice}"
                );
                assert_eq!(
                    lattice.is_horizontal_join_of_total_orders(),
                    expected,
                    "{lattice}"
                );
            }
        }
    }
}
