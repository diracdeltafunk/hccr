use hccr::lattice::Lattice;
use hccr::morphism::{LatticeMap, PosetMap};
use hccr::poset::{Edge, EdgeSet};
use hccr::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use hccr::transfer_morphism::{
    CompositionMapError, generated_inverse_image, generated_inverse_image_containment_map,
    pullback, pullback_containment_map, pushforward, pushforward_containment_map,
    try_pullback_composition_map, try_pushforward_composition_map,
};

fn chain(top: usize) -> Lattice {
    Lattice::chain(top)
}

fn boolean_two() -> Lattice {
    Lattice::boolean(2)
}

fn proper_edges(pairs: impl IntoIterator<Item = (usize, usize)>) -> EdgeSet {
    pairs.into_iter().map(Edge::from).collect()
}

fn containment_id(order: &TransferLattice, system: &TransferSystem) -> usize {
    order
        .id_of(system)
        .expect("the transfer system should occur in its containment lattice")
}

fn composition_id(order: &TransferPoset, system: &TransferSystem) -> usize {
    order
        .id_of(system)
        .expect("the transfer system should occur in its composition-closed order")
}

fn assert_batch_pushforward_agrees(
    homomorphism: &LatticeMap,
    domain: &TransferLattice,
    codomain: &TransferLattice,
) {
    let map = pushforward_containment_map(homomorphism, domain, codomain).unwrap();
    for (source_id, system) in domain.systems().iter().enumerate() {
        let expected = pushforward(homomorphism, system).unwrap();
        assert!(
            codomain.system(map.apply(source_id)) == &expected,
            "batch pushforward disagrees at source element {source_id}"
        );
    }
}

fn assert_batch_pullback_agrees(
    homomorphism: &LatticeMap,
    codomain: &TransferLattice,
    domain: &TransferLattice,
) {
    let map = pullback_containment_map(homomorphism, codomain, domain).unwrap();
    for (source_id, system) in codomain.systems().iter().enumerate() {
        let expected = pullback(homomorphism, system).unwrap();
        assert!(
            domain.system(map.apply(source_id)) == &expected,
            "batch pullback disagrees at source element {source_id}"
        );
    }
}

fn assert_composition_pushforward_agrees(
    homomorphism: &LatticeMap,
    domain: &TransferPoset,
    codomain: &TransferPoset,
) {
    let map = try_pushforward_composition_map(homomorphism, domain, codomain).unwrap();
    for (source_id, system) in domain.systems().iter().enumerate() {
        let expected = pushforward(homomorphism, system).unwrap();
        let expected_id = composition_id(codomain, &expected);
        assert_eq!(map.apply(source_id), expected_id);
        assert!(codomain.system(expected_id) == &expected);
    }
}

fn assert_composition_pullback_agrees(
    homomorphism: &LatticeMap,
    codomain: &TransferPoset,
    domain: &TransferPoset,
) {
    let map = try_pullback_composition_map(homomorphism, codomain, domain).unwrap();
    for (source_id, system) in codomain.systems().iter().enumerate() {
        let expected = pullback(homomorphism, system).unwrap();
        let expected_id = composition_id(domain, &expected);
        assert_eq!(map.apply(source_id), expected_id);
        assert!(domain.system(expected_id) == &expected);
    }
}

/// Checks that `error` exhibits a genuine failure of monotonicity for the
/// map of composition-closed orders induced by `image`: a cover `R < S` in
/// `source` whose images are incomparable in `target`, together with a square
/// formed by an arrow `x -> y` of `image(R)` and an arrow `x' -> y'` of
/// `image(S)`, with `x <= x'` and `y <= y'`, that has no factorization
/// `x <= z <= x'`, `y <= w` with `z -> w` in `image(R)` and `w -> y'` in
/// `image(S)`.
fn assert_is_a_monotonicity_failure(
    error: &CompositionMapError,
    source: &TransferPoset,
    target: &TransferPoset,
    image: impl Fn(&TransferSystem) -> TransferSystem,
) {
    let CompositionMapError::NotMonotone {
        source_cover,
        lower_image,
        upper_image,
        failed_square: (first, second),
    } = *error
    else {
        panic!("expected a monotonicity failure, got {error:?}")
    };

    assert!(source.cover_relations().contains(&source_cover));
    let lower = image(source.system(source_cover.from));
    let upper = image(source.system(source_cover.to));
    assert_eq!(target.system(lower_image), &lower);
    assert_eq!(target.system(upper_image), &upper);
    assert!(!target.leq(lower_image, upper_image));

    let lattice = target.base_lattice();
    assert!(lower.contains_relation(first));
    assert!(upper.contains_relation(second));
    assert!(lattice.leq(first.from, second.from) && lattice.leq(first.to, second.to));
    let has_witness = lattice.ids().any(|z| {
        lattice.leq(first.from, z)
            && lattice.leq(z, second.from)
            && lattice.ids().any(|w| {
                lattice.leq(first.to, w)
                    && lower.contains_relation(Edge::new(z, w))
                    && upper.contains_relation(Edge::new(w, second.to))
            })
    });
    assert!(!has_witness, "the reported square has a factorization");
}

#[test]
fn pointwise_maps_close_images_and_handle_collapsed_and_duplicate_arrows() {
    let domain_lattice = chain(2);
    let codomain_lattice = chain(1);
    let quotient = LatticeMap::new(&domain_lattice, &codomain_lattice, vec![0, 0, 1]).unwrap();
    let domain = domain_lattice.clone();
    let codomain = codomain_lattice.clone();

    // Both 0 -> 2 and 1 -> 2 have image 0 -> 1. The first generator also
    // forces 0 -> 1 in the source by restriction.
    let source_top = domain
        .transfer_system_generated_by([Edge::new(0, 2), Edge::new(1, 2)])
        .unwrap();
    assert_eq!(
        source_top.edges(false),
        proper_edges([(0, 1), (0, 2), (1, 2)])
    );
    let image = pushforward(&quotient, &source_top).unwrap();
    assert_eq!(image.edges(false), proper_edges([(0, 1)]));

    // Pulling back the diagonal includes every arrow collapsed by the map.
    let target_bottom = codomain
        .transfer_system_generated_by(std::iter::empty::<Edge>())
        .unwrap();
    let inverse_image = pullback(&quotient, &target_bottom).unwrap();
    assert_eq!(inverse_image.edges(false), proper_edges([(0, 1)]));

    // Meet preservation makes the raw inverse image a transfer system, so
    // the right-adjoint and generated variants agree without extra closure.
    // Forgetting the certified lattice structure selects the general
    // right-adjoint algorithm and must produce the same result.
    let quotient_as_poset_map = quotient.as_poset_map();
    for target in codomain.transfer_systems().systems() {
        let fast_pullback = pullback(&quotient, target).unwrap();
        assert_eq!(
            generated_inverse_image(&quotient, target).unwrap(),
            fast_pullback
        );
        assert_eq!(
            pullback(&quotient_as_poset_map, target).unwrap(),
            fast_pullback
        );
    }
}

#[test]
fn containment_maps_are_pointwise_correct_adjoint_and_preserve_the_expected_operations() {
    let domain_lattice = chain(2);
    let codomain_lattice = chain(1);
    let quotient = LatticeMap::new(&domain_lattice, &codomain_lattice, vec![0, 0, 1]).unwrap();
    let domain_universe = domain_lattice.clone();
    let codomain_universe = codomain_lattice.clone();
    let domain = domain_universe.transfer_systems();
    let codomain = codomain_universe.transfer_systems();

    assert_batch_pushforward_agrees(&quotient, &domain, &codomain);
    assert_batch_pullback_agrees(&quotient, &codomain, &domain);

    let push = pushforward_containment_map(&quotient, &domain, &codomain).unwrap();
    let pull = pullback_containment_map(&quotient, &codomain, &domain).unwrap();

    for r in 0..domain.size() {
        for s in 0..codomain.size() {
            assert_eq!(
                codomain.leq(push.apply(r), s),
                domain.leq(r, pull.apply(s)),
                "the containment adjunction failed for ({r}, {s})"
            );
        }
    }

    assert_eq!(push.apply(domain.bottom()), codomain.bottom());
    for left in 0..domain.size() {
        for right in 0..domain.size() {
            assert_eq!(
                push.apply(domain.join(left, right)),
                codomain.join(push.apply(left), push.apply(right))
            );
        }
    }

    assert_eq!(pull.apply(codomain.top()), domain.top());
    for left in 0..codomain.size() {
        for right in 0..codomain.size() {
            assert_eq!(
                pull.apply(codomain.meet(left, right)),
                domain.meet(pull.apply(left), pull.apply(right))
            );
        }
    }
}

#[test]
fn composition_monotonicity_is_checked_independently_in_each_direction() {
    // C3 -> B2 has non-monotone pushforward but monotone pullback.
    let c3 = chain(2);
    let b2 = boolean_two();
    let into_b2 = LatticeMap::new(&c3, &b2, vec![0, 1, 3]).unwrap();
    let c3_universe = c3.clone();
    let b2_universe = b2.clone();
    let cc_c3 = c3_universe.transfer_systems_composition_closed();
    let cc_b2 = b2_universe.transfer_systems_composition_closed();

    let r = c3_universe
        .transfer_system_generated_by([Edge::new(1, 2)])
        .unwrap();
    let s = c3_universe
        .transfer_system_generated_by([Edge::new(0, 1), Edge::new(1, 2)])
        .unwrap();
    assert!(cc_c3.leq(composition_id(&cc_c3, &r), composition_id(&cc_c3, &s)));
    let image_r = pushforward(&into_b2, &r).unwrap();
    let image_s = pushforward(&into_b2, &s).unwrap();
    assert_eq!(image_r.edges(false), proper_edges([(0, 2), (1, 3)]));
    assert_eq!(
        image_s.edges(false),
        proper_edges([(0, 1), (0, 2), (0, 3), (1, 3)])
    );
    assert!(!cc_b2.leq(
        composition_id(&cc_b2, &image_r),
        composition_id(&cc_b2, &image_s)
    ));
    let push_error = try_pushforward_composition_map(&into_b2, &cc_c3, &cc_b2).unwrap_err();
    assert_is_a_monotonicity_failure(&push_error, &cc_c3, &cc_b2, |system| {
        pushforward(&into_b2, system).unwrap()
    });
    assert_composition_pullback_agrees(&into_b2, &cc_b2, &cc_c3);

    // C3 -> C4 has monotone pushforward but non-monotone pullback.
    let c4 = chain(3);
    let into_c4 = LatticeMap::new(&c3, &c4, vec![0, 1, 3]).unwrap();
    let c4_universe = c4.clone();
    let cc_c4 = c4_universe.transfer_systems_composition_closed();
    let r = c4_universe
        .transfer_system_generated_by([Edge::new(0, 2)])
        .unwrap();
    let s = c4_universe
        .transfer_system_generated_by([Edge::new(0, 3), Edge::new(2, 3)])
        .unwrap();
    assert!(cc_c4.leq(composition_id(&cc_c4, &r), composition_id(&cc_c4, &s)));
    let inverse_r = pullback(&into_c4, &r).unwrap();
    let inverse_s = pullback(&into_c4, &s).unwrap();
    assert_eq!(inverse_r.edges(false), proper_edges([(0, 1)]));
    assert_eq!(inverse_s.edges(false), proper_edges([(0, 1), (0, 2)]));
    assert!(!cc_c3.leq(
        composition_id(&cc_c3, &inverse_r),
        composition_id(&cc_c3, &inverse_s)
    ));
    assert_composition_pushforward_agrees(&into_c4, &cc_c3, &cc_c4);
    let pull_error = try_pullback_composition_map(&into_c4, &cc_c4, &cc_c3).unwrap_err();
    assert_is_a_monotonicity_failure(&pull_error, &cc_c4, &cc_c3, |system| {
        pullback(&into_c4, system).unwrap()
    });
}

#[test]
fn pointwise_maps_are_functorial_and_identity_maps_work_for_both_orders() {
    let c4 = chain(3);
    let c3 = chain(2);
    let c2 = chain(1);
    let f = LatticeMap::new(&c4, &c3, vec![0, 0, 1, 2]).unwrap();
    let g = LatticeMap::new(&c3, &c2, vec![0, 0, 1]).unwrap();
    let composite = LatticeMap::new(&c4, &c2, vec![0, 0, 0, 1]).unwrap();
    let identity = LatticeMap::new(&c3, &c3, vec![0, 1, 2]).unwrap();
    assert_eq!(g.compose(&f).unwrap(), composite);
    assert_eq!(identity, LatticeMap::identity(&c3));
    let u4 = c4.clone();
    let u3 = c3.clone();
    let u2 = c2.clone();

    for system in u4.transfer_systems() {
        let sequential = pushforward(&g, &pushforward(&f, &system).unwrap()).unwrap();
        assert_eq!(sequential, pushforward(&composite, &system).unwrap());
    }
    for system in u2.transfer_systems() {
        let sequential = pullback(&f, &pullback(&g, &system).unwrap()).unwrap();
        assert_eq!(sequential, pullback(&composite, &system).unwrap());
    }
    for system in u3.transfer_systems() {
        assert_eq!(pushforward(&identity, &system).unwrap(), system);
        assert_eq!(pullback(&identity, &system).unwrap(), system);
    }

    let containment = u3.transfer_systems();
    let composition = u3.transfer_systems_composition_closed();
    assert!(pushforward_containment_map(&identity, &containment, &containment).is_ok());
    assert!(pullback_containment_map(&identity, &containment, &containment).is_ok());
    assert!(try_pushforward_composition_map(&identity, &composition, &composition).is_ok());
    assert!(try_pullback_composition_map(&identity, &composition, &composition).is_ok());
}

#[test]
fn monotone_map_pullback_is_the_right_adjoint_but_generated_inverse_image_is_not() {
    // The two atoms and the top all map to the top of C2. This is monotone,
    // but it does not preserve the meet of the atoms.
    let b2 = boolean_two();
    let c2 = chain(1);
    let f = PosetMap::new(&b2, &c2, vec![0, 1, 1, 1]).unwrap();
    let b2_universe = b2.clone();
    let c2_universe = c2.clone();
    let tr_b2 = b2_universe.transfer_systems();
    let tr_c2 = c2_universe.transfer_systems();

    let direct = pushforward_containment_map(&f, &tr_b2, &tr_c2).unwrap();
    let right_adjoint = pullback_containment_map(&f, &tr_c2, &tr_b2).unwrap();
    for (source_id, system) in tr_c2.systems().iter().enumerate() {
        let expected = pullback(&f, system).unwrap();
        let actual = tr_b2.system(right_adjoint.apply(source_id));
        assert_eq!(actual, &expected);
    }
    for r in 0..tr_b2.size() {
        for s in 0..tr_c2.size() {
            assert_eq!(
                tr_c2.leq(direct.apply(r), s),
                tr_b2.leq(r, right_adjoint.apply(s)),
                "the containment adjunction failed for ({r}, {s})"
            );
        }
    }

    let target_bottom = c2_universe
        .transfer_system_generated_by(std::iter::empty::<Edge>())
        .unwrap();
    let corrected = pullback(&f, &target_bottom).unwrap();
    assert_eq!(corrected.edges(false), EdgeSet::new());

    // The raw inverse image consists of the two collapsed arrows from the
    // atoms to the top. Closing those arrows forces every arrow of B2.
    let generated = generated_inverse_image(&f, &target_bottom).unwrap();
    assert_eq!(
        generated.edges(false),
        proper_edges([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
    );
    let generated_map = generated_inverse_image_containment_map(&f, &tr_c2, &tr_b2).unwrap();
    assert_eq!(generated_map.apply(tr_c2.bottom()), tr_b2.top());

    // This is an explicit failure of the converse adjunction implication for
    // the generated inverse image.
    let r = b2_universe
        .transfer_system_generated_by([Edge::new(1, 3)])
        .unwrap();
    assert_eq!(r.edges(false), proper_edges([(0, 2), (1, 3)]));
    let r_id = containment_id(&tr_b2, &r);
    let image = pushforward(&f, &r).unwrap();
    assert!(tr_b2.leq(r_id, generated_map.apply(tr_c2.bottom())));
    assert_eq!(image.edges(false), proper_edges([(0, 1)]));
    assert!(!tr_c2.leq(containment_id(&tr_c2, &image), tr_c2.bottom()));
}
