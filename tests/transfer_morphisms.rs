use hccr::lattice::Lattice;
use hccr::morphism::LatticeMap;
use hccr::poset::{Edge, EdgeSet};
use hccr::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use hccr::transfer_morphism::{
    CompositionMapError, TransferMapError, pullback, pullback_containment_map, pushforward,
    pushforward_containment_map, try_pullback_composition_map, try_pushforward_composition_map,
};
use std::sync::Arc;

fn chain(top: usize) -> Arc<Lattice<usize>> {
    Arc::new(Lattice::chain(top).expect("a finite chain should be a lattice"))
}

fn boolean_two() -> Arc<Lattice<usize>> {
    Arc::new(Lattice::boolean(2).expect("B2 should be a lattice"))
}

fn proper_edges(pairs: impl IntoIterator<Item = (usize, usize)>) -> EdgeSet {
    pairs.into_iter().map(Edge::from).collect()
}

fn containment_id<A>(order: &TransferLattice<A>, system: &TransferSystem<A>) -> usize {
    order
        .raw_lattice()
        .elements()
        .iter()
        .position(|raw| raw == system.raw())
        .expect("the transfer system should occur in its containment lattice")
}

fn composition_id<A>(order: &TransferPoset<A>, system: &TransferSystem<A>) -> usize {
    order
        .raw_poset()
        .elements()
        .iter()
        .position(|raw| raw == system.raw())
        .expect("the transfer system should occur in its composition-closed order")
}

fn assert_batch_pushforward_agrees<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferLattice<A>,
    codomain: &TransferLattice<B>,
) {
    let map = pushforward_containment_map(homomorphism, domain, codomain).unwrap();
    for (source_id, system) in domain.systems().enumerate() {
        let source_label = map.domain().element(source_id).unwrap();
        assert!(source_label == &system);
        assert!(Arc::ptr_eq(source_label.universe(), domain.universe()));

        let expected = pushforward(homomorphism, &system, codomain.universe()).unwrap();
        let actual_id = map.apply(source_id).unwrap();
        let actual = map.codomain().element(actual_id).unwrap();
        assert!(Arc::ptr_eq(actual.universe(), codomain.universe()));
        assert!(
            actual == &expected,
            "batch pushforward disagrees at source element {source_id}"
        );
    }
}

fn assert_batch_pullback_agrees<A, B>(
    homomorphism: &LatticeMap<A, B>,
    codomain: &TransferLattice<B>,
    domain: &TransferLattice<A>,
) {
    let map = pullback_containment_map(homomorphism, codomain, domain).unwrap();
    for (source_id, system) in codomain.systems().enumerate() {
        let source_label = map.domain().element(source_id).unwrap();
        assert!(source_label == &system);
        assert!(Arc::ptr_eq(source_label.universe(), codomain.universe()));

        let expected = pullback(homomorphism, &system, domain.universe()).unwrap();
        let actual_id = map.apply(source_id).unwrap();
        let actual = map.codomain().element(actual_id).unwrap();
        assert!(Arc::ptr_eq(actual.universe(), domain.universe()));
        assert!(
            actual == &expected,
            "batch pullback disagrees at source element {source_id}"
        );
    }
}

fn assert_composition_pushforward_agrees<A, B>(
    homomorphism: &LatticeMap<A, B>,
    domain: &TransferPoset<A>,
    codomain: &TransferPoset<B>,
) {
    let map = try_pushforward_composition_map(homomorphism, domain, codomain).unwrap();
    for (source_id, system) in domain.systems().enumerate() {
        let expected = pushforward(homomorphism, &system, codomain.universe()).unwrap();
        let expected_id = composition_id(codomain, &expected);
        assert_eq!(map.apply(source_id), Some(expected_id));
        assert!(map.codomain().element(expected_id).unwrap() == &expected);
    }
}

fn assert_composition_pullback_agrees<A, B>(
    homomorphism: &LatticeMap<A, B>,
    codomain: &TransferPoset<B>,
    domain: &TransferPoset<A>,
) {
    let map = try_pullback_composition_map(homomorphism, codomain, domain).unwrap();
    for (source_id, system) in codomain.systems().enumerate() {
        let expected = pullback(homomorphism, &system, domain.universe()).unwrap();
        let expected_id = composition_id(domain, &expected);
        assert_eq!(map.apply(source_id), Some(expected_id));
        assert!(map.codomain().element(expected_id).unwrap() == &expected);
    }
}

#[test]
fn pointwise_maps_close_images_and_handle_collapsed_and_duplicate_arrows() {
    let domain_lattice = chain(2);
    let codomain_lattice = chain(1);
    let quotient = LatticeMap::new(
        Arc::clone(&domain_lattice),
        Arc::clone(&codomain_lattice),
        vec![0, 0, 1],
    )
    .unwrap();
    let domain = Arc::clone(&domain_lattice).transfer_universe();
    let codomain = Arc::clone(&codomain_lattice).transfer_universe();

    // Both 0 -> 2 and 1 -> 2 have image 0 -> 1. The first generator also
    // forces 0 -> 1 in the source by restriction.
    let source_top = domain
        .generated_by([Edge::new(0, 2), Edge::new(1, 2)])
        .unwrap();
    assert_eq!(
        source_top.edges(false),
        proper_edges([(0, 1), (0, 2), (1, 2)])
    );
    let image = pushforward(&quotient, &source_top, &codomain).unwrap();
    assert!(Arc::ptr_eq(image.universe(), &codomain));
    assert_eq!(image.edges(false), proper_edges([(0, 1)]));

    // Pulling back the diagonal includes every arrow collapsed by the map.
    let target_bottom = codomain.generated_by(std::iter::empty::<Edge>()).unwrap();
    let inverse_image = pullback(&quotient, &target_bottom, &domain).unwrap();
    assert!(Arc::ptr_eq(inverse_image.universe(), &domain));
    assert_eq!(inverse_image.edges(false), proper_edges([(0, 1)]));

    let wrong_domain = chain(3).transfer_universe();
    let wrong_source = wrong_domain
        .generated_by(std::iter::empty::<Edge>())
        .unwrap();
    assert_eq!(
        pushforward(&quotient, &wrong_source, &codomain).unwrap_err(),
        TransferMapError::DomainMismatch
    );

    let wrong_codomain = boolean_two().transfer_universe();
    assert_eq!(
        pushforward(&quotient, &source_top, &wrong_codomain).unwrap_err(),
        TransferMapError::CodomainMismatch
    );
}

#[test]
fn maps_to_the_trivial_lattice_handle_empty_coordinates_and_all_collapsed_arrows() {
    let c3 = chain(2);
    let point = chain(0);
    let collapse = LatticeMap::new(Arc::clone(&c3), Arc::clone(&point), vec![0, 0, 0]).unwrap();
    let c3_universe = Arc::clone(&c3).transfer_universe();
    let point_universe = Arc::clone(&point).transfer_universe();

    let source = c3_universe
        .generated_by([Edge::new(0, 2), Edge::new(1, 2)])
        .unwrap();
    let target = point_universe
        .generated_by(std::iter::empty::<Edge>())
        .unwrap();
    assert_eq!(
        pushforward(&collapse, &source, &point_universe).unwrap(),
        target
    );
    assert_eq!(
        pullback(&collapse, &target, &c3_universe)
            .unwrap()
            .edges(false),
        proper_edges([(0, 1), (0, 2), (1, 2)])
    );

    let tr_c3 = c3_universe.containment_lattice().unwrap();
    let tr_point = point_universe.containment_lattice().unwrap();
    let push = pushforward_containment_map(&collapse, &tr_c3, &tr_point).unwrap();
    let pull = pullback_containment_map(&collapse, &tr_point, &tr_c3).unwrap();
    assert!(push.map().iter().all(|&image| image == tr_point.bottom()));
    assert_eq!(pull.apply(tr_point.bottom()), Some(tr_c3.top()));

    let cc_c3 = c3_universe.composition_closed_order().unwrap();
    let cc_point = point_universe.composition_closed_order().unwrap();
    assert!(try_pushforward_composition_map(&collapse, &cc_c3, &cc_point).is_ok());
    assert!(try_pullback_composition_map(&collapse, &cc_point, &cc_c3).is_ok());
}

#[test]
fn containment_maps_are_pointwise_correct_adjoint_and_preserve_the_expected_operations() {
    let domain_lattice = chain(2);
    let codomain_lattice = chain(1);
    let quotient = LatticeMap::new(
        Arc::clone(&domain_lattice),
        Arc::clone(&codomain_lattice),
        vec![0, 0, 1],
    )
    .unwrap();
    let domain_universe = Arc::clone(&domain_lattice).transfer_universe();
    let codomain_universe = Arc::clone(&codomain_lattice).transfer_universe();
    let domain = domain_universe.containment_lattice().unwrap();
    let codomain = codomain_universe.containment_lattice().unwrap();

    assert_batch_pushforward_agrees(&quotient, &domain, &codomain);
    assert_batch_pullback_agrees(&quotient, &codomain, &domain);

    let push = pushforward_containment_map(&quotient, &domain, &codomain).unwrap();
    let pull = pullback_containment_map(&quotient, &codomain, &domain).unwrap();

    for r in 0..domain.size() {
        for s in 0..codomain.size() {
            assert_eq!(
                codomain.as_poset().leq(push.apply(r).unwrap(), s),
                domain.as_poset().leq(r, pull.apply(s).unwrap()),
                "the containment adjunction failed for ({r}, {s})"
            );
        }
    }

    assert_eq!(push.apply(domain.bottom()), Some(codomain.bottom()));
    for left in 0..domain.size() {
        for right in 0..domain.size() {
            assert_eq!(
                push.apply(domain.join_id(left, right)),
                Some(codomain.join_id(push.apply(left).unwrap(), push.apply(right).unwrap()))
            );
        }
    }

    assert_eq!(pull.apply(codomain.top()), Some(domain.top()));
    for left in 0..codomain.size() {
        for right in 0..codomain.size() {
            assert_eq!(
                pull.apply(codomain.meet_id(left, right)),
                Some(domain.meet_id(pull.apply(left).unwrap(), pull.apply(right).unwrap()))
            );
        }
    }
}

#[test]
fn containment_maps_need_not_be_lattice_homomorphisms() {
    // Pullback need not preserve joins: include the endpoints of C2 as the
    // bottom and top of C3.
    let c2 = chain(1);
    let c3 = chain(2);
    let inclusion = LatticeMap::new(Arc::clone(&c2), Arc::clone(&c3), vec![0, 2]).unwrap();
    let c2_universe = Arc::clone(&c2).transfer_universe();
    let c3_universe = Arc::clone(&c3).transfer_universe();
    let tr_c2 = c2_universe.containment_lattice().unwrap();
    let tr_c3 = c3_universe.containment_lattice().unwrap();
    let inverse_image = pullback_containment_map(&inclusion, &tr_c3, &tr_c2).unwrap();
    let left = c3_universe.generated_by([Edge::new(0, 1)]).unwrap();
    let right = c3_universe.generated_by([Edge::new(1, 2)]).unwrap();
    let left_id = containment_id(&tr_c3, &left);
    let right_id = containment_id(&tr_c3, &right);
    let image_of_join = inverse_image
        .apply(tr_c3.join_id(left_id, right_id))
        .unwrap();
    let join_of_images = tr_c2.join_id(
        inverse_image.apply(left_id).unwrap(),
        inverse_image.apply(right_id).unwrap(),
    );
    assert_ne!(image_of_join, join_of_images);

    // A quotient demonstrates both failure of pullback to preserve bottom and
    // failure of pushforward to preserve meets.
    let quotient = LatticeMap::new(Arc::clone(&c3), Arc::clone(&c2), vec![0, 0, 1]).unwrap();
    let direct_image = pushforward_containment_map(&quotient, &tr_c3, &tr_c2).unwrap();
    let inverse_image = pullback_containment_map(&quotient, &tr_c2, &tr_c3).unwrap();
    assert_ne!(inverse_image.apply(tr_c2.bottom()), Some(tr_c3.bottom()));

    let r = c3_universe.generated_by([Edge::new(0, 2)]).unwrap();
    let q = c3_universe.generated_by([Edge::new(1, 2)]).unwrap();
    let r_id = containment_id(&tr_c3, &r);
    let q_id = containment_id(&tr_c3, &q);
    let image_of_meet = direct_image.apply(tr_c3.meet_id(r_id, q_id)).unwrap();
    let meet_of_images = tr_c2.meet_id(
        direct_image.apply(r_id).unwrap(),
        direct_image.apply(q_id).unwrap(),
    );
    assert_ne!(image_of_meet, meet_of_images);
}

#[test]
fn composition_monotonicity_is_checked_independently_in_each_direction() {
    // C3 -> B2 has non-monotone pushforward but monotone pullback.
    let c3 = chain(2);
    let b2 = boolean_two();
    let into_b2 = LatticeMap::new(Arc::clone(&c3), Arc::clone(&b2), vec![0, 1, 3]).unwrap();
    let c3_universe = Arc::clone(&c3).transfer_universe();
    let b2_universe = Arc::clone(&b2).transfer_universe();
    let cc_c3 = c3_universe.composition_closed_order().unwrap();
    let cc_b2 = b2_universe.composition_closed_order().unwrap();

    let r = c3_universe.generated_by([Edge::new(1, 2)]).unwrap();
    let s = c3_universe
        .generated_by([Edge::new(0, 1), Edge::new(1, 2)])
        .unwrap();
    assert!(
        cc_c3
            .raw_poset()
            .leq(composition_id(&cc_c3, &r), composition_id(&cc_c3, &s))
    );
    let image_r = pushforward(&into_b2, &r, &b2_universe).unwrap();
    let image_s = pushforward(&into_b2, &s, &b2_universe).unwrap();
    assert_eq!(image_r.edges(false), proper_edges([(0, 2), (1, 3)]));
    assert_eq!(
        image_s.edges(false),
        proper_edges([(0, 1), (0, 2), (0, 3), (1, 3)])
    );
    assert!(!cc_b2.raw_poset().leq(
        composition_id(&cc_b2, &image_r),
        composition_id(&cc_b2, &image_s)
    ));
    let push_error = try_pushforward_composition_map(&into_b2, &cc_c3, &cc_b2).unwrap_err();
    let CompositionMapError::NotMonotone { failed_square, .. } = push_error else {
        panic!("the pushforward should fail the composition-order check")
    };
    assert_eq!(failed_square, (Edge::new(0, 2), Edge::new(0, 3)));
    assert_composition_pullback_agrees(&into_b2, &cc_b2, &cc_c3);

    // C3 -> C4 has monotone pushforward but non-monotone pullback.
    let c4 = chain(3);
    let into_c4 = LatticeMap::new(Arc::clone(&c3), Arc::clone(&c4), vec![0, 1, 3]).unwrap();
    let c4_universe = Arc::clone(&c4).transfer_universe();
    let cc_c4 = c4_universe.composition_closed_order().unwrap();
    let r = c4_universe.generated_by([Edge::new(0, 2)]).unwrap();
    let s = c4_universe
        .generated_by([Edge::new(0, 3), Edge::new(2, 3)])
        .unwrap();
    assert!(
        cc_c4
            .raw_poset()
            .leq(composition_id(&cc_c4, &r), composition_id(&cc_c4, &s))
    );
    let inverse_r = pullback(&into_c4, &r, &c3_universe).unwrap();
    let inverse_s = pullback(&into_c4, &s, &c3_universe).unwrap();
    assert_eq!(inverse_r.edges(false), proper_edges([(0, 1)]));
    assert_eq!(inverse_s.edges(false), proper_edges([(0, 1), (0, 2)]));
    assert!(!cc_c3.raw_poset().leq(
        composition_id(&cc_c3, &inverse_r),
        composition_id(&cc_c3, &inverse_s)
    ));
    assert_composition_pushforward_agrees(&into_c4, &cc_c3, &cc_c4);
    let pull_error = try_pullback_composition_map(&into_c4, &cc_c4, &cc_c3).unwrap_err();
    let CompositionMapError::NotMonotone { failed_square, .. } = pull_error else {
        panic!("the pullback should fail the composition-order check")
    };
    assert_eq!(failed_square, (Edge::new(0, 1), Edge::new(0, 2)));
}

#[test]
fn pullback_along_a_surjection_embeds_the_composition_closed_order() {
    let c4 = chain(3);
    let c3 = chain(2);
    let quotient = LatticeMap::new(Arc::clone(&c4), Arc::clone(&c3), vec![0, 0, 1, 2]).unwrap();
    let c4_universe = Arc::clone(&c4).transfer_universe();
    let c3_universe = Arc::clone(&c3).transfer_universe();
    let cc_c4 = c4_universe.composition_closed_order().unwrap();
    let cc_c3 = c3_universe.composition_closed_order().unwrap();
    let inverse_image = try_pullback_composition_map(&quotient, &cc_c3, &cc_c4).unwrap();

    for left in 0..cc_c3.size() {
        for right in 0..cc_c3.size() {
            assert_eq!(
                cc_c3.raw_poset().leq(left, right),
                cc_c4.raw_poset().leq(
                    inverse_image.apply(left).unwrap(),
                    inverse_image.apply(right).unwrap()
                )
            );
        }
    }

    // Surjectivity does not force the pushforward to be monotone for this order.
    assert!(matches!(
        try_pushforward_composition_map(&quotient, &cc_c4, &cc_c3),
        Err(CompositionMapError::NotMonotone { .. })
    ));
}

#[test]
fn pointwise_maps_are_functorial_and_identity_maps_work_for_both_orders() {
    let c4 = chain(3);
    let c3 = chain(2);
    let c2 = chain(1);
    let f = LatticeMap::new(Arc::clone(&c4), Arc::clone(&c3), vec![0, 0, 1, 2]).unwrap();
    let g = LatticeMap::new(Arc::clone(&c3), Arc::clone(&c2), vec![0, 0, 1]).unwrap();
    let composite = LatticeMap::new(Arc::clone(&c4), Arc::clone(&c2), vec![0, 0, 0, 1]).unwrap();
    let identity = LatticeMap::new(Arc::clone(&c3), Arc::clone(&c3), vec![0, 1, 2]).unwrap();
    let u4 = Arc::clone(&c4).transfer_universe();
    let u3 = Arc::clone(&c3).transfer_universe();
    let u2 = Arc::clone(&c2).transfer_universe();

    for system in u4.transfer_systems() {
        let sequential = pushforward(&g, &pushforward(&f, &system, &u3).unwrap(), &u2).unwrap();
        assert_eq!(sequential, pushforward(&composite, &system, &u2).unwrap());
    }
    for system in u2.transfer_systems() {
        let sequential = pullback(&f, &pullback(&g, &system, &u3).unwrap(), &u4).unwrap();
        assert_eq!(sequential, pullback(&composite, &system, &u4).unwrap());
    }
    for system in u3.transfer_systems() {
        assert_eq!(pushforward(&identity, &system, &u3).unwrap(), system);
        assert_eq!(pullback(&identity, &system, &u3).unwrap(), system);
    }

    let containment = u3.containment_lattice().unwrap();
    let composition = u3.composition_closed_order().unwrap();
    assert!(pushforward_containment_map(&identity, &containment, &containment).is_ok());
    assert!(pullback_containment_map(&identity, &containment, &containment).is_ok());
    assert!(try_pushforward_composition_map(&identity, &composition, &composition).is_ok());
    assert!(try_pullback_composition_map(&identity, &composition, &composition).is_ok());
}
