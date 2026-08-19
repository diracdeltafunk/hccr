#![cfg(feature = "groups")]
// GAP values are deliberately retained inside the crate's Arc-backed
// mathematical presentations even though the GAP runtime is single-threaded.
#![allow(clippy::arc_with_non_send_sync)]

use gap_sys::GapValue;
use hccr::g_lattice::{GTransferLattice, GTransferSystem, SubgroupGLattice};
use hccr::g_transfer_morphism::{
    GTransferMapError, coinduction, fixed_points, image_pullback, image_pullback_containment_map,
    image_pushforward, image_pushforward_containment_map, induction, inflation,
    preimage_pullback_containment_map, preimage_pushforward_containment_map, pullback,
    pullback_containment_map, pushforward, pushforward_containment_map, restriction,
};
use hccr::poset::Edge;
use hccr::subgroup_morphism::SubgroupMaps;
use hccr::transfer_morphism::{pullback as ordinary_pullback, pushforward as ordinary_pushforward};
use std::error::Error;
use std::sync::Arc;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn subgroup_maps(values: GapValue) -> TestResult<SubgroupMaps> {
    let (domain_group, codomain_group, homomorphism) = {
        let gap = gap_sys::global()?;
        (
            gap.list_get(&values, 0)?,
            gap.list_get(&values, 1)?,
            gap.list_get(&values, 2)?,
        )
    };
    let domain = Arc::new(SubgroupGLattice::new(&domain_group)?);
    let codomain = Arc::new(SubgroupGLattice::new(&codomain_group)?);
    Ok(SubgroupMaps::new(&homomorphism, domain, codomain)?)
}

fn inclusion_c2_into_s3() -> TestResult<SubgroupMaps> {
    subgroup_maps(gap_sys::eval(
        "(function() local g, h, f;
            h := SymmetricGroup(3);
            g := Subgroup(h, [(1,2)]);
            f := GroupHomomorphismByFunction(g, h, x -> x);
            return [g, h, f];
        end)();",
    )?)
}

fn quotient_s3_to_c2() -> TestResult<SubgroupMaps> {
    subgroup_maps(gap_sys::eval(
        "(function() local g, n, h, f;
            g := SymmetricGroup(3);
            n := DerivedSubgroup(g);
            f := NaturalHomomorphismByNormalSubgroup(g, n);
            h := Range(f);
            return [g, h, f];
        end)();",
    )?)
}

fn collapse_c2_to_one() -> TestResult<SubgroupMaps> {
    subgroup_maps(gap_sys::eval(
        "(function() local g, h, f;
            g := Group((1,2));
            f := NaturalHomomorphismByNormalSubgroup(g, g);
            h := Range(f);
            return [g, h, f];
        end)();",
    )?)
}

fn identity_maps(group: &GapValue) -> TestResult<SubgroupMaps> {
    let homomorphism = {
        let gap = gap_sys::global()?;
        gap.call_global("IdentityMapping", &[group])?
    };
    let lattice = Arc::new(SubgroupGLattice::new(group)?);
    Ok(SubgroupMaps::new(
        &homomorphism,
        Arc::clone(&lattice),
        lattice,
    )?)
}

fn contained<A>(lower: &GTransferSystem<A>, upper: &GTransferSystem<A>) -> bool {
    assert!(Arc::ptr_eq(lower.universe(), upper.universe()));
    lower
        .raw()
        .orbit_arrows()
        .iter_ones()
        .all(|orbit| upper.raw().orbit_arrows()[orbit])
}

fn containment_id<A>(order: &GTransferLattice<A>, system: &GTransferSystem<A>) -> usize {
    order
        .raw_lattice()
        .elements()
        .iter()
        .position(|raw| raw == system.raw())
        .expect("the G-transfer system should occur in its containment lattice")
}

fn assert_rubin_adjunctions_and_batch_maps(maps: &SubgroupMaps) -> TestResult {
    let domain = maps.domain().transfer_systems_containment()?;
    let codomain = maps.codomain().transfer_systems_containment()?;
    let domain_systems = domain.systems().collect::<Vec<_>>();
    let codomain_systems = codomain.systems().collect::<Vec<_>>();

    let image_push = pushforward_containment_map(maps.image_map(), &domain, &codomain)?;
    let image_pull = pullback_containment_map(maps.image_map(), &codomain, &domain)?;
    let preimage_push = pushforward_containment_map(maps.preimage_map(), &codomain, &domain)?;
    let preimage_pull = pullback_containment_map(maps.preimage_map(), &domain, &codomain)?;
    let named_image_push = image_pushforward_containment_map(maps, &domain, &codomain)?;
    let named_image_pull = image_pullback_containment_map(maps, &codomain, &domain)?;
    let named_preimage_push = preimage_pushforward_containment_map(maps, &codomain, &domain)?;
    let named_preimage_pull = preimage_pullback_containment_map(maps, &domain, &codomain)?;

    for (domain_id, source) in domain_systems.iter().enumerate() {
        let pointwise = pushforward(maps.image_map(), source, codomain.universe())?;
        assert_eq!(
            image_push.apply(domain_id),
            Some(containment_id(&codomain, &pointwise))
        );
        assert_eq!(
            named_image_push.apply(domain_id),
            image_push.apply(domain_id)
        );

        let pointwise = pullback(maps.preimage_map(), source, codomain.universe())?;
        assert_eq!(
            preimage_pull.apply(domain_id),
            Some(containment_id(&codomain, &pointwise))
        );
        assert_eq!(
            named_preimage_pull.apply(domain_id),
            preimage_pull.apply(domain_id)
        );
    }

    for (codomain_id, target) in codomain_systems.iter().enumerate() {
        let pointwise = pullback(maps.image_map(), target, domain.universe())?;
        assert_eq!(
            image_pull.apply(codomain_id),
            Some(containment_id(&domain, &pointwise))
        );
        assert_eq!(
            named_image_pull.apply(codomain_id),
            image_pull.apply(codomain_id)
        );

        let pointwise = pushforward(maps.preimage_map(), target, domain.universe())?;
        assert_eq!(
            preimage_push.apply(codomain_id),
            Some(containment_id(&domain, &pointwise))
        );
        assert_eq!(
            named_preimage_push.apply(codomain_id),
            preimage_push.apply(codomain_id)
        );
    }

    for source in &domain_systems {
        for target in &codomain_systems {
            let image = pushforward(maps.image_map(), source, codomain.universe())?;
            let inverse = pullback(maps.image_map(), target, domain.universe())?;
            assert_eq!(
                contained(&image, target),
                contained(source, &inverse),
                "the subgroup-image pushforward and pullback should be adjoint"
            );

            let inverse_image = pushforward(maps.preimage_map(), target, domain.universe())?;
            let right_image = pullback(maps.preimage_map(), source, codomain.universe())?;
            assert_eq!(
                contained(&inverse_image, source),
                contained(target, &right_image),
                "the subgroup-preimage pushforward and pullback should be adjoint"
            );
        }
    }

    Ok(())
}

#[test]
fn group_transfer_morphisms_realize_rubins_four_functors() -> TestResult {
    check_inclusion_orbit_closure_and_core()?;
    check_quotient_inflation_and_fixed_points()?;
    check_noninjective_inverse_distinction()?;
    check_identity_and_composition_laws()?;
    Ok(())
}

fn check_inclusion_orbit_closure_and_core() -> TestResult {
    let maps = inclusion_c2_into_s3()?;
    assert!(maps.is_injective());
    assert!(!maps.is_surjective());

    let domain = maps.domain().transfer_systems_containment()?;
    let codomain = maps.codomain().transfer_systems_containment()?;
    let source_top = domain.system(domain.top()).expect("top should exist");
    let source_bottom = domain.system(domain.bottom()).expect("bottom should exist");

    let s3_bottom = maps.codomain().lattice().bottom();
    let included_c2 = maps
        .image_map()
        .apply(maps.domain().lattice().top())
        .expect("the image map should be total");
    let c2_orbit = maps
        .codomain()
        .g_lattice()
        .relation_orbit(Edge::new(s3_bottom, included_c2))
        .expect("the three bottom-to-C2 relations should form an orbit");
    assert_eq!(c2_orbit.relations().len(), 3);

    let induced = pushforward(maps.image_map(), &source_top, codomain.universe())?;
    assert_eq!(induction(&maps, &source_top, codomain.universe())?, induced);
    assert_eq!(
        image_pushforward(&maps, &source_top, codomain.universe())?,
        induced
    );
    assert!(
        c2_orbit
            .relations()
            .iter()
            .all(|&edge| induced.contains_relation(edge))
    );

    let ordinary_induced = ordinary_pushforward(
        maps.image_map(),
        &source_top.underlying_transfer_system(),
        codomain.universe().underlying_transfer_universe(),
    )?;
    assert_eq!(
        c2_orbit
            .relations()
            .iter()
            .filter(|&&edge| ordinary_induced.contains_relation(edge))
            .count(),
        1,
        "ordinary pushforward should remember the chosen C2 and miss its H-conjugates"
    );

    let coinduced = pullback(maps.preimage_map(), &source_bottom, codomain.universe())?;
    assert_eq!(
        coinduction(&maps, &source_bottom, codomain.universe())?,
        coinduced
    );
    assert!(
        c2_orbit
            .relations()
            .iter()
            .all(|&edge| !coinduced.contains_relation(edge))
    );

    let ordinary_coinduced = ordinary_pullback(
        maps.preimage_map(),
        &source_bottom.underlying_transfer_system(),
        codomain.universe().underlying_transfer_universe(),
    )?;
    assert_eq!(
        c2_orbit
            .relations()
            .iter()
            .filter(|&&edge| ordinary_coinduced.contains_relation(edge))
            .count(),
        2,
        "ordinary pullback should contain only the two C2s disjoint from the chosen image"
    );

    for target in codomain.systems() {
        assert_eq!(
            restriction(&maps, &target, domain.universe())?,
            image_pullback(&maps, &target, domain.universe())?,
            "the two Rubin restriction constructions agree for an injection"
        );
    }
    let target_bottom = codomain.system(codomain.bottom()).unwrap();
    assert_eq!(
        inflation(&maps, &target_bottom, domain.universe()).unwrap_err(),
        GTransferMapError::NotSurjective
    );
    assert_eq!(
        fixed_points(&maps, &source_top, codomain.universe()).unwrap_err(),
        GTransferMapError::NotSurjective
    );

    assert_rubin_adjunctions_and_batch_maps(&maps)
}

fn check_quotient_inflation_and_fixed_points() -> TestResult {
    let maps = quotient_s3_to_c2()?;
    assert!(!maps.is_injective());
    assert!(maps.is_surjective());

    let domain = maps.domain().transfer_systems_containment()?;
    let codomain = maps.codomain().transfer_systems_containment()?;
    let kernel = maps
        .preimage_map()
        .apply(maps.codomain().lattice().bottom())
        .expect("subgroup preimage should be total");
    let source_lattice = maps.domain().lattice();

    for quotient_system in codomain.systems() {
        let inflated = pushforward(maps.preimage_map(), &quotient_system, domain.universe())?;
        assert_eq!(
            inflation(&maps, &quotient_system, domain.universe())?,
            inflated
        );

        for edge in source_lattice.as_poset().all_relations_iter() {
            let quotient_edge = Edge::new(
                maps.image_map().apply(edge.from).unwrap(),
                maps.image_map().apply(edge.to).unwrap(),
            );
            let kernel_saturation = source_lattice.join_id(edge.from, kernel);
            let is_restriction_of_saturated_arrow =
                edge.from == source_lattice.meet_id(kernel_saturation, edge.to);
            let expected = quotient_system.contains_relation(quotient_edge)
                && is_restriction_of_saturated_arrow;
            assert_eq!(
                inflated.contains_relation(edge),
                expected,
                "inflation formula failed on {} <= {}",
                edge.from,
                edge.to
            );
        }

        let recovered = pullback(maps.preimage_map(), &inflated, codomain.universe())?;
        assert_eq!(recovered, quotient_system);
    }

    for source_system in domain.systems() {
        let fixed = pullback(maps.preimage_map(), &source_system, codomain.universe())?;
        assert_eq!(
            fixed_points(&maps, &source_system, codomain.universe())?,
            fixed
        );
        for edge in maps.codomain().lattice().as_poset().all_relations_iter() {
            let lifted = Edge::new(
                maps.preimage_map().apply(edge.from).unwrap(),
                maps.preimage_map().apply(edge.to).unwrap(),
            );
            assert_eq!(
                fixed.contains_relation(edge),
                source_system.contains_relation(lifted),
                "fixed-points formula failed on {} <= {}",
                edge.from,
                edge.to
            );
        }
    }

    assert_rubin_adjunctions_and_batch_maps(&maps)
}

fn check_noninjective_inverse_distinction() -> TestResult {
    let maps = collapse_c2_to_one()?;
    assert!(!maps.is_injective());
    assert!(maps.is_surjective());

    let domain = maps.domain().transfer_systems_containment()?;
    let codomain = maps.codomain().transfer_systems_containment()?;
    assert_eq!(codomain.size(), 1);
    let unique = codomain
        .system(codomain.bottom())
        .expect("the trivial group has one transfer system");

    let rubin_restriction = pushforward(maps.preimage_map(), &unique, domain.universe())?;
    assert_eq!(
        restriction(&maps, &unique, domain.universe())?,
        rubin_restriction
    );
    let other_inverse = pullback(maps.image_map(), &unique, domain.universe())?;
    assert_eq!(
        rubin_restriction.raw(),
        domain.raw_lattice().element(domain.bottom()).unwrap()
    );
    assert_eq!(
        other_inverse.raw(),
        domain.raw_lattice().element(domain.top()).unwrap()
    );
    assert_ne!(rubin_restriction, other_inverse);
    assert_eq!(
        induction(
            &maps,
            &domain.system(domain.bottom()).unwrap(),
            codomain.universe()
        )
        .unwrap_err(),
        GTransferMapError::NotInjective
    );

    assert_rubin_adjunctions_and_batch_maps(&maps)
}

fn check_identity_and_composition_laws() -> TestResult {
    let s3 = gap_sys::eval("SymmetricGroup(3);")?;
    let identity = identity_maps(&s3)?;
    let systems = identity.domain().transfer_systems_containment()?;
    for system in systems.systems() {
        assert_eq!(
            pushforward(identity.image_map(), &system, systems.universe())?,
            system
        );
        assert_eq!(
            pullback(identity.image_map(), &system, systems.universe())?,
            system
        );
        assert_eq!(
            pushforward(identity.preimage_map(), &system, systems.universe())?,
            system
        );
        assert_eq!(
            pullback(identity.preimage_map(), &system, systems.universe())?,
            system
        );
    }

    let composition_data = gap_sys::eval(
        "(function() local g, h, k, f, q, qf;
            h := SymmetricGroup(3);
            g := Subgroup(h, [(1,2)]);
            f := GroupHomomorphismByFunction(g, h, x -> x);
            q := NaturalHomomorphismByNormalSubgroup(h, DerivedSubgroup(h));
            k := Range(q);
            qf := CompositionMapping(q, f);
            return [g, h, k, f, q, qf];
        end)();",
    )?;
    let (c2, s3, sign_range, inclusion, sign, composite) = {
        let gap = gap_sys::global()?;
        (
            gap.list_get(&composition_data, 0)?,
            gap.list_get(&composition_data, 1)?,
            gap.list_get(&composition_data, 2)?,
            gap.list_get(&composition_data, 3)?,
            gap.list_get(&composition_data, 4)?,
            gap.list_get(&composition_data, 5)?,
        )
    };
    let c2_lattice = Arc::new(SubgroupGLattice::new(&c2)?);
    let s3_lattice = Arc::new(SubgroupGLattice::new(&s3)?);
    let sign_lattice = Arc::new(SubgroupGLattice::new(&sign_range)?);
    let first = SubgroupMaps::new(&inclusion, Arc::clone(&c2_lattice), Arc::clone(&s3_lattice))?;
    let second = SubgroupMaps::new(&sign, Arc::clone(&s3_lattice), Arc::clone(&sign_lattice))?;
    let composed = SubgroupMaps::new(
        &composite,
        Arc::clone(&c2_lattice),
        Arc::clone(&sign_lattice),
    )?;
    assert!(first.is_injective());
    assert!(second.is_surjective());
    assert!(composed.is_injective());
    assert!(composed.is_surjective());

    let c2_systems = first.domain().transfer_systems_containment()?;
    let s3_systems = first.codomain().transfer_systems_containment()?;
    let sign_systems = second.codomain().transfer_systems_containment()?;

    for source in c2_systems.systems() {
        let first_image = pushforward(first.image_map(), &source, s3_systems.universe())?;
        let iterated_image =
            pushforward(second.image_map(), &first_image, sign_systems.universe())?;
        let direct_image = pushforward(composed.image_map(), &source, sign_systems.universe())?;
        assert_eq!(iterated_image, direct_image);

        let first_right_image = pullback(first.preimage_map(), &source, s3_systems.universe())?;
        let iterated_right_image = pullback(
            second.preimage_map(),
            &first_right_image,
            sign_systems.universe(),
        )?;
        let direct_right_image =
            pullback(composed.preimage_map(), &source, sign_systems.universe())?;
        assert_eq!(iterated_right_image, direct_right_image);
    }

    for target in sign_systems.systems() {
        let second_left_inverse =
            pushforward(second.preimage_map(), &target, s3_systems.universe())?;
        let iterated_left_inverse = pushforward(
            first.preimage_map(),
            &second_left_inverse,
            c2_systems.universe(),
        )?;
        let direct_left_inverse =
            pushforward(composed.preimage_map(), &target, c2_systems.universe())?;
        assert_eq!(iterated_left_inverse, direct_left_inverse);

        let second_right_inverse = pullback(second.image_map(), &target, s3_systems.universe())?;
        let iterated_right_inverse = pullback(
            first.image_map(),
            &second_right_inverse,
            c2_systems.universe(),
        )?;
        let direct_right_inverse = pullback(composed.image_map(), &target, c2_systems.universe())?;
        assert_eq!(iterated_right_inverse, direct_right_inverse);
    }

    Ok(())
}
