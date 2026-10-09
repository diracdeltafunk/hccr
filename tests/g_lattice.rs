#![cfg(feature = "groups")]

use hccr::g_lattice::{GLattice, RelationOrbit, RelationTransporter, SubgroupGLattice};
use hccr::lattice::Lattice;
use hccr::poset::{Edge, EdgeSet};
use std::collections::BTreeSet;
use std::error::Error;

#[test]
fn group_actions_produce_the_correct_relation_orbits_and_transfer_systems()
-> Result<(), Box<dyn Error>> {
    let diamond = diamond_lattice();
    let group = gap_sys::eval("Group((1,2));")?;

    let from_generators =
        GLattice::from_generator_images(&diamond, &group, vec![vec![0, 2, 1, 3]])?;

    let homomorphism =
        gap_sys::eval("GroupHomomorphismByImages(Group((1,2)), Group((2,3)), [(1,2)], [(2,3)]);")?;
    let from_gap = GLattice::from_gap_homomorphism(&diamond, &group, &homomorphism)?;

    // The swap of the two atoms a, b acts on the nine relations of the
    // diamond with these six orbits, however the action is presented.
    let e = |x: &str, y: &str| diamond.edge(x, y).unwrap();
    let expected_orbits = BTreeSet::from([
        BTreeSet::from([e("0", "0")]),
        BTreeSet::from([e("0", "a"), e("0", "b")]),
        BTreeSet::from([e("0", "1")]),
        BTreeSet::from([e("a", "a"), e("b", "b")]),
        BTreeSet::from([e("a", "1"), e("b", "1")]),
        BTreeSet::from([e("1", "1")]),
    ]);
    for g_lattice in [&from_generators, &from_gap] {
        assert_eq!(
            g_lattice.relations().iter().copied().collect::<EdgeSet>(),
            diamond.all_relations_iter().collect::<EdgeSet>()
        );
        assert_eq!(orbits(g_lattice), expected_orbits);
    }

    let generated = from_generators.transfer_system_generated_by([e("0", "1")])?;
    assert_eq!(
        generated.relations(false),
        EdgeSet::from([e("0", "a"), e("0", "b"), e("0", "1")])
    );

    let fixed = from_generators
        .relation_orbit(e("0", "0"))
        .expect("fixed identity relation should have an orbit");
    assert_eq!(order(fixed.stabilizer())?, 2);

    let swapped = from_generators
        .relation_orbit(e("0", "a"))
        .expect("swapped relation should have an orbit");
    assert_eq!(order(swapped.stabilizer())?, 1);

    assert_transfer_system_containment_lattice_uses_orbit_inclusion(&from_generators)?;

    for orbit in from_generators.relation_orbits() {
        for transporter in orbit.transporters() {
            assert_transporter(&from_generators, orbit, transporter)?;
        }
    }

    check_subgroup_lattice_constructor_uses_conjugation_action()?;
    Ok(())
}

fn assert_transfer_system_containment_lattice_uses_orbit_inclusion(
    g_lattice: &GLattice,
) -> Result<(), Box<dyn Error>> {
    let expected_labels = g_lattice.non_identity_relation_orbit_labels();

    let containment = g_lattice.transfer_systems();
    assert_eq!(containment.size(), 4);
    assert_eq!(containment.cover_relations().len(), 3);

    let bottom = containment.system(containment.bottom());
    assert_eq!(bottom, &g_lattice.trivial_transfer_system());
    assert!(bottom.relation_orbit_labels().is_empty());
    assert!(bottom.relations(false).is_empty());
    assert_eq!(bottom.relations(true).len(), g_lattice.lattice().size());

    let top = containment.system(containment.top());
    assert_eq!(top, &g_lattice.complete_transfer_system());
    assert_eq!(
        top.relation_orbit_labels()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        expected_labels.into_iter().collect::<BTreeSet<_>>()
    );
    assert_eq!(top.relations(false).len(), 5);
    for relation in [
        Edge::new(0, 1),
        Edge::new(0, 2),
        Edge::new(0, 3),
        Edge::new(1, 3),
        Edge::new(2, 3),
    ] {
        assert!(top.relations(false).contains(&relation));
        assert!(top.contains_relation(relation));
    }
    // Forgetting equivariance preserves exactly the underlying relations.
    for system in &containment {
        let ordinary = system.underlying_transfer_system();
        assert_eq!(ordinary.edges(false), system.relations(false));
        for relation in ordinary.edges(true) {
            assert_eq!(
                ordinary.contains_relation(relation),
                system.contains_relation(relation)
            );
        }
    }

    Ok(())
}

fn check_subgroup_lattice_constructor_uses_conjugation_action() -> Result<(), Box<dyn Error>> {
    let subgroup_lattice = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
    let g_lattice = subgroup_lattice.g_lattice();

    // Sub(S_3): the trivial group, three conjugate subgroups of order 2, the
    // normal subgroup of order 3, and S_3 itself, ordered by inclusion.
    let orders = subgroup_lattice
        .subgroups()
        .iter()
        .map(subgroup_order)
        .collect::<Result<Vec<_>, _>>()?;
    let mut sorted_orders = orders.clone();
    sorted_orders.sort_unstable();
    assert_eq!(sorted_orders, vec![1, 2, 2, 2, 3, 6]);
    assert_eq!(orders[g_lattice.bottom()], 1);
    assert_eq!(orders[g_lattice.top()], 6);
    assert_eq!(g_lattice.relations().len(), 15);

    let of_order = |n: usize| {
        g_lattice
            .ids()
            .filter(|&id| orders[id] == n)
            .collect::<Vec<_>>()
    };
    let involutions = of_order(2);
    let rotations = of_order(3)[0];
    for &h in &involutions {
        assert!(!g_lattice.leq(h, rotations) && !g_lattice.leq(rotations, h));
        for &k in &involutions {
            assert_eq!(g_lattice.leq(h, k), h == k);
        }
    }

    // Conjugation permutes the three subgroups of order 2 transitively, each
    // with stabilizer (its own normalizer) of order 2, and fixes the normal
    // subgroup of order 3, whose stabilizer is all of S_3.
    let c2_identity_orbit = g_lattice
        .relation_orbit(Edge::new(involutions[0], involutions[0]))
        .expect("C2 identity relation should have an orbit");
    assert_eq!(
        c2_identity_orbit
            .relations()
            .iter()
            .copied()
            .collect::<EdgeSet>(),
        involutions.iter().map(|&h| Edge::new(h, h)).collect()
    );
    assert_eq!(order(c2_identity_orbit.stabilizer())?, 2);

    let c3_identity_orbit = g_lattice
        .relation_orbit(Edge::new(rotations, rotations))
        .expect("normal C3 identity relation should have an orbit");
    assert_eq!(
        c3_identity_orbit.relations(),
        &[Edge::new(rotations, rotations)]
    );
    assert_eq!(order(c3_identity_orbit.stabilizer())?, 6);

    assert_eq!(subgroup_lattice.transfer_systems().size(), 9);
    Ok(())
}

fn assert_transporter(
    g_lattice: &GLattice,
    orbit: &RelationOrbit,
    transporter: &RelationTransporter,
) -> Result<(), Box<dyn Error>> {
    let gap = gap_sys::global()?;
    let image = gap.call_global(
        "Image",
        &[
            g_lattice.relation_action_homomorphism(),
            transporter.group_element(),
        ],
    )?;
    let permutation = gap.permutation_images_zero_based(&image, g_lattice.relations().len())?;
    assert_eq!(
        permutation[orbit.canonical_relation_id()],
        transporter.relation_id()
    );
    Ok(())
}

fn order(element: &gap_sys::GapValue) -> Result<usize, Box<dyn Error>> {
    let gap = gap_sys::global()?;
    let order = gap.call_global("Order", &[element])?;
    Ok(gap.to_usize(&order)?)
}

fn subgroup_order(subgroup: &gap_sys::GapValue) -> Result<usize, Box<dyn Error>> {
    let gap = gap_sys::global()?;
    let size = gap.call_global("Size", &[subgroup])?;
    Ok(gap.to_usize(&size)?)
}

fn orbits(g_lattice: &GLattice) -> BTreeSet<BTreeSet<Edge>> {
    g_lattice
        .relation_orbits()
        .iter()
        .map(|orbit| orbit.relations().iter().copied().collect())
        .collect()
}

fn diamond_lattice() -> Lattice {
    Lattice::from_covers(
        ["0", "a", "b", "1"],
        [("0", "a"), ("0", "b"), ("a", "1"), ("b", "1")],
    )
    .unwrap()
}
