#![cfg(feature = "groups")]

use hccr::g_lattice::{
    GLattice, RelationOrbit, RelationOrbitLabel, RelationTransporter, SubgroupGLattice,
};
use hccr::group_theory::GapSubgroup;
use hccr::lattice::Lattice;
use hccr::poset::{Edge, EdgeSet, Poset};
use std::error::Error;
use std::sync::Arc;

#[test]
fn group_actions_produce_the_correct_relation_orbits_and_transfer_systems()
-> Result<(), Box<dyn Error>> {
    let diamond = diamond_lattice();
    let group = gap_sys::eval("Group((1,2));")?;

    let from_generators =
        GLattice::from_generator_images(Arc::clone(&diamond), &group, vec![vec![0, 2, 1, 3]])?;

    let homomorphism =
        gap_sys::eval("GroupHomomorphismByImages(Group((1,2)), Group((2,3)), [(1,2)], [(2,3)]);")?;
    let from_gap = GLattice::from_gap_homomorphism(Arc::clone(&diamond), &group, &homomorphism)?;

    let expected_relations = vec![
        Edge::new(0, 0),
        Edge::new(0, 1),
        Edge::new(0, 2),
        Edge::new(0, 3),
        Edge::new(1, 1),
        Edge::new(1, 3),
        Edge::new(2, 2),
        Edge::new(2, 3),
        Edge::new(3, 3),
    ];
    let expected_orbits = vec![
        vec![0],
        vec![1, 2],
        vec![3],
        vec![4, 6],
        vec![5, 7],
        vec![8],
    ];

    assert_eq!(from_generators.relations(), expected_relations);
    assert_eq!(orbit_ids(&from_generators), expected_orbits);
    assert_eq!(from_gap.relations(), expected_relations);
    assert_eq!(orbit_ids(&from_gap), expected_orbits);
    assert_eq!(
        from_gap.element_generator_permutations(),
        &[vec![0, 2, 1, 3]]
    );

    let generated = from_generators
        .transfer_universe()
        .generated_by([Edge::new(0, 3)])?;
    assert_eq!(
        generated.relations(false),
        EdgeSet::from([Edge::new(0, 1), Edge::new(0, 2), Edge::new(0, 3)])
    );

    let fixed = from_generators
        .relation_orbit(Edge::new(0, 0))
        .expect("fixed identity relation should have an orbit");
    assert_eq!(order(fixed.stabilizer())?, 2);

    let swapped = from_generators
        .relation_orbit(Edge::new(0, 1))
        .expect("swapped relation should have an orbit");
    assert_eq!(swapped.canonical_representative(), Edge::new(0, 1));
    assert_eq!(swapped.relation_ids(), &[1, 2]);
    assert_eq!(order(swapped.stabilizer())?, 1);

    assert_transfer_context_quotients_by_non_identity_relation_orbits(&from_generators);
    assert_transfer_system_containment_lattice_uses_orbit_inclusion(&from_generators)?;

    for orbit in from_generators.relation_orbits() {
        for transporter in orbit.transporters() {
            assert_transporter(&from_generators, orbit, transporter)?;
        }
    }

    check_subgroup_lattice_constructor_uses_conjugation_action()?;
    Ok(())
}

fn assert_transfer_context_quotients_by_non_identity_relation_orbits(
    g_lattice: &GLattice<&'static str>,
) {
    let context = g_lattice.transfer_context();
    let expected_labels = vec![
        RelationOrbitLabel::new(1, 1, Edge::new(0, 1)),
        RelationOrbitLabel::new(2, 3, Edge::new(0, 3)),
        RelationOrbitLabel::new(4, 5, Edge::new(1, 3)),
    ];
    assert_eq!(context.objects, expected_labels);
    assert_eq!(context.attributes, expected_labels);
    assert_eq!(
        g_lattice.non_identity_relation_orbit_labels(),
        expected_labels
    );
    assert!(context.attributes.windows(2).all(|labels| {
        let left = labels[0].canonical_representative();
        let right = labels[1].canonical_representative();
        (left.to, left.from) <= (right.to, right.from)
    }));

    let lower_middle = expected_labels[0];
    let bottom_top = expected_labels[1];
    let middle_top = expected_labels[2];

    assert!(!context.get_relation(&lower_middle, &lower_middle));
    assert!(context.get_relation(&lower_middle, &bottom_top));
    assert!(context.get_relation(&lower_middle, &middle_top));

    assert!(!context.get_relation(&bottom_top, &lower_middle));
    assert!(!context.get_relation(&bottom_top, &bottom_top));
    assert!(context.get_relation(&bottom_top, &middle_top));

    assert!(!context.get_relation(&middle_top, &lower_middle));
    assert!(!context.get_relation(&middle_top, &bottom_top));
    assert!(!context.get_relation(&middle_top, &middle_top));
}

fn assert_transfer_system_containment_lattice_uses_orbit_inclusion(
    g_lattice: &GLattice<&'static str>,
) -> Result<(), Box<dyn Error>> {
    let expected_labels = g_lattice.non_identity_relation_orbit_labels();
    let universe = g_lattice.transfer_universe();
    assert_eq!(universe.relation_orbit_labels(), expected_labels);

    let containment = g_lattice.transfer_systems_containment()?;
    assert_eq!(containment.size(), 4);
    assert_eq!(containment.as_poset().cover_relations().len(), 3);

    let bottom = containment
        .system(containment.bottom())
        .expect("bottom G-transfer system should exist");
    assert!(bottom.relation_orbit_labels().is_empty());
    assert!(bottom.relations(false).is_empty());
    assert_eq!(bottom.relations(true).len(), g_lattice.lattice().size());

    let top = containment
        .system(containment.top())
        .expect("top G-transfer system should exist");
    assert_eq!(top.relation_orbit_labels(), expected_labels);
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
    for system in containment.systems() {
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
    let group = gap_sys::eval("SymmetricGroup(3);")?;
    let subgroup_lattice = SubgroupGLattice::new(&group)?;
    let g_lattice = subgroup_lattice.g_lattice();

    assert_eq!(subgroup_lattice.subgroups().len(), 6);
    assert_eq!(g_lattice.lattice().size(), 6);
    assert_eq!(
        g_lattice.lattice().elements(),
        &[
            GapSubgroup::new(0, 0),
            GapSubgroup::new(1, 0),
            GapSubgroup::new(1, 1),
            GapSubgroup::new(1, 2),
            GapSubgroup::new(2, 0),
            GapSubgroup::new(3, 0),
        ]
    );

    assert_eq!(g_lattice.lattice().bottom(), 0);
    assert_eq!(g_lattice.lattice().top(), 5);
    assert_eq!(g_lattice.relations().len(), 15);
    assert!((0..6).all(|id| g_lattice.lattice().leq(0, id)));
    assert!((0..6).all(|id| g_lattice.lattice().leq(id, 5)));
    assert!(!g_lattice.lattice().leq(1, 2));
    assert!(!g_lattice.lattice().leq(1, 4));

    assert_eq!(
        g_lattice.element_generator_permutations(),
        &[vec![0, 3, 1, 2, 4, 5], vec![0, 3, 2, 1, 4, 5]]
    );

    let c2_identity_orbit = g_lattice
        .relation_orbit(Edge::new(1, 1))
        .expect("C2 identity relation should have an orbit");
    assert_eq!(
        c2_identity_orbit.relations(),
        &[Edge::new(1, 1), Edge::new(2, 2), Edge::new(3, 3)]
    );
    assert_eq!(order(c2_identity_orbit.stabilizer())?, 2);

    let c3_identity_orbit = g_lattice
        .relation_orbit(Edge::new(4, 4))
        .expect("normal C3 identity relation should have an orbit");
    assert_eq!(c3_identity_orbit.relations(), &[Edge::new(4, 4)]);
    assert_eq!(order(c3_identity_orbit.stabilizer())?, 6);

    let subgroup_transfer_lattice = subgroup_lattice.transfer_systems_containment()?;
    assert_eq!(
        subgroup_transfer_lattice.size(),
        g_lattice.transfer_systems_containment()?.size()
    );
    Ok(())
}

fn assert_transporter<A>(
    g_lattice: &GLattice<A>,
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

fn orbit_ids<A>(g_lattice: &GLattice<A>) -> Vec<Vec<usize>> {
    g_lattice
        .relation_orbits()
        .iter()
        .map(|orbit| orbit.relation_ids().to_vec())
        .collect()
}

fn diamond_lattice() -> Arc<Lattice<&'static str>> {
    Arc::new(
        Lattice::new(
            Poset::from_edges(
                vec!["0", "a", "b", "1"],
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
    )
}
