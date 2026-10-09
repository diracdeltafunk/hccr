use hccr::lattice::Lattice;
use hccr::poset::{Edge, Poset};

#[test]
fn counting_agrees_with_enumeration_on_small_lattices() {
    let examples = [
        Poset::from_edges(vec![0], [] as [Edge; 0]).expect("the one-point poset is valid"),
        Poset::from_edges(vec![0, 1, 2], [Edge::new(0, 1), Edge::new(1, 2)])
            .expect("the three-element chain is valid"),
        Poset::from_edges(
            vec![0, 1, 2, 3],
            [
                Edge::new(0, 1),
                Edge::new(0, 2),
                Edge::new(1, 3),
                Edge::new(2, 3),
            ],
        )
        .expect("the diamond is valid"),
    ];

    for poset in examples {
        let lattice = Lattice::new(poset).expect("example should be a lattice");
        assert_eq!(
            lattice.transfer_system_count(),
            lattice.transfer_systems().size()
        );
    }
}

#[test]
fn horizontal_joins_of_chains_have_the_expected_counts() {
    let c3 = Lattice::chain(3);
    let fusion = Lattice::horizontal_join([&c3, &c3, &c3]).unwrap();
    assert_eq!(fusion.size(), 8);
    assert!(fusion.is_fusion_of_total_orders());
    assert_eq!(fusion.transfer_system_count(), 298);
    assert_eq!(fusion.transfer_systems().size(), 298);
}

#[cfg(feature = "groups")]
#[test]
fn equivariant_counting_agrees_with_enumeration() -> Result<(), Box<dyn std::error::Error>> {
    use hccr::g_lattice::SubgroupGLattice;

    let lattice = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
    assert_eq!(lattice.transfer_system_count(), 9);
    assert_eq!(lattice.transfer_systems().size(), 9);
    Ok(())
}
