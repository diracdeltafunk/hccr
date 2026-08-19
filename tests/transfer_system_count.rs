use hccr::lattice::Lattice;
use hccr::poset::{Edge, Poset};
use std::sync::Arc;

#[test]
fn counting_agrees_with_enumeration_on_small_lattices() {
    let examples = [
        Poset::from_edges(vec![0], []).expect("the one-point poset is valid"),
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
        let lattice = Arc::new(Lattice::new(poset).expect("example should be a lattice"));
        let universe = Arc::clone(&lattice).transfer_universe();
        assert_eq!(
            universe.transfer_system_count(),
            universe.transfer_systems().len()
        );
        assert_eq!(
            Arc::clone(&lattice).transfer_system_count(),
            universe.transfer_system_count()
        );
    }
}

#[cfg(feature = "groups")]
#[test]
fn equivariant_counting_agrees_with_enumeration() -> Result<(), Box<dyn std::error::Error>> {
    use hccr::g_lattice::SubgroupGLattice;

    let group = gap_sys::eval("SymmetricGroup(3);")?;
    let lattice = SubgroupGLattice::new(&group)?;
    let universe = lattice.g_lattice().transfer_universe();
    assert_eq!(
        lattice.transfer_system_count(),
        universe.transfer_system_count()
    );
    assert_eq!(
        universe.transfer_system_count(),
        universe.transfer_systems().len()
    );
    Ok(())
}
