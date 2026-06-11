use hccr::lattice::{Lattice, LatticeError, horizontal_join};
use hccr::morphism::{LatticeMap, MapError, PosetMap};
use hccr::poset::{Edge, Either, Poset, PosetError};
use hccr::tikz::{PosetTikzOptions, TikzLabel, ToTikz, poset_to_tikz_with};
use hccr::transfer_lattice::TransferSystems;
use std::sync::Arc;

fn chain(n: usize) -> Lattice<usize> {
    let edges = (0..n.saturating_sub(1)).map(|i| Edge::new(i, i + 1));
    Lattice::new(Poset::from_edges((0..n).collect(), edges).unwrap()).unwrap()
}

fn diamond() -> Lattice<usize> {
    Lattice::new(
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
    )
    .unwrap()
}

#[test]
fn poset_construction_and_cover_relations_are_validated() {
    let poset = Poset::from_edges(vec!["0", "1", "2"], [Edge::new(0, 1), Edge::new(1, 2)]).unwrap();

    assert!(poset.validate().is_ok());
    assert!(poset.leq(0, 2));
    assert_eq!(poset.bottom(), Some(0));
    assert_eq!(poset.top(), Some(2));

    let covers = poset.cover_relations();
    assert!(covers.contains(&Edge::new(0, 1)));
    assert!(covers.contains(&Edge::new(1, 2)));
    assert!(!covers.contains(&Edge::new(0, 2)));

    let bad = Poset::from_edges(vec![0, 1], [Edge::new(0, 2)]);
    assert!(matches!(bad, Err(PosetError::EdgeOutOfBounds { .. })));

    let cycle = Poset::from_edges(vec![0, 1], [Edge::new(0, 1), Edge::new(1, 0)]);
    assert!(matches!(cycle, Err(PosetError::NotAntisymmetric { .. })));
}

#[test]
fn products_and_disjoint_unions_return_valid_maps() {
    let left = Arc::new(Poset::from_edges(vec!["0", "1"], [Edge::new(0, 1)]).unwrap());
    let right = Arc::new(Poset::from_edges(vec!["a", "b"], [Edge::new(0, 1)]).unwrap());

    let coproduct =
        Poset::<Either<&str, &str>>::disjoint_union(Arc::clone(&left), Arc::clone(&right)).unwrap();
    assert_eq!(coproduct.poset.size(), 4);
    assert_eq!(coproduct.left.map(), &[0, 1]);
    assert_eq!(coproduct.right.map(), &[2, 3]);

    let product = Poset::<(&str, &str)>::product(left, right).unwrap();
    assert_eq!(product.poset.size(), 4);
    assert_eq!(product.left_projection.map(), &[0, 0, 1, 1]);
    assert_eq!(product.right_projection.map(), &[0, 1, 0, 1]);
}

#[test]
fn relabel_preserves_order_and_allows_duplicate_labels() {
    let poset = Poset::from_edges(vec![0, 1, 2], [Edge::new(0, 1), Edge::new(1, 2)]).unwrap();
    let relabeled = poset.relabel(|id| if *id == 1 { "middle" } else { "end" });

    assert_eq!(relabeled.elements(), &["end", "middle", "end"]);
    assert!(relabeled.leq(0, 2));
    assert_eq!(relabeled.cover_relations(), poset.cover_relations());

    let lattice = chain(3);
    let relabeled_lattice = lattice.relabel(|id| format!("x{id}"));
    assert_eq!(relabeled_lattice.elements(), &["x0", "x1", "x2"]);
    assert_eq!(relabeled_lattice.meet_id(1, 2), 1);
    assert_eq!(relabeled_lattice.join_id(0, 1), 1);
}

#[test]
fn lattice_validation_checks_meets_and_joins() {
    let empty = Lattice::new(Poset::<usize>::from_relation(vec![], vec![]).unwrap());
    assert!(matches!(empty, Err(LatticeError::Empty)));

    let vee = Poset::from_edges(vec![0, 1, 2], [Edge::new(0, 1), Edge::new(0, 2)]).unwrap();
    assert!(vee.meet(1, 2).is_some());
    assert!(vee.join(1, 2).is_none());
    assert!(!vee.is_lattice());
    assert!(matches!(
        Lattice::new(vee),
        Err(LatticeError::NotALattice { .. })
    ));

    let lattice = diamond();
    assert_eq!(lattice.meet_id(1, 2), 0);
    assert_eq!(lattice.join_id(1, 2), 3);
    assert_eq!(lattice.bottom(), 0);
    assert_eq!(lattice.top(), 3);
    assert!(lattice.is_fusion_of_total_orders());
    assert!(chain(4).is_fusion_of_total_orders());

    let grid = Lattice::new(
        Poset::from_vec_by(
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            |left, right| left.0 <= right.0 && left.1 <= right.1,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!grid.is_fusion_of_total_orders());
}

#[test]
fn morphisms_validate_monotonicity_and_lattice_preservation() {
    let chain = Arc::new(chain(2));
    let chain_poset = Arc::new(chain.as_poset().clone());

    let bad_poset_map = PosetMap::new(
        Arc::clone(&chain_poset),
        Arc::clone(&chain_poset),
        vec![1, 0],
    );
    assert!(matches!(bad_poset_map, Err(MapError::NotMonotone { .. })));

    let diamond = Arc::new(diamond());
    let bad_lattice_map = LatticeMap::new(diamond, chain, vec![0, 1, 1, 1]);
    assert!(matches!(
        bad_lattice_map,
        Err(MapError::DoesNotPreserveMeet { .. })
    ));
}

#[test]
fn lattice_fusion_identifies_bottom_and_top_with_embeddings() {
    let left = Arc::new(chain(3));
    let right = Arc::new(chain(3));

    let fusion = horizontal_join(Arc::clone(&left), Arc::clone(&right)).unwrap();

    assert_eq!(fusion.lattice.size(), 4);
    assert_eq!(fusion.left.map(), &[0, 1, 2]);
    assert_eq!(fusion.right.map(), &[0, 3, 2]);
    assert_eq!(fusion.lattice.meet_id(1, 3), 0);
    assert_eq!(fusion.lattice.join_id(1, 3), 2);
    assert!(fusion.lattice.is_fusion_of_total_orders());

    let collapsed = fusion.lattice.relabel(|side| match side {
        Either::Left(label) | Either::Right(label) => *label,
    });
    assert_eq!(collapsed.elements(), &[0, 1, 2, 1]);
    assert_eq!(collapsed.meet_id(1, 3), 0);
    assert_eq!(collapsed.join_id(1, 3), 2);
}

#[test]
fn transfer_systems_are_domain_types_with_containment_lattices() {
    let lattice = Arc::new(chain(2));
    let systems = TransferSystems::on(Arc::clone(&lattice)).unwrap();

    assert_eq!(systems.proper_edges(), &[Edge::new(0, 1)]);
    assert_eq!(systems.systems().len(), 2);
    assert_eq!(systems.containment_lattice().size(), 2);

    let identity = LatticeMap::new(Arc::clone(&lattice), Arc::clone(&lattice), vec![0, 1]).unwrap();
    let pullback = systems.pullback(&identity, &systems).unwrap();
    assert_eq!(pullback.map(), &[0, 1]);
}

#[test]
fn tikz_rendering_has_structured_hasse_and_transfer_outputs() {
    let lattice = chain(2);
    let tikz = lattice.to_tikz().render();
    assert!(tikz.contains("\\begin{tikzpicture}"));
    assert!(tikz.contains("\\draw (p0) -- (p1);"));
    assert!(tikz.contains("\\node[circle, draw, inner sep=1.5pt] (p0)"));

    let systems = TransferSystems::on(Arc::new(lattice)).unwrap();
    let transfer_tikz = systems.to_tikz().render();
    assert!(transfer_tikz.contains("\\tikz[scale=.35"));
    assert!(transfer_tikz.contains("draw=blue"));
}

#[test]
fn tikz_bends_colinear_edges_that_would_cross_nodes() {
    let poset = chain(3).into_poset();
    let options = PosetTikzOptions {
        full_relations: true,
        ..PosetTikzOptions::default()
    };

    let tikz =
        poset_to_tikz_with(&poset, &options, |id, _| TikzLabel::escaped(id.to_string())).render();

    assert!(tikz.contains("\\draw (p0) -- (p1);"));
    assert!(tikz.contains("\\draw (p0) to[bend"));
    assert!(tikz.contains("(p2);"));
    assert!(tikz.contains("\\draw (p1) -- (p2);"));
    assert!(!tikz.contains("\\draw (p1) to[bend"));
}
