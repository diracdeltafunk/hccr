use hccr::prelude::*;
use hccr::tikz::{Glyph, PosetLayout};

fn pentagon() -> Lattice {
    Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )
    .unwrap()
}

fn drawn(edges: &[hccr::tikz::DrawnEdge]) -> EdgeSet {
    edges.iter().map(|edge| edge.relation()).collect()
}

fn is_highlighted(glyph: &Glyph, edge: &hccr::tikz::DrawnEdge) -> bool {
    edge.options == glyph.highlight_options
}

#[test]
fn drawings_show_the_requested_relations() {
    let lattice = pentagon();
    let drawing = lattice.drawing();
    assert_eq!(drawing.positions.len(), lattice.size());
    assert_eq!(drawing.nodes.len(), lattice.size());
    assert_eq!(drawn(&drawing.edges), lattice.covers());

    let drawing = drawing.edges(lattice.proper_relations());
    assert_eq!(drawn(&drawing.edges), lattice.proper_relations());
}

#[test]
fn glyphs_highlight_exactly_the_drawn_relations_of_their_system() {
    let lattice = pentagon();
    let tr = lattice.transfer_systems();

    // By default each glyph draws every relation of the lattice.
    let drawing = tr.drawing();
    for (id, node) in drawing.nodes.iter().enumerate() {
        let glyph = &node.content;
        assert_eq!(drawn(&glyph.drawing.edges), lattice.proper_relations());
        for edge in &glyph.drawing.edges {
            assert_eq!(
                is_highlighted(glyph, edge),
                tr.system(id).contains_relation(edge.relation())
            );
        }
    }

    // On the Hasse diagram of the lattice, a cover is highlighted when it
    // belongs to the transfer system, even if it is not a cover of the
    // transfer system itself.
    let drawing = tr.drawing().glyph_edges(lattice.covers());
    for (id, node) in drawing.nodes.iter().enumerate() {
        let glyph = &node.content;
        assert_eq!(drawn(&glyph.drawing.edges), lattice.covers());
        for edge in &glyph.drawing.edges {
            assert_eq!(
                is_highlighted(glyph, edge),
                tr.system(id).contains_relation(edge.relation())
            );
        }
    }
}

#[test]
fn layouts_can_be_supplied_and_replaced() {
    let lattice = pentagon();
    let diagonal = |poset: &Poset| poset.ids().map(|id| (id as f64, id as f64)).collect();
    let drawing = lattice.drawing_with(diagonal);
    for (id, &position) in drawing.positions.iter().enumerate() {
        assert_eq!(position, (id as f64, id as f64));
    }

    let relaid = drawing.layout(PosetLayout::TEXTBOOK);
    assert_eq!(
        relaid.positions,
        lattice.drawing_with(PosetLayout::TEXTBOOK).positions
    );
}
