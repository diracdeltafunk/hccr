use super::*;
use std::sync::Arc;

#[test]
fn tikz_options_render_empty_and_non_empty_lists() {
    let options = TikzOptions::default();
    assert!(options.is_empty());
    assert_eq!(options.render_inner(), "");
    assert_eq!(options.render_brackets(), "");
    assert_eq!(options.render_braces(), "{}");
    assert_eq!(options.iter().collect::<Vec<_>>(), Vec::<&str>::new());

    let mut options = TikzOptions::new(["draw"]);
    options.push("thick");
    options.extend(["orange"]);
    assert!(!options.is_empty());
    assert_eq!(
        options.iter().collect::<Vec<_>>(),
        vec!["draw", "thick", "orange"]
    );
    assert_eq!(options.render_inner(), "draw, thick, orange");
    assert_eq!(options.render_brackets(), "[draw, thick, orange]");
    assert_eq!(options.render_braces(), "{draw, thick, orange}");
}

#[test]
fn style_rendering_uses_tikz_style_body_syntax() {
    let mut picture = TikzPicture::new();
    picture.push(TikzStyle {
        name: "foo".to_string(),
        options: TikzOptions::new(["draw", "circle"]),
    });

    assert_eq!(
        picture.render(),
        "\\begin{tikzpicture}\n\\tikzset{foo/.style={draw, circle}}\n\\end{tikzpicture}\n"
    );
}

#[test]
fn straight_paths_render_with_line_operation() {
    let mut picture = TikzPicture::new();
    picture.push(TikzPath {
        from: TikzCoord::named("a"),
        to: TikzCoord::named("b"),
        operation: TikzPathOperation::Line,
        options: TikzOptions::new(["thick"]),
    });

    assert_eq!(
        picture.render(),
        "\\begin{tikzpicture}\n\\draw[thick] (a) -- (b);\n\\end{tikzpicture}\n"
    );
}

#[test]
fn bent_paths_render_with_to_operation() {
    let mut picture = TikzPicture::new();
    picture.push(TikzPath {
        from: TikzCoord::named("a"),
        to: TikzCoord::named("b"),
        operation: TikzPathOperation::To,
        options: TikzOptions::new(["bend left=18"]),
    });

    assert_eq!(
        picture.render(),
        "\\begin{tikzpicture}\n\\draw[bend left=18] (a) to (b);\n\\end{tikzpicture}\n"
    );
}

#[test]
fn picture_items_render_in_insertion_order() {
    let mut picture = TikzPicture::new();
    picture.push_raw("% first");
    picture.push(TikzNode {
        name: None,
        at: TikzCoord::point(0.0, 0.0),
        label: TikzLabel::raw("x"),
        options: TikzOptions::default(),
    });
    picture.push_raw("% last");

    assert_eq!(
        picture.render(),
        "\\begin{tikzpicture}\n% first\n\\node at (0.000,0.000) {x};\n% last\n\\end{tikzpicture}\n"
    );
}

#[test]
fn transfer_system_glyphs_render_as_inline_tikz_pictures() {
    let chain = Arc::new(
        Lattice::new(Poset::from_edges(vec![0i32, 1], [Edge::new(0, 1)]).unwrap()).unwrap(),
    );
    let system = chain
        .transfer_universe()
        .transfer_systems()
        .into_iter()
        .max_by_key(|system| system.edges(false).len())
        .expect("two-element chain should have transfer systems");

    let inline = small_transfer_system_picture(&system, &TransferSystemGlyphOptions::default())
        .render_inline();

    assert!(inline.starts_with("\\tikz[scale=0.35, baseline=-.5ex]{"));
    assert!(inline.contains("\\draw[line width=.75pt, orange]"));
    assert!(inline.contains("\\fill[black]"));
    assert!(inline.ends_with('}'));
}

#[test]
fn transfer_system_glyphs_show_all_relations_by_default() {
    let diamond = Arc::new(
        Lattice::new(
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
        )
        .unwrap(),
    );
    let containment = diamond.clone().transfer_systems_containment().unwrap();
    let top = containment
        .system(containment.top())
        .expect("the transfer-system lattice has a top element");

    let default =
        small_transfer_system_picture(&top, &TransferSystemGlyphOptions::default()).render();
    assert_eq!(default.matches("line width=.75pt, orange").count(), 5);
    assert_eq!(default.matches("black!30").count(), 0);
    assert_eq!(default.matches("\\draw").count(), 5);

    let glyph_options = TransferSystemGlyphOptions::default();
    let mixed = SuborderGlyphRenderer::new(diamond.as_ref(), &glyph_options)
        .render_with(|edge| edge == Edge::new(0, 1))
        .render();
    assert_eq!(mixed.matches("line width=.75pt, orange").count(), 1);
    assert_eq!(mixed.matches("black!30").count(), 4);
    assert_eq!(mixed.matches("\\draw").count(), 5);

    let bottom = containment
        .system(containment.bottom())
        .expect("the transfer-system lattice has a bottom element");
    let bottom_default =
        small_transfer_system_picture(&bottom, &TransferSystemGlyphOptions::default()).render();
    assert_eq!(
        bottom_default.matches("line width=.75pt, orange").count(),
        0
    );
    assert_eq!(bottom_default.matches("black!30").count(), 5);

    let covers_only = TransferSystemGlyphOptions {
        ambient_relations: RelationDisplay::Covers,
        highlighted_relations: RelationDisplay::Covers,
        ..TransferSystemGlyphOptions::default()
    };
    let covers = small_transfer_system_picture(&top, &covers_only).render();
    assert_eq!(covers.matches("line width=.75pt, orange").count(), 4);
    assert_eq!(covers.matches("black!30").count(), 0);
    assert_eq!(covers.matches("\\draw").count(), 4);
}

#[test]
fn transfer_system_glyphs_bend_transitive_relations_around_colinear_nodes() {
    let chain = Arc::new(
        Lattice::new(
            Poset::from_edges(vec![0i32, 1, 2], [Edge::new(0, 1), Edge::new(1, 2)]).unwrap(),
        )
        .unwrap(),
    );
    let containment = chain.transfer_systems_containment().unwrap();
    let top = containment
        .system(containment.top())
        .expect("the transfer-system lattice has a top element");

    let options = TransferSystemGlyphOptions::default();
    let rendered = small_transfer_system_picture(&top, &options).render();
    assert_eq!(rendered.matches("bend right=18").count(), 1);
    assert!(!rendered.contains("\\draw[black!30, bend right=18]"));
    assert!(rendered.contains("\\draw[line width=.75pt, orange, bend right=18]"));

    let mut straight_options = options;
    straight_options.bend_colinear_edges = false;
    let straight = small_transfer_system_picture(&top, &straight_options).render();
    assert!(!straight.contains("bend"));
}

#[test]
fn transfer_system_glyphs_can_use_labels_in_place_of_dots() {
    let chain = Arc::new(
        Lattice::new(Poset::from_edges(vec![0i32, 1], [Edge::new(0, 1)]).unwrap()).unwrap(),
    );
    let containment = chain.transfer_systems_containment().unwrap();
    let top = containment
        .system(containment.top())
        .expect("the transfer-system lattice has a top element");
    let options = TransferSystemGlyphOptions {
        node_display: GlyphNodeDisplay::escaped(["bottom", "top"]),
        ..TransferSystemGlyphOptions::default()
    };

    let rendered = small_transfer_system_picture(&top, &options).render();
    assert!(rendered.contains("{bottom}"));
    assert!(rendered.contains("{top}"));
    assert!(rendered.contains("font=\\tiny"));
    assert!(!rendered.contains("\\fill[black]"));
}

#[test]
fn escape_tikz_escapes_latex_special_characters() {
    assert_eq!(
        escape_tikz("\\{}$&%#_^~"),
        "\\textbackslash{}\\{\\}\\$\\&\\%\\#\\_\\^{}\\~{}"
    );
}

#[test]
fn layout_uses_element_ids_as_the_antichain_tie_break() {
    let poset = Poset::from_edges(vec![0i32, 1, 2], []).unwrap();
    let picture = poset_to_tikz_with(&poset, &PosetTikzOptions::default(), |_id, element| {
        TikzLabel::escaped(element.to_string())
    });

    let rendered = picture.render();
    assert!(rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p0) at (-1.800,0.000) {0};"));
    assert!(rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p1) at (0.000,0.000) {1};"));
    assert!(rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p2) at (1.800,0.000) {2};"));
}

#[test]
fn centered_layout_gives_n5_five_distinct_heights() {
    // The long branch is 0 < a < c < 1, while b lies strictly between
    // 0 and 1. Centering b in its feasible rank interval gives the usual
    // pentagon rather than placing it level with a.
    let n5 = Poset::from_edges(
        vec!["0", "a", "b", "c", "1"],
        [
            Edge::new(0, 1),
            Edge::new(1, 3),
            Edge::new(3, 4),
            Edge::new(0, 2),
            Edge::new(2, 4),
        ],
    )
    .unwrap();

    let coords = default_layout(&n5, 1.8, 1.4);
    let expected_heights = [0.0, 1.4, 2.1, 2.8, 4.2];
    for (id, expected) in expected_heights.into_iter().enumerate() {
        assert!(
            (coords[&id].1 - expected).abs() < 1e-9,
            "unexpected N5 coordinates: {coords:?}"
        );
    }

    assert!((coords[&1].0 - coords[&3].0).abs() < 1e-9);
    assert!(
        (coords[&1].0 - coords[&2].0) * (coords[&3].0 - coords[&2].0) > 0.0,
        "the short branch should be on the other side of the long branch: {coords:?}"
    );
}

#[test]
fn layered_sweeps_remove_an_avoidable_crossing() {
    // Element-id order crosses 1--4 with 2--3. The adjacency-based order
    // should instead keep the two branches on consistent sides.
    let poset = Poset::from_edges(
        (0..6).collect::<Vec<_>>(),
        [
            Edge::new(0, 1),
            Edge::new(0, 2),
            Edge::new(1, 4),
            Edge::new(2, 3),
            Edge::new(3, 5),
            Edge::new(4, 5),
        ],
    )
    .unwrap();

    let coords = default_layout(&poset, 1.8, 1.4);
    assert!(
        (coords[&1].0 - coords[&2].0) * (coords[&4].0 - coords[&3].0) > 0.0,
        "the two middle cover edges should not cross: {coords:?}"
    );
}

#[test]
fn adjacent_switching_separates_staggered_branches() {
    // One shorter branch has a half-rank vertex between two vertices of
    // the longer branch. Adjacent exact-score switches should place the
    // branches on separate rails rather than stacking all three vertices.
    let poset = Poset::from_edges(
        (0..6).collect::<Vec<_>>(),
        [
            Edge::new(0, 1),
            Edge::new(0, 2),
            Edge::new(1, 5),
            Edge::new(2, 3),
            Edge::new(3, 4),
            Edge::new(3, 5),
        ],
    )
    .unwrap();
    let mut covers: Vec<_> = poset.cover_relations().into_iter().collect();
    covers.sort_unstable();

    let coords = default_layout(&poset, 2.0, 2.0);
    assert!((coords[&1].0 - coords[&5].0).abs() < 1e-9);
    assert!((coords[&2].0 - coords[&3].0).abs() < 1e-9);
    assert!((coords[&3].0 - coords[&4].0).abs() < 1e-9);
    assert!((coords[&1].0 - coords[&2].0).abs() > 1e-9);

    let points: Vec<_> = (0..poset.size())
        .map(|id| GridPoint {
            x: coords[&id].0.round() as i64,
            y: coords[&id].1.round() as i64,
        })
        .collect();
    assert_eq!(
        straight_geometry_defects(&covers, &points),
        StraightGeometryDefects {
            proper_crossings: 0,
            node_intersections: 0,
            collinear_overlaps: 0,
        }
    );
    assert!(
        edge_bend_decisions(&covers, &coords, 1e-6)
            .into_iter()
            .all(|bend| !bend)
    );
}

#[test]
fn local_geometry_delta_matches_full_rescoring() {
    let edges = [
        Edge::new(0, 1),
        Edge::new(0, 2),
        Edge::new(1, 5),
        Edge::new(2, 3),
        Edge::new(3, 4),
        Edge::new(3, 5),
    ];
    let points = [
        GridPoint { x: 0, y: 0 },
        GridPoint { x: -1, y: 3 },
        GridPoint { x: -1, y: 2 },
        GridPoint { x: -1, y: 4 },
        GridPoint { x: -1, y: 6 },
        GridPoint { x: 1, y: 6 },
    ];
    let full_before = straight_geometry_defects(&edges, &points);

    for moved in 0..points.len() {
        for delta in [-2, 2] {
            let old_contribution = straight_geometry_contribution(&edges, &points, &[moved]);
            let mut changed = points;
            changed[moved].x += delta;
            let new_contribution = straight_geometry_contribution(&edges, &changed, &[moved]);
            assert_eq!(
                full_before.replace_contribution(old_contribution, new_contribution),
                straight_geometry_defects(&edges, &changed)
            );
        }
    }

    for left in 0..points.len() {
        for right in left + 1..points.len() {
            let moved = [left, right];
            let old_contribution = straight_geometry_contribution(&edges, &points, &moved);
            let mut changed = points;
            let (left_points, right_points) = changed.split_at_mut(right);
            std::mem::swap(&mut left_points[left].x, &mut right_points[0].x);
            let new_contribution = straight_geometry_contribution(&edges, &changed, &moved);
            assert_eq!(
                full_before.replace_contribution(old_contribution, new_contribution),
                straight_geometry_defects(&edges, &changed)
            );
        }
    }
}

#[test]
fn straight_edge_score_prevents_a_dummy_route_regression() {
    // Optimizing only the subdivided dummy routes can cross 1--3 with
    // 2--4 after those routes are replaced by TikZ's straight segments.
    let poset = Poset::from_edges(
        (0..5).collect::<Vec<_>>(),
        [
            Edge::new(0, 1),
            Edge::new(1, 3),
            Edge::new(1, 4),
            Edge::new(2, 4),
        ],
    )
    .unwrap();
    let mut covers: Vec<_> = poset.cover_relations().into_iter().collect();
    covers.sort_unstable();

    let coords = default_layout(&poset, 2.0, 2.0);
    let points: Vec<_> = (0..poset.size())
        .map(|id| GridPoint {
            x: coords[&id].0.round() as i64,
            y: coords[&id].1.round() as i64,
        })
        .collect();
    assert_eq!(
        straight_geometry_defects(&covers, &points).proper_crossings,
        0
    );
}
