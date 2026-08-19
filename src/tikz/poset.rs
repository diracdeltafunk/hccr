use super::ToTikz;
use super::edge_routing::edge_bend_decisions;
use super::layout::{PosetLayoutAlgorithm, layout_with_covers};
use super::syntax::*;
use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};
use std::collections::HashMap;
use std::fmt;

/// Layout and styling options for rendering a finite poset.
#[derive(Debug, Clone)]
pub struct PosetTikzOptions {
    /// Options placed on the outer `tikzpicture`.
    pub picture_options: TikzOptions,
    /// Minimum horizontal spacing between elements of the same rank.
    pub x_spacing: f64,
    /// Vertical spacing between consecutive ranks on a longest chain.
    pub y_spacing: f64,
    /// The algorithm used to assign coordinates to elements.
    pub layout_algorithm: PosetLayoutAlgorithm,
    /// Options applied to each element node.
    pub node_options: TikzOptions,
    /// Options applied to each order-relation edge.
    pub edge_options: TikzOptions,
    /// Whether to draw small auxiliary labels showing element ids.
    pub debug_element_ids: bool,
    /// Options applied to the auxiliary element-id labels.
    pub debug_id_options: TikzOptions,
    /// Offset for auxiliary element-id labels.
    pub debug_id_offset: (f64, f64),
    /// Whether to draw all proper relations instead of only cover relations.
    pub full_relations: bool,
    /// Whether to bend edges that would overlap nodes or preferred edges.
    pub bend_colinear_edges: bool,
    /// Bend angle used when an edge must be curved.
    pub bend_angle: f64,
    /// Numerical tolerance used in colinearity checks.
    pub colinear_tolerance: f64,
    /// Explicit coordinates for selected element ids.
    pub coordinate_overrides: HashMap<ElementId, (f64, f64)>,
}

impl Default for PosetTikzOptions {
    fn default() -> Self {
        Self {
            picture_options: TikzOptions::default(),
            x_spacing: 1.8,
            y_spacing: 1.4,
            layout_algorithm: PosetLayoutAlgorithm::default(),
            node_options: TikzOptions::new(["circle", "draw", "inner sep=1.5pt"]),
            edge_options: TikzOptions::default(),
            debug_element_ids: false,
            debug_id_options: TikzOptions::new([
                "circle",
                "draw=red!70!black",
                "fill=white",
                "inner sep=.6pt",
                "font=\\scriptsize",
            ]),
            debug_id_offset: (0., 0.),
            full_relations: false,
            bend_colinear_edges: true,
            bend_angle: 18.0,
            colinear_tolerance: 1e-6,
            coordinate_overrides: HashMap::new(),
        }
    }
}

/// Renders a poset as a TikZ Hasse diagram with custom labels.
///
/// By default, only cover relations are drawn.  Set
/// [`PosetTikzOptions::full_relations`] to draw every proper relation.
pub fn poset_to_tikz_with<A, F>(
    poset: &Poset<A>,
    options: &PosetTikzOptions,
    mut label: F,
) -> TikzPicture
where
    F: FnMut(ElementId, &A) -> TikzLabel,
{
    let mut covers: Vec<_> = poset.cover_relations().into_iter().collect();
    covers.sort_unstable();
    let mut coords = layout_with_covers(
        poset.size(),
        &covers,
        options.x_spacing,
        options.y_spacing,
        options.layout_algorithm,
    );
    for (&id, &coordinate) in &options.coordinate_overrides {
        if id < poset.size() {
            coords.insert(id, coordinate);
        }
    }

    let mut picture = TikzPicture::with_options(options.picture_options.clone());
    let edges: Vec<Edge> = if options.full_relations {
        poset.proper_relations_iter().collect()
    } else {
        covers
    };
    let bend_edges = if options.bend_colinear_edges {
        edge_bend_decisions(&edges, &coords, options.colinear_tolerance)
    } else {
        vec![false; edges.len()]
    };

    for (id, element) in poset.elements().iter().enumerate() {
        picture.push(TikzNode {
            name: Some(node_name(id)),
            at: coords[&id].into(),
            label: label(id, element),
            options: options.node_options.clone(),
        });
    }

    for (index, edge) in edges.iter().copied().enumerate() {
        let mut edge_options = options.edge_options.clone();
        let operation = if bend_edges[index] {
            let direction = if index % 2 == 0 { "left" } else { "right" };
            edge_options.push(format!("bend {direction}={}", options.bend_angle));
            TikzPathOperation::To
        } else {
            TikzPathOperation::Line
        };
        picture.push(TikzPath {
            from: TikzCoord::named(node_name(edge.from)),
            to: TikzCoord::named(node_name(edge.to)),
            operation,
            options: edge_options,
        });
    }

    if options.debug_element_ids {
        picture.push(TikzScope {
            options: TikzOptions::default(),
            picture: debug_id_picture(poset.size(), options),
        });
    }

    picture
}

impl<A: fmt::Display> ToTikz for Poset<A> {
    type Options = PosetTikzOptions;

    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture {
        poset_to_tikz_with(self, options, |_id, element| {
            TikzLabel::escaped(element.to_string())
        })
    }
}

impl<A: fmt::Display> ToTikz for Lattice<A> {
    type Options = PosetTikzOptions;

    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture {
        poset_to_tikz_with(self.as_poset(), options, |_id, element| {
            TikzLabel::escaped(element.to_string())
        })
    }
}

fn debug_id_picture(size: usize, options: &PosetTikzOptions) -> TikzPicture {
    let mut picture = TikzPicture::new();
    for id in 0..size {
        picture.push(TikzNode {
            name: None,
            at: TikzCoord::raw(format!(
                "([xshift={},yshift={}]{}.north east)",
                options.debug_id_offset.0,
                options.debug_id_offset.1,
                node_name(id)
            )),
            label: TikzLabel::raw(format!("$ {id} $")),
            options: options.debug_id_options.clone(),
        });
    }
    picture
}

fn node_name(id: ElementId) -> String {
    format!("p{id}")
}
