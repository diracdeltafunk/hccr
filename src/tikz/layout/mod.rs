mod geometry;
mod layered;

use crate::poset::{Edge, ElementId};
use std::collections::{BTreeSet, HashMap};

pub(in crate::tikz) use geometry::{
    GridPoint, StraightGeometryDefects, straight_geometry_contribution, straight_geometry_defects,
};
use layered::{LayeredCoverGraph, doubled_centered_slot};

/// The algorithm used to assign coordinates to elements of a finite poset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PosetLayoutAlgorithm {
    /// Place elements at their earliest possible height and order by element id.
    ///
    /// The height of `x` is the length of a longest chain from a minimal
    /// element to `x`. This deterministic layout performs no crossing
    /// optimization.
    Ranked,
    /// Center feasible heights and heuristically reduce edge crossings.
    ///
    /// Elements of a non-graded poset may have vertical slack between their
    /// earliest and latest feasible heights. The algorithm centers each one in
    /// that interval, inserts temporary vertices where cover edges span
    /// multiple layers, and repeatedly reorders layers to improve straight-line
    /// geometry. Temporary vertices are used only for optimization and are not
    /// rendered.
    #[default]
    CrossingReduced,
}

pub(super) fn layout_with_covers(
    size: usize,
    covers: &[Edge],
    x_spacing: f64,
    y_spacing: f64,
    algorithm: PosetLayoutAlgorithm,
) -> HashMap<ElementId, (f64, f64)> {
    let vertical = vertical_levels(size, covers);
    match algorithm {
        PosetLayoutAlgorithm::Ranked => {
            id_order_coordinates(&vertical.earliest, x_spacing, y_spacing)
        }
        PosetLayoutAlgorithm::CrossingReduced => {
            let mut layout = LayeredCoverGraph::new(&vertical.centered, covers);
            let improved_defects = layout.reduce_crossings();

            // Crossing reduction is performed on a properized graph with temporary
            // dummy vertices, but TikZ emits straight cover segments. Judge the real
            // segments and retain the candidate with better unrounded cover geometry.
            // Styling, manual overrides, and optional bends happen later.
            let baseline_points = id_order_grid_points(&vertical.earliest);
            if straight_geometry_defects(covers, &baseline_points) < improved_defects {
                id_order_coordinates(&vertical.earliest, x_spacing, y_spacing)
            } else {
                layout.real_coordinates(x_spacing, y_spacing)
            }
        }
    }
}

/// Assigns twice the centered feasible rank of each element, following the
/// vertical-ranking rule in Freese's automated lattice-drawing algorithm.
///
/// `height[v]` is the earliest rank at which `v` can occur, while
/// `longest_height - depth[v]` is its latest feasible rank. Their midpoint
/// makes graded posets look exactly as before and uses the available vertical
/// slack in non-graded posets. Keeping the doubled value as an integer also
/// gives crossing reduction explicit half-rank layers to work with.
struct VerticalLevels {
    earliest: Vec<usize>,
    centered: Vec<usize>,
}

fn vertical_levels(size: usize, covers: &[Edge]) -> VerticalLevels {
    let mut incoming = vec![Vec::new(); size];
    let mut outgoing = vec![Vec::new(); size];
    for &edge in covers {
        incoming[edge.to].push(edge.from);
        outgoing[edge.from].push(edge.to);
    }

    let mut remaining_predecessors: Vec<_> = incoming.iter().map(Vec::len).collect();
    let mut ready: BTreeSet<_> = remaining_predecessors
        .iter()
        .enumerate()
        .filter_map(|(id, &count)| (count == 0).then_some(id))
        .collect();
    let mut topological_order = Vec::with_capacity(size);
    let mut heights = vec![0usize; size];

    while let Some(id) = ready.pop_first() {
        topological_order.push(id);
        for &upper in &outgoing[id] {
            heights[upper] = heights[upper].max(heights[id] + 1);
            remaining_predecessors[upper] -= 1;
            if remaining_predecessors[upper] == 0 {
                ready.insert(upper);
            }
        }
    }
    debug_assert_eq!(topological_order.len(), size);

    let mut depths = vec![0usize; size];
    for &id in topological_order.iter().rev() {
        depths[id] = outgoing[id]
            .iter()
            .map(|&upper| depths[upper] + 1)
            .max()
            .unwrap_or(0);
    }

    let longest_height = heights.iter().copied().max().unwrap_or(0);
    let centered = heights
        .iter()
        .copied()
        .zip(depths)
        .map(|(height, depth)| height + longest_height - depth)
        .collect();
    VerticalLevels {
        earliest: heights,
        centered,
    }
}

fn id_order_layers(ranks: &[usize]) -> Vec<Vec<ElementId>> {
    let layer_count = ranks.iter().copied().max().map_or(0, |maximum| maximum + 1);
    let mut layers = vec![Vec::new(); layer_count];
    for (id, &rank) in ranks.iter().enumerate() {
        layers[rank].push(id);
    }
    layers
}

fn id_order_grid_points(ranks: &[usize]) -> Vec<GridPoint> {
    let layers = id_order_layers(ranks);
    let mut points = vec![GridPoint { x: 0, y: 0 }; ranks.len()];
    for (rank, layer) in layers.iter().enumerate() {
        for (position, &id) in layer.iter().enumerate() {
            points[id] = GridPoint {
                x: doubled_centered_slot(position, layer.len()),
                y: 2 * rank as i64,
            };
        }
    }
    points
}

fn id_order_coordinates(
    ranks: &[usize],
    x_spacing: f64,
    y_spacing: f64,
) -> HashMap<ElementId, (f64, f64)> {
    let points = id_order_grid_points(ranks);
    points
        .into_iter()
        .enumerate()
        .map(|(id, point)| {
            (
                id,
                (
                    point.x as f64 * x_spacing / 2.0,
                    point.y as f64 * y_spacing / 2.0,
                ),
            )
        })
        .collect()
}
