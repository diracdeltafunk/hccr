mod coordinates;
mod geometry;
mod layered;

use crate::poset::{Edge, ElementId};
use std::collections::{BTreeSet, HashMap};

pub(in crate::tikz) use geometry::{
    GridPoint, StraightGeometryDefects, straight_geometry_contribution, straight_geometry_defects,
};
use layered::{LayeredCoverGraph, doubled_centered_slot};

/// How the elements of a finite poset are placed in a drawing.
///
/// A layout is built in three independent stages, in the style of Sugiyama's
/// layered drawing method:
///
/// 1. [`levels`](Self::levels) chooses the height of every element;
/// 2. [`ordering`](Self::ordering) chooses the left-to-right order of the
///    elements at each height;
/// 3. [`coordinates`](Self::coordinates) turns those orders into horizontal
///    positions.
///
/// Every combination of stages is valid. The associated constants name the
/// standard combinations; start from one of them and change a single stage
/// to experiment:
///
/// ```
/// use hccr::tikz::{LayoutCoordinates, PosetLayout};
///
/// let layout = PosetLayout {
///     coordinates: LayoutCoordinates::Aligned,
///     ..PosetLayout::TEXTBOOK
/// };
/// assert_ne!(layout, PosetLayout::TEXTBOOK);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PosetLayout {
    /// How heights are chosen.
    pub levels: LayoutLevels,
    /// How elements at the same height are ordered.
    pub ordering: LayoutOrdering,
    /// How horizontal positions are chosen.
    pub coordinates: LayoutCoordinates,
}

impl PosetLayout {
    /// Elements at their longest-path heights, ordered by element id and
    /// evenly spaced. No optimization of any kind.
    pub const RANKED: Self = Self {
        levels: LayoutLevels::LongestPath,
        ordering: LayoutOrdering::ElementOrder,
        coordinates: LayoutCoordinates::Grid,
    };

    /// The textbook Sugiyama layout: longest-path heights, barycenter
    /// ordering, and evenly spaced layers.
    pub const TEXTBOOK: Self = Self {
        levels: LayoutLevels::LongestPath,
        ordering: LayoutOrdering::Barycenter,
        coordinates: LayoutCoordinates::Grid,
    };

    /// Centered heights, crossing reduction scored on the drawn straight
    /// edges, and evenly spaced layers. This is the default.
    pub const CROSSING_REDUCED: Self = Self {
        levels: LayoutLevels::Centered,
        ordering: LayoutOrdering::Refined,
        coordinates: LayoutCoordinates::Grid,
    };

    /// As [`Self::CROSSING_REDUCED`], but with
    /// [`LayoutCoordinates::Balanced`] positions: compact, with every element
    /// near its neighbours.
    pub const BALANCED: Self = Self {
        coordinates: LayoutCoordinates::Balanced,
        ..Self::CROSSING_REDUCED
    };

    /// As [`Self::CROSSING_REDUCED`], but with [`LayoutCoordinates::Aligned`]
    /// positions: straight chains and parallel edges.
    pub const ALIGNED: Self = Self {
        coordinates: LayoutCoordinates::Aligned,
        ..Self::CROSSING_REDUCED
    };
}

impl Default for PosetLayout {
    fn default() -> Self {
        Self::CROSSING_REDUCED
    }
}

/// How the heights of elements are chosen.
///
/// Both choices are strictly monotone: if `x < y` then `x` is drawn strictly
/// below `y`. They agree on graded posets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LayoutLevels {
    /// The height of `x` is the length of a longest chain from a minimal
    /// element up to `x`, so every element sits as low as it can.
    LongestPath,
    /// Each element is centered between the lowest and highest heights it
    /// could occupy, following Freese's automated lattice-drawing algorithm.
    /// The height of `x` is the average of its longest-path height and its
    /// highest feasible height (the length of a longest chain in the poset,
    /// minus the length of a longest chain from `x` up to a maximal element),
    /// so it may be a half-integer.
    #[default]
    Centered,
}

/// How the elements at each height are ordered from left to right.
///
/// Where a cover relation spans more than one layer, the orderings other than
/// [`Self::ElementOrder`] route it through invisible bend points on the
/// intermediate layers. These take part in the ordering and reserve room for
/// the edge, but are not drawn: edges are always straight segments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LayoutOrdering {
    /// Order by element id. Long edges reserve no room.
    ElementOrder,
    /// Alternately sweep up and down the layers, sorting each layer by the
    /// mean position of its neighbours in the layer just swept, and keep the
    /// order with the fewest crossings.
    Barycenter,
    /// [`Self::Barycenter`], followed by swaps of adjacent elements whenever a
    /// swap improves the straight-line drawing: fewer edges through nodes,
    /// fewer overlapping edges, and fewer crossings. With
    /// [`LayoutCoordinates::Grid`], the result is also compared with
    /// [`PosetLayout::RANKED`], and the better of the two is kept.
    #[default]
    Refined,
}

/// How horizontal positions are chosen once each layer is ordered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LayoutCoordinates {
    /// Space the vertices of each layer, including the bend points of long
    /// edges, evenly and centered. Layers crossed by many long edges become
    /// wide.
    #[default]
    Grid,
    /// Place each element as close as possible to the mean position of its
    /// neighbours, in the least-squares sense, keeping the orders and a
    /// minimum spacing; then remove any overall lean. Compact, but edges are
    /// rarely exactly parallel.
    Balanced,
    /// The Brandes–Köpf algorithm: line each element up vertically with a
    /// median neighbour wherever possible, so that chains are drawn straight
    /// and edges come out parallel, then balance four such alignments
    /// (leaning up or down, left or right) against each other.
    Aligned,
}

pub(super) fn layout_with_covers(
    size: usize,
    covers: &[Edge],
    x_spacing: f64,
    y_spacing: f64,
    layout: PosetLayout,
) -> HashMap<ElementId, (f64, f64)> {
    let vertical = vertical_levels(size, covers);
    // Levels are stored doubled, so that centered heights are integers.
    // Longest-path heights are whole ranks; centered ones may be half-ranks.
    let (levels, level_step): (Vec<usize>, usize) = match layout.levels {
        LayoutLevels::LongestPath => (
            vertical.earliest.iter().map(|&height| 2 * height).collect(),
            2,
        ),
        LayoutLevels::Centered => (vertical.centered.clone(), 1),
    };

    let mut graph = match layout.ordering {
        LayoutOrdering::ElementOrder => {
            LayeredCoverGraph::without_bend_points(&levels, level_step, covers)
        }
        LayoutOrdering::Barycenter | LayoutOrdering::Refined => {
            LayeredCoverGraph::new(&levels, level_step, covers)
        }
    };
    let refined_defects = match layout.ordering {
        LayoutOrdering::ElementOrder => None,
        LayoutOrdering::Barycenter => {
            graph.order_by_barycenter();
            None
        }
        LayoutOrdering::Refined => Some(graph.reduce_crossings()),
    };

    match layout.coordinates {
        LayoutCoordinates::Grid => {
            // Crossing reduction is performed on a properized graph with
            // temporary dummy vertices, but TikZ emits straight cover segments.
            // Judge the real segments and retain the candidate with better
            // unrounded cover geometry. Styling, manual overrides, and optional
            // bends happen later.
            if let Some(defects) = refined_defects {
                let baseline_points = id_order_grid_points(&vertical.earliest);
                if straight_geometry_defects(covers, &baseline_points) < defects {
                    return id_order_coordinates(&vertical.earliest, x_spacing, y_spacing);
                }
            }
            graph.grid_coordinates(x_spacing, y_spacing)
        }
        LayoutCoordinates::Balanced => graph.balanced_coordinates(x_spacing, y_spacing),
        LayoutCoordinates::Aligned => graph.aligned_coordinates(x_spacing, y_spacing),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Lattice;
    use crate::poset::Poset;

    const X_SPACING: f64 = 1.8;
    const Y_SPACING: f64 = 1.4;

    fn all_layouts() -> Vec<PosetLayout> {
        let mut layouts = Vec::new();
        for levels in [LayoutLevels::LongestPath, LayoutLevels::Centered] {
            for ordering in [
                LayoutOrdering::ElementOrder,
                LayoutOrdering::Barycenter,
                LayoutOrdering::Refined,
            ] {
                for coordinates in [
                    LayoutCoordinates::Grid,
                    LayoutCoordinates::Balanced,
                    LayoutCoordinates::Aligned,
                ] {
                    layouts.push(PosetLayout {
                        levels,
                        ordering,
                        coordinates,
                    });
                }
            }
        }
        layouts
    }

    fn pentagon() -> Poset {
        Lattice::from_covers(
            ["0", "a", "b", "c", "1"],
            [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
        )
        .unwrap()
        .as_poset()
        .clone()
    }

    fn examples() -> Vec<(&'static str, Poset, bool)> {
        // (name, poset, whether it is graded)
        vec![
            ("empty", Poset::antichain(0), true),
            ("point", Poset::antichain(1), true),
            ("antichain", Poset::antichain(3), true),
            ("chain", Poset::chain(3), true),
            ("cube", Lattice::boolean(3).as_poset().clone(), true),
            ("pentagon", pentagon(), false),
            (
                "disjoint union",
                Poset::disjoint_union([Poset::chain(2), pentagon()]),
                false,
            ),
            (
                "Tr([3])",
                Lattice::chain(3).transfer_systems().as_poset().clone(),
                false,
            ),
            (
                "Tr(pentagon)",
                Lattice::from_covers(
                    ["0", "a", "b", "c", "1"],
                    [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
                )
                .unwrap()
                .transfer_systems()
                .as_poset()
                .clone(),
                false,
            ),
        ]
    }

    /// Every layout is a valid Hasse diagram: each element gets a finite
    /// position, each cover relation points strictly upwards, and elements
    /// at the same height are at least `x_spacing` apart.
    #[test]
    fn every_layout_draws_a_hasse_diagram() {
        for (name, poset, _) in examples() {
            let covers = poset.sorted_cover_relations();
            for layout in all_layouts() {
                let coords =
                    layout_with_covers(poset.size(), &covers, X_SPACING, Y_SPACING, layout);
                let context = format!("{name} with {layout:?}");
                assert_eq!(coords.len(), poset.size(), "{context}");
                for id in poset.ids() {
                    let (x, y) = coords[&id];
                    assert!(x.is_finite() && y.is_finite(), "{context}");
                }
                for edge in &covers {
                    assert!(coords[&edge.from].1 < coords[&edge.to].1, "{context}");
                }
                for a in poset.ids() {
                    for b in poset.ids().filter(|&b| b > a) {
                        let ((xa, ya), (xb, yb)) = (coords[&a], coords[&b]);
                        if (ya - yb).abs() < 1e-9 {
                            assert!((xa - xb).abs() >= X_SPACING - 1e-9, "{context}");
                        }
                    }
                }
            }
        }
    }

    /// On a graded poset, every layout draws each element at its rank, so
    /// every cover relation rises by exactly `y_spacing`.
    #[test]
    fn graded_posets_are_drawn_by_rank() {
        for (name, poset, graded) in examples() {
            if !graded {
                continue;
            }
            let covers = poset.sorted_cover_relations();
            for layout in all_layouts() {
                let coords =
                    layout_with_covers(poset.size(), &covers, X_SPACING, Y_SPACING, layout);
                for edge in &covers {
                    let rise = coords[&edge.to].1 - coords[&edge.from].1;
                    assert!((rise - Y_SPACING).abs() < 1e-9, "{name} with {layout:?}");
                }
                for id in poset.ids() {
                    if poset
                        .ids()
                        .all(|other| other == id || !poset.leq(other, id))
                    {
                        assert!(coords[&id].1.abs() < 1e-9, "{name} with {layout:?}");
                    }
                }
            }
        }
    }
}
