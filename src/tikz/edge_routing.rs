use crate::poset::{Edge, ElementId};
use std::collections::HashMap;

pub(super) fn edge_bend_decisions(
    edges: &[Edge],
    coords: &HashMap<ElementId, (f64, f64)>,
    tolerance: f64,
) -> Vec<bool> {
    edges
        .iter()
        .copied()
        .enumerate()
        .map(|(index, edge)| {
            let from = coords[&edge.from];
            let to = coords[&edge.to];
            let blocked_by_node = coords.iter().any(|(&id, &point)| {
                id != edge.from
                    && id != edge.to
                    && point_strictly_on_segment(from, to, point, tolerance)
            });
            let has_preferred_overlapping_edge =
                edges
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(other_index, other)| {
                        other_index != index
                            && preferred_overlapping_edge(
                                edge,
                                index,
                                other,
                                other_index,
                                coords,
                                tolerance,
                            )
                    });
            blocked_by_node || has_preferred_overlapping_edge
        })
        .collect()
}

fn preferred_overlapping_edge(
    edge: Edge,
    edge_index: usize,
    other: Edge,
    other_index: usize,
    coords: &HashMap<ElementId, (f64, f64)>,
    tolerance: f64,
) -> bool {
    let from = coords[&edge.from];
    let to = coords[&edge.to];
    let other_from = coords[&other.from];
    let other_to = coords[&other.to];
    if !segments_colinear_overlap(from, to, other_from, other_to, tolerance) {
        return false;
    }

    let length = edge_length_squared(from, to);
    let other_length = edge_length_squared(other_from, other_to);
    other_length + tolerance < length
        || ((other_length - length).abs() <= tolerance && other_index < edge_index)
}

fn edge_length_squared(from: (f64, f64), to: (f64, f64)) -> f64 {
    let segment = (to.0 - from.0, to.1 - from.1);
    dot(segment, segment)
}

fn point_strictly_on_segment(
    from: (f64, f64),
    to: (f64, f64),
    point: (f64, f64),
    tolerance: f64,
) -> bool {
    let segment = (to.0 - from.0, to.1 - from.1);
    let relative = (point.0 - from.0, point.1 - from.1);
    let length_squared = dot(segment, segment);
    if length_squared <= tolerance * tolerance {
        return false;
    }

    let distance_from_line = cross(segment, relative).abs() / length_squared.sqrt();
    if distance_from_line > tolerance {
        return false;
    }

    let projection = dot(relative, segment);
    projection > tolerance && projection < length_squared - tolerance
}

fn segments_colinear_overlap(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    d: (f64, f64),
    tolerance: f64,
) -> bool {
    let segment = (b.0 - a.0, b.1 - a.1);
    let length_squared = dot(segment, segment);
    if length_squared <= tolerance * tolerance {
        return false;
    }
    let line_length = length_squared.sqrt();
    if cross(segment, (c.0 - a.0, c.1 - a.1)).abs() / line_length > tolerance
        || cross(segment, (d.0 - a.0, d.1 - a.1)).abs() / line_length > tolerance
    {
        return false;
    }

    let c_projection = dot((c.0 - a.0, c.1 - a.1), segment) / length_squared;
    let d_projection = dot((d.0 - a.0, d.1 - a.1), segment) / length_squared;
    let other_start = c_projection.min(d_projection);
    let other_end = c_projection.max(d_projection);
    let overlap_start = other_start.max(0.0);
    let overlap_end = other_end.min(1.0);
    overlap_end - overlap_start > tolerance / line_length
}

fn dot(left: (f64, f64), right: (f64, f64)) -> f64 {
    left.0 * right.0 + left.1 * right.1
}

fn cross(left: (f64, f64), right: (f64, f64)) -> f64 {
    left.0 * right.1 - left.1 * right.0
}
