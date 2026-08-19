use crate::poset::{Edge, ElementId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::tikz) struct GridPoint {
    pub(in crate::tikz) x: i64,
    pub(in crate::tikz) y: i64,
}

#[derive(Debug, Clone, Copy)]
struct GridSegment {
    from: GridPoint,
    to: GridPoint,
    min_x: i64,
    max_x: i64,
    min_y: i64,
    max_y: i64,
}

impl GridSegment {
    fn new(from: GridPoint, to: GridPoint) -> Self {
        Self {
            from,
            to,
            min_x: from.x.min(to.x),
            max_x: from.x.max(to.x),
            min_y: from.y.min(to.y),
            max_y: from.y.max(to.y),
        }
    }

    fn bounding_box_overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }

    fn bounding_box_contains(self, point: GridPoint) -> bool {
        self.min_x <= point.x
            && point.x <= self.max_x
            && self.min_y <= point.y
            && point.y <= self.max_y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::tikz) struct StraightGeometryDefects {
    pub(in crate::tikz) proper_crossings: u64,
    pub(in crate::tikz) node_intersections: u64,
    pub(in crate::tikz) collinear_overlaps: u64,
}

impl StraightGeometryDefects {
    pub(in crate::tikz) fn is_clean(self) -> bool {
        self.proper_crossings == 0 && self.node_intersections == 0 && self.collinear_overlaps == 0
    }

    pub(in crate::tikz) fn replace_contribution(self, before: Self, after: Self) -> Self {
        debug_assert!(before.proper_crossings <= self.proper_crossings);
        debug_assert!(before.node_intersections <= self.node_intersections);
        debug_assert!(before.collinear_overlaps <= self.collinear_overlaps);
        Self {
            proper_crossings: self.proper_crossings - before.proper_crossings
                + after.proper_crossings,
            node_intersections: self.node_intersections - before.node_intersections
                + after.node_intersections,
            collinear_overlaps: self.collinear_overlaps - before.collinear_overlaps
                + after.collinear_overlaps,
        }
    }
}

fn edges_share_endpoint(left: Edge, right: Edge) -> bool {
    left.from == right.from || left.from == right.to || left.to == right.from || left.to == right.to
}

fn add_segment_pair_defects(
    defects: &mut StraightGeometryDefects,
    segment: GridSegment,
    other: GridSegment,
) {
    if !segment.bounding_box_overlaps(other) {
        return;
    }
    if grid_segments_properly_cross(segment.from, segment.to, other.from, other.to) {
        defects.proper_crossings += 1;
    } else if grid_segments_collinearly_overlap(segment.from, segment.to, other.from, other.to) {
        defects.collinear_overlaps += 1;
    }
}

fn segment_contains_unrelated_node(
    edge: Edge,
    segment: GridSegment,
    node: ElementId,
    point: GridPoint,
) -> bool {
    node != edge.from
        && node != edge.to
        && segment.bounding_box_contains(point)
        && grid_point_strictly_on_segment(segment.from, segment.to, point)
}

pub(in crate::tikz) fn straight_geometry_defects(
    edges: &[Edge],
    points: &[GridPoint],
) -> StraightGeometryDefects {
    let segments: Vec<_> = edges
        .iter()
        .map(|edge| GridSegment::new(points[edge.from], points[edge.to]))
        .collect();
    let mut defects = StraightGeometryDefects {
        proper_crossings: 0,
        node_intersections: 0,
        collinear_overlaps: 0,
    };
    for (index, &edge) in edges.iter().enumerate() {
        let segment = segments[index];
        for (offset, &other) in edges[index + 1..].iter().enumerate() {
            if edges_share_endpoint(edge, other) {
                continue;
            }
            let other_segment = segments[index + 1 + offset];
            add_segment_pair_defects(&mut defects, segment, other_segment);
        }
    }

    for (&edge, &segment) in edges.iter().zip(&segments) {
        for (id, &point) in points.iter().enumerate() {
            if segment_contains_unrelated_node(edge, segment, id, point) {
                defects.node_intersections += 1;
            }
        }
    }

    defects
}

/// Counts exactly those defects whose truth can change when `moved_vertices`
/// move horizontally. This lets adjacent switching update the global score
/// without rechecking pairs of unaffected edges and nodes.
pub(in crate::tikz) fn straight_geometry_contribution(
    edges: &[Edge],
    points: &[GridPoint],
    moved_vertices: &[ElementId],
) -> StraightGeometryDefects {
    let segments: Vec<_> = edges
        .iter()
        .map(|edge| GridSegment::new(points[edge.from], points[edge.to]))
        .collect();
    let affected_edges: Vec<_> = edges
        .iter()
        .map(|edge| {
            moved_vertices
                .iter()
                .any(|&vertex| edge.from == vertex || edge.to == vertex)
        })
        .collect();
    let mut defects = StraightGeometryDefects {
        proper_crossings: 0,
        node_intersections: 0,
        collinear_overlaps: 0,
    };

    for (index, &is_affected) in affected_edges.iter().enumerate() {
        if !is_affected {
            continue;
        }
        for other_index in 0..edges.len() {
            if other_index == index
                || (affected_edges[other_index] && other_index < index)
                || edges_share_endpoint(edges[index], edges[other_index])
            {
                continue;
            }
            add_segment_pair_defects(&mut defects, segments[index], segments[other_index]);
        }
    }

    for (index, (&edge, &segment)) in edges.iter().zip(&segments).enumerate() {
        if affected_edges[index] {
            for (node, &point) in points.iter().enumerate() {
                if segment_contains_unrelated_node(edge, segment, node, point) {
                    defects.node_intersections += 1;
                }
            }
        } else {
            for &node in moved_vertices {
                if segment_contains_unrelated_node(edge, segment, node, points[node]) {
                    defects.node_intersections += 1;
                }
            }
        }
    }

    defects
}

fn grid_segments_properly_cross(a: GridPoint, b: GridPoint, c: GridPoint, d: GridPoint) -> bool {
    let ab_c = grid_orientation(a, b, c);
    let ab_d = grid_orientation(a, b, d);
    let cd_a = grid_orientation(c, d, a);
    let cd_b = grid_orientation(c, d, b);
    have_opposite_signs(ab_c, ab_d) && have_opposite_signs(cd_a, cd_b)
}

fn grid_segments_collinearly_overlap(
    a: GridPoint,
    b: GridPoint,
    c: GridPoint,
    d: GridPoint,
) -> bool {
    if grid_orientation(a, b, c) != 0 || grid_orientation(a, b, d) != 0 {
        return false;
    }

    if (b.x - a.x).abs() >= (b.y - a.y).abs() {
        intervals_strictly_overlap(a.x, b.x, c.x, d.x)
    } else {
        intervals_strictly_overlap(a.y, b.y, c.y, d.y)
    }
}

fn intervals_strictly_overlap(a: i64, b: i64, c: i64, d: i64) -> bool {
    a.min(b).max(c.min(d)) < a.max(b).min(c.max(d))
}

fn grid_point_strictly_on_segment(a: GridPoint, b: GridPoint, point: GridPoint) -> bool {
    if grid_orientation(a, b, point) != 0 {
        return false;
    }
    let relative = (point.x - a.x, point.y - a.y);
    let segment = (b.x - a.x, b.y - a.y);
    let projection =
        relative.0 as i128 * segment.0 as i128 + relative.1 as i128 * segment.1 as i128;
    let length_squared =
        segment.0 as i128 * segment.0 as i128 + segment.1 as i128 * segment.1 as i128;
    projection > 0 && projection < length_squared
}

fn grid_orientation(a: GridPoint, b: GridPoint, c: GridPoint) -> i128 {
    (b.x - a.x) as i128 * (c.y - a.y) as i128 - (b.y - a.y) as i128 * (c.x - a.x) as i128
}

fn have_opposite_signs(left: i128, right: i128) -> bool {
    (left < 0 && right > 0) || (left > 0 && right < 0)
}
