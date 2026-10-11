use super::{
    GridPoint, StraightGeometryDefects, straight_geometry_contribution, straight_geometry_defects,
};
use crate::poset::{Edge, ElementId};
use std::collections::HashMap;

/// A proper layered graph made by subdividing every Hasse edge at each
/// intervening layer. Vertices below `real_count` are poset elements; the
/// rest are temporary dummy vertices used only by crossing reduction.
///
/// Levels are doubled heights, so that centered heights are integers; a
/// vertex at level `l` is drawn at height `l / 2`. There is a layer at every
/// multiple of `level_step`, the spacing of the heights an element can have
/// (a whole rank, or half a rank for centered heights).
pub(super) struct LayeredCoverGraph {
    pub(super) real_count: usize,
    real_edges: Vec<Edge>,
    pub(super) levels: Vec<usize>,
    /// The index in `layers` of the layer containing each vertex.
    layer_of: Vec<usize>,
    pub(super) layers: Vec<Vec<usize>>,
    pub(super) incoming: Vec<Vec<usize>>,
    pub(super) outgoing: Vec<Vec<usize>>,
}

impl LayeredCoverGraph {
    pub(super) fn new(real_levels: &[usize], level_step: usize, covers: &[Edge]) -> Self {
        Self::build(real_levels, level_step, covers, true)
    }

    /// The same graph without dummy vertices: every cover is a single edge,
    /// possibly between non-adjacent layers. Layers are in element id order.
    pub(super) fn without_bend_points(
        real_levels: &[usize],
        level_step: usize,
        covers: &[Edge],
    ) -> Self {
        Self::build(real_levels, level_step, covers, false)
    }

    fn build(real_levels: &[usize], level_step: usize, covers: &[Edge], subdivide: bool) -> Self {
        let real_count = real_levels.len();
        debug_assert!(real_levels.iter().all(|level| level % level_step == 0));
        let layer_count = real_levels
            .iter()
            .max()
            .map_or(0, |maximum| maximum / level_step + 1);
        let mut graph = Self {
            real_count,
            real_edges: covers.to_vec(),
            levels: real_levels.to_vec(),
            layer_of: real_levels.iter().map(|level| level / level_step).collect(),
            layers: vec![Vec::new(); layer_count],
            incoming: vec![Vec::new(); real_count],
            outgoing: vec![Vec::new(); real_count],
        };

        for id in 0..real_count {
            graph.layers[graph.layer_of[id]].push(id);
        }

        for &edge in covers {
            let from_layer = graph.layer_of[edge.from];
            let to_layer = graph.layer_of[edge.to];
            debug_assert!(from_layer < to_layer);

            let mut previous = edge.from;
            for layer in (from_layer + 1..to_layer).filter(|_| subdivide) {
                let dummy = graph.levels.len();
                graph.levels.push(layer * level_step);
                graph.layer_of.push(layer);
                graph.layers[layer].push(dummy);
                graph.incoming.push(Vec::new());
                graph.outgoing.push(Vec::new());
                graph.add_edge(previous, dummy);
                previous = dummy;
            }
            graph.add_edge(previous, edge.to);
        }

        graph
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        debug_assert!(self.layer_of[from] < self.layer_of[to]);
        self.outgoing[from].push(to);
        self.incoming[to].push(from);
    }

    /// The textbook ordering step: barycenter sweeps, keeping the layer orders
    /// with the fewest crossings in the layered graph.
    pub(super) fn order_by_barycenter(&mut self) {
        if self.layers.len() < 2 {
            return;
        }

        let mut best_layers = self.layers.clone();
        let mut best_crossings = self.routed_crossings(&self.vertex_positions());
        for _ in 0..8 {
            let cycle_start = self.layers.clone();
            self.sweep_downward();
            self.remember_if_fewer_crossings(&mut best_layers, &mut best_crossings);
            self.sweep_upward();
            self.remember_if_fewer_crossings(&mut best_layers, &mut best_crossings);
            if self.layers == cycle_start {
                break;
            }
        }
        self.layers = best_layers;
    }

    fn remember_if_fewer_crossings(&self, best_layers: &mut Vec<Vec<usize>>, best: &mut u64) {
        let crossings = self.routed_crossings(&self.vertex_positions());
        if crossings < *best {
            *best = crossings;
            *best_layers = self.layers.clone();
        }
    }

    pub(super) fn reduce_crossings(&mut self) -> StraightGeometryDefects {
        if self.layers.len() < 2 {
            return self.straight_geometry_defects();
        }

        let mut best_layers = self.layers.clone();
        let mut best_score = self.ordering_score();
        for _ in 0..8 {
            let cycle_start = self.layers.clone();
            self.sweep_downward();
            self.remember_if_better(&mut best_layers, &mut best_score);
            self.sweep_upward();
            self.remember_if_better(&mut best_layers, &mut best_score);
            if self.layers == cycle_start {
                break;
            }
        }
        self.layers = best_layers;
        self.greedy_switch_adjacent()
    }

    /// Escapes barycenter ties and local plateaus by trying adjacent layer
    /// swaps against the exact emitted-edge score. Dummy-dummy swaps cannot
    /// immediately change real geometry, so only pairs containing a real
    /// vertex are considered.
    fn greedy_switch_adjacent(&mut self) -> StraightGeometryDefects {
        let positions = self.vertex_positions();
        let mut points = self.real_grid_points_with_positions(&positions);
        let mut current_geometry = straight_geometry_defects(&self.real_edges, &points);
        if current_geometry.is_clean() {
            return current_geometry;
        }
        for _ in 0..self.real_count.max(1) {
            let mut improved = false;
            for level in 0..self.layers.len() {
                for position in 0..self.layers[level].len().saturating_sub(1) {
                    let left = self.layers[level][position];
                    let right = self.layers[level][position + 1];
                    if left >= self.real_count && right >= self.real_count {
                        continue;
                    }

                    let (moved_storage, moved_count) =
                        match (left < self.real_count, right < self.real_count) {
                            (true, true) => ([left, right], 2),
                            (true, false) => ([left, left], 1),
                            (false, true) => ([right, right], 1),
                            (false, false) => unreachable!("dummy-dummy swaps are skipped"),
                        };
                    let moved_vertices = &moved_storage[..moved_count];
                    let old_contribution =
                        straight_geometry_contribution(&self.real_edges, &points, moved_vertices);
                    self.layers[level].swap(position, position + 1);
                    if left < self.real_count {
                        points[left].x += 2;
                    }
                    if right < self.real_count {
                        points[right].x -= 2;
                    }
                    let new_contribution =
                        straight_geometry_contribution(&self.real_edges, &points, moved_vertices);
                    let candidate_geometry =
                        current_geometry.replace_contribution(old_contribution, new_contribution);
                    if candidate_geometry < current_geometry {
                        current_geometry = candidate_geometry;
                        improved = true;
                        if current_geometry.is_clean() {
                            debug_assert_eq!(
                                current_geometry,
                                straight_geometry_defects(&self.real_edges, &points)
                            );
                            return current_geometry;
                        }
                    } else {
                        self.layers[level].swap(position, position + 1);
                        if left < self.real_count {
                            points[left].x -= 2;
                        }
                        if right < self.real_count {
                            points[right].x += 2;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        debug_assert_eq!(
            current_geometry,
            straight_geometry_defects(&self.real_edges, &points)
        );
        current_geometry
    }

    fn remember_if_better(
        &self,
        best_layers: &mut Vec<Vec<usize>>,
        best_score: &mut OrderingScore,
    ) {
        let score = self.ordering_score();
        if score < *best_score {
            *best_score = score;
            *best_layers = self.layers.clone();
        }
    }

    fn sweep_downward(&mut self) {
        let mut positions = self.vertex_positions();
        for level in 1..self.layers.len() {
            self.reorder_layer(level, level - 1, true, &mut positions);
        }
    }

    fn sweep_upward(&mut self) {
        let mut positions = self.vertex_positions();
        for level in (0..self.layers.len() - 1).rev() {
            self.reorder_layer(level, level + 1, false, &mut positions);
        }
    }

    fn reorder_layer(
        &mut self,
        level: usize,
        neighbor_level: usize,
        use_incoming: bool,
        positions: &mut [usize],
    ) {
        if self.layers[level].len() < 2 {
            return;
        }

        let layer_len = self.layers[level].len() as f64;
        let neighbor_len = self.layers[neighbor_level].len() as f64;
        let mut keyed: Vec<_> = self.layers[level]
            .iter()
            .copied()
            .enumerate()
            .map(|(old_position, vertex)| {
                let neighbors = if use_incoming {
                    &self.incoming[vertex]
                } else {
                    &self.outgoing[vertex]
                };
                let barycenter = if neighbors.is_empty() {
                    // A source or sink in a shorter disconnected component has
                    // no neighbor in this sweep direction. Preserve its rough
                    // horizontal position relative to the adjacent layer.
                    (old_position as f64 + 0.5) * neighbor_len / layer_len - 0.5
                } else {
                    neighbors
                        .iter()
                        .map(|&neighbor| positions[neighbor] as f64)
                        .sum::<f64>()
                        / neighbors.len() as f64
                };
                (vertex, barycenter, old_position)
            })
            .collect();

        keyed.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.0.cmp(&right.0))
        });
        self.layers[level] = keyed.into_iter().map(|(vertex, _, _)| vertex).collect();
        for (position, &vertex) in self.layers[level].iter().enumerate() {
            positions[vertex] = position;
        }
    }

    fn ordering_score(&self) -> OrderingScore {
        let positions = self.vertex_positions();
        let geometry = straight_geometry_defects(
            &self.real_edges,
            &self.real_grid_points_with_positions(&positions),
        );
        let routed_crossings = self.routed_crossings(&positions);
        let mut horizontal_length = 0u64;
        for level in 0..self.layers.len().saturating_sub(1) {
            let upper_width = self.layers[level + 1].len();
            for &source in &self.layers[level] {
                for &target in &self.outgoing[source] {
                    let source_x =
                        doubled_centered_slot(positions[source], self.layers[level].len());
                    let target_x = doubled_centered_slot(positions[target], upper_width);
                    horizontal_length =
                        horizontal_length.saturating_add(source_x.abs_diff(target_x));
                }
            }
        }

        OrderingScore {
            geometry,
            routed_crossings,
            horizontal_length,
        }
    }

    /// Counts pairs of crossing edges between adjacent layers.
    fn routed_crossings(&self, positions: &[usize]) -> u64 {
        let mut crossings = 0u64;
        for level in 0..self.layers.len().saturating_sub(1) {
            let mut targets_seen = FenwickTree::new(self.layers[level + 1].len());
            let mut seen_count = 0u64;
            for &source in &self.layers[level] {
                // Query the whole source group before inserting it, so edges
                // with a shared endpoint are not counted as crossings.
                for &target in &self.outgoing[source] {
                    crossings = crossings.saturating_add(
                        seen_count - targets_seen.prefix_sum(positions[target] + 1),
                    );
                }
                for &target in &self.outgoing[source] {
                    targets_seen.add(positions[target], 1);
                    seen_count += 1;
                }
            }
        }
        crossings
    }

    fn straight_geometry_defects(&self) -> StraightGeometryDefects {
        let positions = self.vertex_positions();
        straight_geometry_defects(
            &self.real_edges,
            &self.real_grid_points_with_positions(&positions),
        )
    }

    fn real_grid_points_with_positions(&self, positions: &[usize]) -> Vec<GridPoint> {
        (0..self.real_count)
            .map(|id| GridPoint {
                x: doubled_centered_slot(positions[id], self.layers[self.layer_of[id]].len()),
                y: self.levels[id] as i64,
            })
            .collect()
    }

    pub(super) fn vertex_positions(&self) -> Vec<usize> {
        let mut positions = vec![0usize; self.levels.len()];
        for layer in &self.layers {
            for (position, &vertex) in layer.iter().enumerate() {
                positions[vertex] = position;
            }
        }
        positions
    }

    /// Evenly spaced, centered layers.
    pub(super) fn grid_coordinates(
        &self,
        x_spacing: f64,
        y_spacing: f64,
    ) -> HashMap<ElementId, (f64, f64)> {
        let positions = self.vertex_positions();
        (0..self.real_count)
            .map(|id| {
                let width = self.layers[self.layer_of[id]].len().saturating_sub(1) as f64;
                let x = (positions[id] as f64 - width / 2.0) * x_spacing;
                let y = self.levels[id] as f64 * y_spacing / 2.0;
                (id, (x, y))
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct OrderingScore {
    geometry: StraightGeometryDefects,
    routed_crossings: u64,
    horizontal_length: u64,
}

pub(super) fn doubled_centered_slot(position: usize, layer_width: usize) -> i64 {
    2 * position as i64 - layer_width.saturating_sub(1) as i64
}

struct FenwickTree {
    entries: Vec<u64>,
}

impl FenwickTree {
    fn new(size: usize) -> Self {
        Self {
            entries: vec![0; size + 1],
        }
    }

    fn add(&mut self, index: usize, value: u64) {
        let mut index = index + 1;
        while index < self.entries.len() {
            self.entries[index] += value;
            index += index.isolate_lowest_one();
        }
    }

    /// Returns the sum over indices strictly below `end`.
    fn prefix_sum(&self, end: usize) -> u64 {
        let mut index = end;
        let mut sum = 0;
        while index > 0 {
            sum += self.entries[index];
            index &= index - 1;
        }
        sum
    }
}
