//! Horizontal coordinate assignment for an ordered layered graph.

use super::layered::LayeredCoverGraph;
use crate::poset::ElementId;
use std::collections::{HashMap, HashSet};

impl LayeredCoverGraph {
    /// Chooses horizontal coordinates for the current layer orders.
    ///
    /// The order within each layer is kept. Adjacent vertices keep a minimum
    /// gap: one unit between two poset elements and half a unit when a dummy
    /// vertex is involved, so dummies reserve room for long edges without
    /// spreading out the visible elements. Starting from evenly spaced layers,
    /// alternating sweeps move every vertex towards the mean position of its
    /// neighbours, solving each layer exactly as a least-squares problem under
    /// the order and gap constraints.
    pub(super) fn balanced_coordinates(
        &self,
        x_spacing: f64,
        y_spacing: f64,
    ) -> HashMap<ElementId, (f64, f64)> {
        let mut x = vec![0.0; self.levels.len()];
        for layer in &self.layers {
            let mut position = 0.0;
            for (index, &vertex) in layer.iter().enumerate() {
                if index > 0 {
                    position += self.minimum_gap(layer[index - 1], vertex);
                }
                x[vertex] = position;
            }
            let shift = position / 2.0;
            for &vertex in layer {
                x[vertex] -= shift;
            }
        }

        let anchors = x.clone();
        for _ in 0..BALANCING_SWEEPS {
            for level in 0..self.layers.len() {
                self.place_layer(level, &mut x, &anchors);
            }
            for level in (0..self.layers.len()).rev() {
                self.place_layer(level, &mut x, &anchors);
            }
        }

        self.remove_lean(&mut x);
        self.centered_coordinates(&x, x_spacing, y_spacing)
    }

    /// Chooses horizontal coordinates for the current layer orders by the
    /// method of Brandes and Köpf, "Fast and simple horizontal coordinate
    /// assignment" (Graph Drawing 2001).
    ///
    /// Each of four passes aligns every vertex with a median neighbour in the
    /// layer above or below it, scanning the layers left to right or right to
    /// left, so that aligned vertices form vertical blocks. Blocks are then
    /// packed as tightly as the minimum gaps allow. The final position of
    /// each vertex is the average of its two middle positions among the four
    /// passes, which balances their opposite leanings; any lean that remains
    /// is sheared away as for [`Self::balanced_coordinates`].
    pub(super) fn aligned_coordinates(
        &self,
        x_spacing: f64,
        y_spacing: f64,
    ) -> HashMap<ElementId, (f64, f64)> {
        // An empty layer has no edges across it, so skipping it changes no
        // adjacency.
        let layers: Vec<&[usize]> = self
            .layers
            .iter()
            .filter(|layer| !layer.is_empty())
            .map(Vec::as_slice)
            .collect();
        let conflicts = self.type_one_conflicts(&layers);

        let candidates: Vec<(bool, Vec<f64>)> =
            [(false, false), (false, true), (true, false), (true, true)]
                .into_iter()
                .map(|(from_top, from_right)| {
                    (
                        from_right,
                        self.aligned_candidate(&layers, &conflicts, from_top, from_right),
                    )
                })
                .collect();

        // Shift every candidate to the extent of the narrowest one: those
        // packed from the left share its left edge, those packed from the
        // right its right edge.
        let extent = |x: &[f64]| {
            x.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
                    (low.min(value), high.max(value))
                })
        };
        let (narrowest_low, narrowest_high) = candidates
            .iter()
            .map(|(_, x)| extent(x))
            .min_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
            .unwrap_or((0.0, 0.0));
        let shifted: Vec<Vec<f64>> = candidates
            .into_iter()
            .map(|(from_right, x)| {
                let (low, high) = extent(&x);
                let shift = if from_right {
                    narrowest_high - high
                } else {
                    narrowest_low - low
                };
                x.into_iter().map(|value| value + shift).collect()
            })
            .collect();

        // Averaging the middle two of four order-preserving placements keeps
        // the order and the minimum gaps.
        let mut x: Vec<f64> = (0..self.levels.len())
            .map(|vertex| {
                let mut values = shifted.iter().map(|x| x[vertex]).collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                (values[1] + values[2]) / 2.0
            })
            .collect();
        // A shear keeps straight chains straight and parallel edges parallel.
        self.remove_lean(&mut x);
        self.centered_coordinates(&x, x_spacing, y_spacing)
    }

    /// One Brandes–Köpf pass. Layers are scanned from the top or the bottom,
    /// and each layer from the right or the left; the result is expressed in
    /// the usual left-to-right coordinates.
    fn aligned_candidate(
        &self,
        layers: &[&[usize]],
        conflicts: &HashSet<(usize, usize)>,
        from_top: bool,
        from_right: bool,
    ) -> Vec<f64> {
        let count = self.levels.len();
        let mut ordered: Vec<Vec<usize>> = layers
            .iter()
            .map(|layer| {
                let mut layer = layer.to_vec();
                if from_right {
                    layer.reverse();
                }
                layer
            })
            .collect();
        if from_top {
            ordered.reverse();
        }
        let mut layer_of = vec![0usize; count];
        let mut position = vec![0usize; count];
        for (index, layer) in ordered.iter().enumerate() {
            for (slot, &vertex) in layer.iter().enumerate() {
                layer_of[vertex] = index;
                position[vertex] = slot;
            }
        }

        // Vertical alignment: `root[v]` is the first vertex of v's block, and
        // `align` links each block into a cycle.
        let mut root: Vec<usize> = (0..count).collect();
        let mut align: Vec<usize> = (0..count).collect();
        for (index, layer) in ordered.iter().enumerate().skip(1) {
            let mut last_aligned: Option<usize> = None;
            for &vertex in layer {
                let neighbours = if from_top {
                    &self.outgoing[vertex]
                } else {
                    &self.incoming[vertex]
                };
                let mut neighbours: Vec<usize> = neighbours
                    .iter()
                    .copied()
                    .filter(|&neighbour| layer_of[neighbour] + 1 == index)
                    .collect();
                if neighbours.is_empty() {
                    continue;
                }
                neighbours.sort_by_key(|&neighbour| position[neighbour]);
                let degree = neighbours.len();
                for median in [(degree - 1) / 2, degree / 2] {
                    let neighbour = neighbours[median];
                    if align[vertex] == vertex
                        && !conflicts.contains(&segment(neighbour, vertex))
                        && last_aligned.is_none_or(|last| last < position[neighbour])
                    {
                        align[neighbour] = vertex;
                        root[vertex] = root[neighbour];
                        align[vertex] = root[vertex];
                        last_aligned = Some(position[neighbour]);
                    }
                }
            }
        }

        // Horizontal compaction on the graph of blocks: an arc from the
        // block of each vertex to the block of its right neighbour, weighted
        // by their minimum gap. Place every block as far left as the arcs
        // allow, then move blocks right towards their successors to close
        // gaps that nothing requires.
        let mut arcs: HashMap<(usize, usize), f64> = HashMap::new();
        for layer in &ordered {
            for pair in layer.windows(2) {
                let gap = self.minimum_gap(pair[0], pair[1]);
                let weight = arcs.entry((root[pair[0]], root[pair[1]])).or_insert(gap);
                *weight = weight.max(gap);
            }
        }
        let mut successors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
        let mut predecessors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
        let mut pending = vec![0usize; count];
        for (&(left, right), &gap) in &arcs {
            successors[left].push((right, gap));
            predecessors[right].push((left, gap));
            pending[right] += 1;
        }
        let mut order = Vec::new();
        let mut ready: Vec<usize> = (0..count)
            .filter(|&vertex| root[vertex] == vertex && pending[vertex] == 0)
            .collect();
        while let Some(block) = ready.pop() {
            order.push(block);
            for &(next, _) in &successors[block] {
                pending[next] -= 1;
                if pending[next] == 0 {
                    ready.push(next);
                }
            }
        }
        debug_assert_eq!(
            order.len(),
            (0..count).filter(|&vertex| root[vertex] == vertex).count(),
            "the block graph of a valid alignment is acyclic"
        );

        let mut block_x = vec![0.0; count];
        for &block in &order {
            block_x[block] = predecessors[block]
                .iter()
                .map(|&(left, gap)| block_x[left] + gap)
                .fold(0.0, f64::max);
        }
        for &block in order.iter().rev() {
            let limit = successors[block]
                .iter()
                .map(|&(right, gap)| block_x[right] - gap)
                .fold(f64::INFINITY, f64::min);
            if limit.is_finite() && limit > block_x[block] {
                block_x[block] = limit;
            }
        }

        let sign = if from_right { -1.0 } else { 1.0 };
        (0..count)
            .map(|vertex| sign * block_x[root[vertex]])
            .collect()
    }

    /// Marks the segments that must not be aligned because they cross an
    /// inner segment (an edge between two dummy vertices). Keeping inner
    /// segments vertical keeps long edges straight.
    fn type_one_conflicts(&self, layers: &[&[usize]]) -> HashSet<(usize, usize)> {
        let mut layer_of = vec![0usize; self.levels.len()];
        let mut position = vec![0usize; self.levels.len()];
        for (index, layer) in layers.iter().enumerate() {
            for (slot, &vertex) in layer.iter().enumerate() {
                layer_of[vertex] = index;
                position[vertex] = slot;
            }
        }
        let layer_of = &layer_of;
        let is_dummy = |vertex: usize| vertex >= self.real_count;
        let lower_neighbours = |vertex: usize, index: usize| {
            self.incoming[vertex]
                .iter()
                .copied()
                .filter(move |&neighbour| layer_of[neighbour] + 1 == index)
        };

        let mut conflicts = HashSet::new();
        for index in 1..layers.len() {
            let (lower, upper) = (layers[index - 1], layers[index]);
            let mut low = 0;
            let mut scanned = 0;
            for (slot, &vertex) in upper.iter().enumerate() {
                let inner = is_dummy(vertex)
                    .then(|| lower_neighbours(vertex, index).find(|&neighbour| is_dummy(neighbour)))
                    .flatten();
                if slot + 1 < upper.len() && inner.is_none() {
                    continue;
                }
                let high = inner.map_or(lower.len() - 1, |neighbour| position[neighbour]);
                for &between in &upper[scanned..=slot] {
                    for neighbour in lower_neighbours(between, index) {
                        let crosses = position[neighbour] < low || position[neighbour] > high;
                        if crosses && !(is_dummy(between) && is_dummy(neighbour)) {
                            conflicts.insert(segment(neighbour, between));
                        }
                    }
                }
                scanned = slot + 1;
                low = high;
            }
        }
        conflicts
    }

    /// Straightens a drawing that leans to one side by applying the shear
    /// `x -> x - k * level`. A shear is affine, so it changes neither the
    /// order within layers nor which straight edges cross or overlap. When the
    /// poset has a unique bottom and top, `k` puts the top directly above the
    /// bottom; otherwise it is the least-squares slope of `x` against level.
    pub(super) fn remove_lean(&self, x: &mut [f64]) {
        let real = 0..self.real_count;
        let level = |id: usize| self.levels[id] as f64;
        let lowest = real.clone().map(|id| self.levels[id]).min();
        let highest = real.clone().map(|id| self.levels[id]).max();
        let (Some(lowest), Some(highest)) = (lowest, highest) else {
            return;
        };
        if lowest == highest {
            return;
        }
        let bottoms = real
            .clone()
            .filter(|&id| self.levels[id] == lowest)
            .collect::<Vec<_>>();
        let tops = real
            .clone()
            .filter(|&id| self.levels[id] == highest)
            .collect::<Vec<_>>();
        let slope = if let ([bottom], [top]) = (bottoms.as_slice(), tops.as_slice()) {
            (x[*top] - x[*bottom]) / (level(*top) - level(*bottom))
        } else {
            let count = self.real_count as f64;
            let mean_level = real.clone().map(level).sum::<f64>() / count;
            let mean_x = real.clone().map(|id| x[id]).sum::<f64>() / count;
            let covariance = real
                .clone()
                .map(|id| (level(id) - mean_level) * (x[id] - mean_x))
                .sum::<f64>();
            let variance = real
                .clone()
                .map(|id| (level(id) - mean_level).powi(2))
                .sum::<f64>();
            covariance / variance
        };
        for (vertex, value) in x.iter_mut().enumerate() {
            *value -= slope * level(vertex);
        }
    }

    fn minimum_gap(&self, left: usize, right: usize) -> f64 {
        if left < self.real_count && right < self.real_count {
            1.0
        } else {
            0.5
        }
    }

    /// Moves one layer's vertices to the least-squares closest positions to
    /// their neighbours' mean positions that keep the layer's order and gaps.
    fn place_layer(&self, level: usize, x: &mut [f64], anchors: &[f64]) {
        let layer = &self.layers[level];
        if layer.is_empty() {
            return;
        }
        // Writing x_i = y_i + offset_i, where offset_i accumulates the minimum
        // gaps, turns the gap constraints into y_0 <= y_1 <= ..., so the
        // weighted least-squares fit is an isotonic regression.
        let mut offsets = Vec::with_capacity(layer.len());
        let mut offset = 0.0;
        for (index, &vertex) in layer.iter().enumerate() {
            if index > 0 {
                offset += self.minimum_gap(layer[index - 1], vertex);
            }
            offsets.push(offset);
        }
        let targets = layer
            .iter()
            .zip(&offsets)
            .map(|(&vertex, &offset)| {
                let neighbours = self.incoming[vertex].iter().chain(&self.outgoing[vertex]);
                let count = self.incoming[vertex].len() + self.outgoing[vertex].len();
                // Each vertex is pulled towards its neighbours and, with weight
                // ANCHOR_WEIGHT, towards its place in the centred layer, which
                // keeps the drawing upright and balanced.
                let target = (neighbours.map(|&neighbour| x[neighbour]).sum::<f64>()
                    + ANCHOR_WEIGHT * anchors[vertex])
                    / (count as f64 + ANCHOR_WEIGHT);
                (target - offset, 1.0)
            })
            .collect::<Vec<_>>();
        for ((&vertex, &offset), y) in layer.iter().zip(&offsets).zip(isotonic_fit(&targets)) {
            x[vertex] = y + offset;
        }
    }

    /// Scales the poset elements' positions and centres their bounding box.
    pub(super) fn centered_coordinates(
        &self,
        x: &[f64],
        x_spacing: f64,
        y_spacing: f64,
    ) -> HashMap<ElementId, (f64, f64)> {
        let real_xs = || x[..self.real_count].iter().copied();
        let centre = (real_xs().fold(f64::INFINITY, f64::min)
            + real_xs().fold(f64::NEG_INFINITY, f64::max))
            / 2.0;
        (0..self.real_count)
            .map(|id| {
                let y = self.levels[id] as f64 * y_spacing / 2.0;
                (id, ((x[id] - centre) * x_spacing, y))
            })
            .collect()
    }
}

const BALANCING_SWEEPS: usize = 30;
const ANCHOR_WEIGHT: f64 = 1.0;

/// An edge as an unordered pair of vertices.
fn segment(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Weighted least-squares nondecreasing fit (pool adjacent violators).
pub(super) fn isotonic_fit(targets: &[(f64, f64)]) -> Vec<f64> {
    // Each block is (weighted mean, total weight, number of entries).
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(targets.len());
    for &(value, weight) in targets {
        blocks.push((value, weight, 1));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 > blocks[blocks.len() - 1].0 {
            let (right_mean, right_weight, right_count) = blocks.pop().expect("two blocks");
            let left = blocks.last_mut().expect("two blocks");
            let weight = left.1 + right_weight;
            left.0 = (left.0 * left.1 + right_mean * right_weight) / weight;
            left.1 = weight;
            left.2 += right_count;
        }
    }
    blocks
        .into_iter()
        .flat_map(|(mean, _, count)| std::iter::repeat_n(mean, count))
        .collect()
}
