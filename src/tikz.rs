use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};
use crate::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

pub trait ToTikz {
    fn to_tikz(&self) -> TikzPicture;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TikzOptions {
    entries: Vec<String>,
}

impl TikzOptions {
    pub fn new(entries: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }

    pub fn push(&mut self, entry: impl Into<String>) {
        self.entries.push(entry.into());
    }

    fn has_bend(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.starts_with("bend left") || entry.starts_with("bend right"))
    }

    fn render_brackets(&self) -> String {
        if self.entries.is_empty() {
            String::new()
        } else {
            format!("[{}]", self.entries.join(", "))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TikzStyle {
    pub name: String,
    pub options: TikzOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TikzLabel(String);

impl TikzLabel {
    pub fn escaped(label: impl AsRef<str>) -> Self {
        Self(escape_tikz(label.as_ref()))
    }

    pub fn raw(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    fn render(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TikzNode {
    pub name: String,
    pub at: (f64, f64),
    pub label: TikzLabel,
    pub options: TikzOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TikzPath {
    pub from: String,
    pub to: String,
    pub options: TikzOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TikzScope {
    pub options: TikzOptions,
    pub picture: TikzPicture,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TikzPicture {
    pub options: TikzOptions,
    pub styles: Vec<TikzStyle>,
    pub nodes: Vec<TikzNode>,
    pub paths: Vec<TikzPath>,
    pub scopes: Vec<TikzScope>,
    pub raw: Vec<String>,
}

impl TikzPicture {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: TikzOptions) -> Self {
        Self {
            options,
            ..Self::default()
        }
    }

    pub fn push_raw(&mut self, raw: impl Into<String>) {
        self.raw.push(raw.into());
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "\\begin{{tikzpicture}}{}\n",
            self.options.render_brackets()
        ));
        self.render_body_into(&mut out, 0);
        out.push_str("\\end{tikzpicture}\n");
        out
    }

    fn render_body_into(&self, out: &mut String, indent: usize) {
        let pad = " ".repeat(indent);
        for style in &self.styles {
            out.push_str(&format!(
                "{pad}\\tikzset{{{} /.style = {}}}\n",
                style.name,
                style.options.render_brackets()
            ));
        }
        for raw in &self.raw {
            out.push_str(&pad);
            out.push_str(raw);
            if !raw.ends_with('\n') {
                out.push('\n');
            }
        }
        for node in &self.nodes {
            out.push_str(&format!(
                "{pad}\\node{} ({}) at ({:.3},{:.3}) {{{}}};\n",
                node.options.render_brackets(),
                node.name,
                node.at.0,
                node.at.1,
                node.label.render()
            ));
        }
        for path in &self.paths {
            if path.options.has_bend() {
                out.push_str(&format!(
                    "{pad}\\draw ({}) to{} ({});\n",
                    path.from,
                    path.options.render_brackets(),
                    path.to
                ));
            } else {
                out.push_str(&format!(
                    "{pad}\\draw{} ({}) -- ({});\n",
                    path.options.render_brackets(),
                    path.from,
                    path.to
                ));
            }
        }
        for scope in &self.scopes {
            out.push_str(&format!(
                "{pad}\\begin{{scope}}{}\n",
                scope.options.render_brackets()
            ));
            scope.picture.render_body_into(out, indent + 2);
            out.push_str(&format!("{pad}\\end{{scope}}\n"));
        }
    }
}

impl fmt::Display for TikzPicture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[derive(Debug, Clone)]
pub struct PosetTikzOptions {
    pub x_spacing: f64,
    pub y_spacing: f64,
    pub node_options: TikzOptions,
    pub edge_options: TikzOptions,
    pub full_relations: bool,
    pub bend_colinear_edges: bool,
    pub bend_angle: f64,
    pub colinear_tolerance: f64,
    pub coordinate_overrides: HashMap<ElementId, (f64, f64)>,
}

impl Default for PosetTikzOptions {
    fn default() -> Self {
        Self {
            x_spacing: 1.8,
            y_spacing: 1.4,
            node_options: TikzOptions::new(["circle", "draw", "inner sep=1.5pt"]),
            edge_options: TikzOptions::default(),
            full_relations: false,
            bend_colinear_edges: true,
            bend_angle: 18.0,
            colinear_tolerance: 1e-6,
            coordinate_overrides: HashMap::new(),
        }
    }
}

pub fn poset_to_tikz_with<A, F>(
    poset: &Poset<A>,
    options: &PosetTikzOptions,
    mut label: F,
) -> TikzPicture
where
    F: FnMut(ElementId, &A) -> TikzLabel,
{
    let mut coords = ranked_layout(poset, options.x_spacing, options.y_spacing);
    for (&id, &coordinate) in &options.coordinate_overrides {
        if id < poset.size() {
            coords.insert(id, coordinate);
        }
    }
    let mut picture = TikzPicture::new();
    let edges: Vec<Edge> = if options.full_relations {
        poset.proper_relations_iter().collect()
    } else {
        let mut covers: Vec<_> = poset.cover_relations().into_iter().collect();
        covers.sort();
        covers
    };
    let bend_edges = if options.bend_colinear_edges {
        edge_bend_decisions(&edges, &coords, options.colinear_tolerance)
    } else {
        vec![false; edges.len()]
    };

    for (index, edge) in edges.iter().copied().enumerate() {
        let mut edge_options = options.edge_options.clone();
        if bend_edges[index] {
            let direction = if index % 2 == 0 { "left" } else { "right" };
            edge_options.push(format!("bend {direction}={}", options.bend_angle));
        }
        picture.paths.push(TikzPath {
            from: node_name(edge.from),
            to: node_name(edge.to),
            options: edge_options,
        });
    }

    for (id, element) in poset.elements().iter().enumerate() {
        picture.nodes.push(TikzNode {
            name: node_name(id),
            at: coords[&id],
            label: label(id, element),
            options: options.node_options.clone(),
        });
    }

    picture
}

impl<A: fmt::Display> ToTikz for Poset<A> {
    fn to_tikz(&self) -> TikzPicture {
        poset_to_tikz_with(self, &PosetTikzOptions::default(), |_id, element| {
            TikzLabel::escaped(element.to_string())
        })
    }
}

impl<A: fmt::Display> ToTikz for Lattice<A> {
    fn to_tikz(&self) -> TikzPicture {
        self.as_poset().to_tikz()
    }
}

pub fn transfer_system_lattice_to_tikz<A>(lattice: &TransferLattice<A>) -> TikzPicture {
    transfer_system_order_to_tikz(lattice.as_poset())
}

pub fn transfer_system_order_to_tikz<A>(order: &TransferPoset<A>) -> TikzPicture {
    let options = PosetTikzOptions {
        node_options: TikzOptions::new(["draw", "inner sep=2pt"]),
        x_spacing: 3.4,
        y_spacing: 2.6,
        ..PosetTikzOptions::default()
    };
    poset_to_tikz_with(order, &options, |_node_id, system| {
        TikzLabel::raw(small_transfer_system_picture(system))
    })
}

pub fn escape_tikz(input: &str) -> String {
    let mut escaped = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\textbackslash{}"),
            '{' => escaped.push_str("\\{"),
            '}' => escaped.push_str("\\}"),
            '$' => escaped.push_str("\\$"),
            '&' => escaped.push_str("\\&"),
            '%' => escaped.push_str("\\%"),
            '#' => escaped.push_str("\\#"),
            '_' => escaped.push_str("\\_"),
            '^' => escaped.push_str("\\^{}"),
            '~' => escaped.push_str("\\~{}"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn ranked_layout<A>(
    poset: &Poset<A>,
    x_spacing: f64,
    y_spacing: f64,
) -> HashMap<ElementId, (f64, f64)> {
    let covers = poset.cover_relations();
    let mut memo = HashMap::new();
    for id in 0..poset.size() {
        rank_of(id, &covers, &mut memo);
    }

    let mut ranks: BTreeMap<usize, Vec<ElementId>> = BTreeMap::new();
    for (id, rank) in memo {
        ranks.entry(rank).or_default().push(id);
    }

    let mut coords = HashMap::new();
    for (rank, mut ids) in ranks {
        ids.sort_unstable();
        let width = ids.len().saturating_sub(1) as f64;
        for (slot, id) in ids.into_iter().enumerate() {
            coords.insert(
                id,
                (
                    (slot as f64 - width / 2.0) * x_spacing,
                    rank as f64 * y_spacing,
                ),
            );
        }
    }
    coords
}

fn rank_of(id: ElementId, covers: &HashSet<Edge>, memo: &mut HashMap<ElementId, usize>) -> usize {
    if let Some(&rank) = memo.get(&id) {
        return rank;
    }
    let rank = covers
        .iter()
        .filter(|edge| edge.to == id)
        .map(|edge| rank_of(edge.from, covers, memo) + 1)
        .max()
        .unwrap_or(0);
    memo.insert(id, rank);
    rank
}

fn edge_bend_decisions(
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

fn node_name(id: ElementId) -> String {
    format!("p{id}")
}

fn small_transfer_system_picture<A>(system: &TransferSystem<A>) -> String {
    let coords = ranked_layout(system.lattice().as_poset(), 1.0, 0.9);
    let mut out = String::from("\\tikz[scale=.35, baseline=-.5ex]{");
    for edge in system.lattice().as_poset().proper_relations_iter() {
        let from = coords[&edge.from];
        let to = coords[&edge.to];
        let highlighted = system.contains_edge(edge);
        let style = if highlighted {
            "line width=.75pt, draw=blue"
        } else {
            "draw=black!30"
        };
        out.push_str(&format!(
            "\\draw[{style}] ({:.3},{:.3}) -- ({:.3},{:.3});",
            from.0, from.1, to.0, to.1
        ));
    }
    for id in 0..system.lattice().size() {
        let at = coords[&id];
        out.push_str(&format!(
            "\\filldraw[fill=white, draw=black] ({:.3},{:.3}) circle (.055);",
            at.0, at.1
        ));
    }
    out.push('}');
    out
}
