//! TikZ rendering for finite posets, lattices, and transfer-system diagrams.
//!
//! The module provides a small typed TikZ abstract syntax tree together with
//! convenience renderers for Hasse diagrams.  The default layout ranks elements
//! by distance from minimal elements and draws cover relations unless full
//! relations are requested.

use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};
use crate::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// A type that can be rendered as a TikZ picture.
pub trait ToTikz {
    /// Renders `self` as a [`TikzPicture`].
    fn to_tikz(&self) -> TikzPicture;
}

/// A comma-separated list of TikZ options.
///
/// The same representation is used for picture options, node options, path
/// options, and style bodies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TikzOptions {
    entries: Vec<String>,
}

impl TikzOptions {
    /// Constructs a TikZ option list from strings.
    pub fn new(entries: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }

    /// Appends one option.
    pub fn push(&mut self, entry: impl Into<String>) {
        self.entries.push(entry.into());
    }

    /// Appends several options.
    pub fn extend(&mut self, entries: impl IntoIterator<Item = impl Into<String>>) {
        self.entries.extend(entries.into_iter().map(Into::into));
    }

    /// Returns whether the option list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates over the option entries.
    pub fn iter(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.iter().map(String::as_str)
    }

    /// Renders the comma-separated option body, without brackets or braces.
    pub fn render_inner(&self) -> String {
        self.entries.join(", ")
    }

    /// Renders the options in TikZ square-bracket syntax.
    ///
    /// Empty option lists render as the empty string.
    pub fn render_brackets(&self) -> String {
        if self.is_empty() {
            String::new()
        } else {
            format!("[{}]", self.render_inner())
        }
    }

    /// Renders the options in TikZ brace syntax.
    pub fn render_braces(&self) -> String {
        format!("{{{}}}", self.render_inner())
    }
}

/// A named TikZ style declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TikzStyle {
    /// The style name.
    pub name: String,
    /// The style body.
    pub options: TikzOptions,
}

/// Text to place inside a TikZ node.
///
/// Labels can either be escaped plain text or raw TeX/TikZ markup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TikzLabel(String);

impl TikzLabel {
    /// Constructs a label by escaping TeX special characters.
    pub fn escaped(label: impl AsRef<str>) -> Self {
        Self(escape_tikz(label.as_ref()))
    }

    /// Constructs a label from raw TeX/TikZ markup.
    pub fn raw(label: impl Into<String>) -> Self {
        Self(label.into())
    }

    fn render(&self) -> &str {
        &self.0
    }
}

/// A TikZ coordinate.
#[derive(Debug, Clone, PartialEq)]
pub enum TikzCoord {
    /// A numeric point `(x, y)`.
    Point(f64, f64),
    /// A reference to a named TikZ coordinate or node.
    Named(String),
    /// Raw coordinate syntax, inserted without modification.
    Raw(String),
}

impl TikzCoord {
    /// Constructs a numeric point.
    pub fn point(x: f64, y: f64) -> Self {
        Self::Point(x, y)
    }

    /// Constructs a coordinate reference by name.
    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }

    /// Constructs a raw TikZ coordinate.
    pub fn raw(coord: impl Into<String>) -> Self {
        Self::Raw(coord.into())
    }

    fn render(&self) -> String {
        match self {
            TikzCoord::Point(x, y) => format!("({x:.3},{y:.3})"),
            TikzCoord::Named(name) => format!("({name})"),
            TikzCoord::Raw(coord) => coord.clone(),
        }
    }
}

impl From<(f64, f64)> for TikzCoord {
    fn from((x, y): (f64, f64)) -> Self {
        Self::Point(x, y)
    }
}

/// A TikZ node command.
#[derive(Debug, Clone, PartialEq)]
pub struct TikzNode {
    /// Optional node name.
    pub name: Option<String>,
    /// The node coordinate.
    pub at: TikzCoord,
    /// The node label.
    pub label: TikzLabel,
    /// TikZ node options.
    pub options: TikzOptions,
}

/// The syntax used to connect two coordinates in a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TikzPathOperation {
    /// A straight `--` path.
    Line,
    /// A TikZ `to` path, useful for bends.
    To,
}

/// A TikZ path between two coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct TikzPath {
    /// The starting coordinate.
    pub from: TikzCoord,
    /// The ending coordinate.
    pub to: TikzCoord,
    /// The path operation.
    pub operation: TikzPathOperation,
    /// TikZ path options.
    pub options: TikzOptions,
}

/// The drawing command used for a circle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TikzDrawCommand {
    /// A `\draw` command.
    Draw,
    /// A `\fill` command.
    Fill,
    /// A `\filldraw` command.
    FillDraw,
}

impl TikzDrawCommand {
    fn render(self) -> &'static str {
        match self {
            TikzDrawCommand::Draw => "draw",
            TikzDrawCommand::Fill => "fill",
            TikzDrawCommand::FillDraw => "filldraw",
        }
    }
}

/// A TikZ circle command.
#[derive(Debug, Clone, PartialEq)]
pub struct TikzCircle {
    /// The center coordinate.
    pub center: TikzCoord,
    /// The circle radius.
    pub radius: f64,
    /// Whether to draw, fill, or fill-draw the circle.
    pub command: TikzDrawCommand,
    /// TikZ options for the command.
    pub options: TikzOptions,
}

/// A TikZ scope with its own options.
#[derive(Debug, Clone, PartialEq)]
pub struct TikzScope {
    /// Options placed on the scope.
    pub options: TikzOptions,
    /// The picture fragment rendered inside the scope.
    pub picture: TikzPicture,
}

/// One item in a TikZ picture.
#[derive(Debug, Clone, PartialEq)]
pub enum TikzItem {
    /// A style declaration.
    Style(TikzStyle),
    /// A node command.
    Node(TikzNode),
    /// A path command.
    Path(TikzPath),
    /// A circle command.
    Circle(TikzCircle),
    /// A nested scope.
    Scope(TikzScope),
    /// Raw TikZ source.
    Raw(String),
}

impl From<TikzStyle> for TikzItem {
    fn from(style: TikzStyle) -> Self {
        Self::Style(style)
    }
}

impl From<TikzNode> for TikzItem {
    fn from(node: TikzNode) -> Self {
        Self::Node(node)
    }
}

impl From<TikzPath> for TikzItem {
    fn from(path: TikzPath) -> Self {
        Self::Path(path)
    }
}

impl From<TikzCircle> for TikzItem {
    fn from(circle: TikzCircle) -> Self {
        Self::Circle(circle)
    }
}

impl From<TikzScope> for TikzItem {
    fn from(scope: TikzScope) -> Self {
        Self::Scope(scope)
    }
}

/// A complete TikZ picture.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TikzPicture {
    /// Options placed on the `tikzpicture` environment.
    pub options: TikzOptions,
    /// Commands rendered inside the picture.
    pub items: Vec<TikzItem>,
}

impl TikzPicture {
    /// Constructs an empty picture with no options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs an empty picture with explicit options.
    pub fn with_options(options: TikzOptions) -> Self {
        Self {
            options,
            ..Self::default()
        }
    }

    /// Appends an item to the picture.
    pub fn push(&mut self, item: impl Into<TikzItem>) {
        self.items.push(item.into());
    }

    /// Appends raw TikZ source to the picture.
    pub fn push_raw(&mut self, raw: impl Into<String>) {
        self.push(TikzItem::Raw(raw.into()));
    }

    /// Renders the picture as a `tikzpicture` environment.
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

    /// Renders the picture as an inline `\tikz{...}` command.
    pub fn render_inline(&self) -> String {
        let mut out = format!("\\tikz{}{{", self.options.render_brackets());
        self.render_body_inline_into(&mut out);
        out.push('}');
        out
    }

    fn render_body_into(&self, out: &mut String, indent: usize) {
        for item in &self.items {
            render_item_into(item, out, indent);
        }
    }

    fn render_body_inline_into(&self, out: &mut String) {
        for item in &self.items {
            render_item_inline_into(item, out);
        }
    }
}

impl fmt::Display for TikzPicture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

/// Layout and styling options for rendering a finite poset.
#[derive(Debug, Clone)]
pub struct PosetTikzOptions {
    /// Options placed on the outer `tikzpicture`.
    pub picture_options: TikzOptions,
    /// Horizontal spacing between elements of the same rank.
    pub x_spacing: f64,
    /// Vertical spacing between ranks.
    pub y_spacing: f64,
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

/// Options for drawing a poset of transfer systems.
#[derive(Debug, Clone)]
pub struct TransferSystemTikzOptions {
    /// Options for the outer poset of transfer systems.
    pub poset: PosetTikzOptions,
    /// Options for the miniature transfer-system glyph at each node.
    pub glyph: TransferSystemGlyphOptions,
}

impl Default for TransferSystemTikzOptions {
    fn default() -> Self {
        let mut poset = PosetTikzOptions {
            node_options: TikzOptions::new(["draw", "thick", "inner sep=2pt"]),
            x_spacing: 2.6,
            y_spacing: 2.6,
            ..PosetTikzOptions::default()
        };
        poset.bend_colinear_edges = true;
        Self {
            poset,
            glyph: TransferSystemGlyphOptions::default(),
        }
    }
}

/// Options for a miniature transfer-system diagram used as a node label.
#[derive(Debug, Clone)]
pub struct TransferSystemGlyphOptions {
    /// Scale factor for the inline glyph.
    pub scale: f64,
    /// TikZ baseline option for inline placement.
    pub baseline: String,
    /// Horizontal spacing in the underlying lattice glyph.
    pub x_spacing: f64,
    /// Vertical spacing in the underlying lattice glyph.
    pub y_spacing: f64,
    /// Options for relations that belong to the transfer system.
    pub highlighted_edge_options: TikzOptions,
    /// Options for proper lattice relations not in the transfer system.
    pub dim_edge_options: TikzOptions,
    /// Options for element dots.
    pub dot_options: TikzOptions,
    /// Radius of element dots.
    pub dot_radius: f64,
}

impl Default for TransferSystemGlyphOptions {
    fn default() -> Self {
        Self {
            scale: 0.35,
            baseline: "-.5ex".to_string(),
            x_spacing: 1.0,
            y_spacing: 0.9,
            highlighted_edge_options: TikzOptions::new(["line width=.75pt", "orange"]),
            dim_edge_options: TikzOptions::new(["black!30"]),
            dot_options: TikzOptions::new(["black"]),
            dot_radius: 0.1,
        }
    }
}

impl TransferSystemGlyphOptions {
    fn picture_options(&self) -> TikzOptions {
        TikzOptions::new([
            format!("scale={}", self.scale),
            format!("baseline={}", self.baseline),
        ])
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
    let mut coords = ranked_layout(poset, options.x_spacing, options.y_spacing);
    for (&id, &coordinate) in &options.coordinate_overrides {
        if id < poset.size() {
            coords.insert(id, coordinate);
        }
    }

    let mut picture = TikzPicture::with_options(options.picture_options.clone());
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

/// Renders the containment lattice of transfer systems as a TikZ picture.
pub fn transfer_system_lattice_to_tikz<A>(lattice: &TransferLattice<A>) -> TikzPicture {
    transfer_system_lattice_to_tikz_with(lattice, &transfer_system_tikz_options())
}

/// Renders the containment lattice of transfer systems with custom options.
pub fn transfer_system_lattice_to_tikz_with<A>(
    lattice: &TransferLattice<A>,
    options: &TransferSystemTikzOptions,
) -> TikzPicture {
    poset_to_tikz_with(lattice.as_poset(), &options.poset, |node_id, _raw| {
        let system = lattice
            .system(node_id)
            .expect("transfer-system lattice node id should be valid");
        TikzLabel::raw(small_transfer_system_picture(&system, &options.glyph).render_inline())
    })
}

/// Renders a poset of transfer systems as a TikZ picture.
pub fn transfer_system_order_to_tikz<A>(order: &TransferPoset<A>) -> TikzPicture {
    transfer_system_order_to_tikz_with(order, &transfer_system_tikz_options())
}

/// Renders a poset of transfer systems with custom options.
pub fn transfer_system_order_to_tikz_with<A>(
    order: &TransferPoset<A>,
    options: &TransferSystemTikzOptions,
) -> TikzPicture {
    poset_to_tikz_with(order.raw_poset(), &options.poset, |node_id, _raw| {
        let system = order
            .system(node_id)
            .expect("transfer-system poset node id should be valid");
        TikzLabel::raw(small_transfer_system_picture(&system, &options.glyph).render_inline())
    })
}

/// Returns the default options for transfer-system diagrams.
pub fn transfer_system_tikz_options() -> TransferSystemTikzOptions {
    TransferSystemTikzOptions::default()
}

/// Escapes TeX special characters in a plain-text label.
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

fn render_item_into(item: &TikzItem, out: &mut String, indent: usize) {
    let pad = " ".repeat(indent);
    match item {
        TikzItem::Style(style) => out.push_str(&format!(
            "{pad}\\tikzset{{{}/.style={}}}\n",
            style.name,
            style.options.render_braces()
        )),
        TikzItem::Node(node) => {
            let name = node
                .name
                .as_ref()
                .map(|name| format!(" ({name})"))
                .unwrap_or_default();
            out.push_str(&format!(
                "{pad}\\node{}{} at {} {{{}}};\n",
                node.options.render_brackets(),
                name,
                node.at.render(),
                node.label.render()
            ));
        }
        TikzItem::Path(path) => render_path_into(path, out, &pad),
        TikzItem::Circle(circle) => out.push_str(&format!(
            "{pad}\\{}{} {} circle ({:.3});\n",
            circle.command.render(),
            circle.options.render_brackets(),
            circle.center.render(),
            circle.radius
        )),
        TikzItem::Scope(scope) => {
            out.push_str(&format!(
                "{pad}\\begin{{scope}}{}\n",
                scope.options.render_brackets()
            ));
            scope.picture.render_body_into(out, indent + 2);
            out.push_str(&format!("{pad}\\end{{scope}}\n"));
        }
        TikzItem::Raw(raw) => render_raw_into(raw, out, indent),
    }
}

fn render_item_inline_into(item: &TikzItem, out: &mut String) {
    match item {
        TikzItem::Style(style) => out.push_str(&format!(
            "\\tikzset{{{}/.style={}}}",
            style.name,
            style.options.render_braces()
        )),
        TikzItem::Node(node) => {
            let name = node
                .name
                .as_ref()
                .map(|name| format!(" ({name})"))
                .unwrap_or_default();
            out.push_str(&format!(
                "\\node{}{} at {} {{{}}};",
                node.options.render_brackets(),
                name,
                node.at.render(),
                node.label.render()
            ));
        }
        TikzItem::Path(path) => render_path_inline_into(path, out),
        TikzItem::Circle(circle) => out.push_str(&format!(
            "\\{}{} {} circle ({:.3});",
            circle.command.render(),
            circle.options.render_brackets(),
            circle.center.render(),
            circle.radius
        )),
        TikzItem::Scope(scope) => {
            out.push_str(&format!(
                "\\begin{{scope}}{}",
                scope.options.render_brackets()
            ));
            scope.picture.render_body_inline_into(out);
            out.push_str("\\end{scope}");
        }
        TikzItem::Raw(raw) => out.push_str(raw),
    }
}

fn render_path_into(path: &TikzPath, out: &mut String, pad: &str) {
    match path.operation {
        TikzPathOperation::Line => out.push_str(&format!(
            "{pad}\\draw{} {} -- {};\n",
            path.options.render_brackets(),
            path.from.render(),
            path.to.render()
        )),
        TikzPathOperation::To => out.push_str(&format!(
            "{pad}\\draw {} to{} {};\n",
            path.from.render(),
            path.options.render_brackets(),
            path.to.render()
        )),
    }
}

fn render_path_inline_into(path: &TikzPath, out: &mut String) {
    match path.operation {
        TikzPathOperation::Line => out.push_str(&format!(
            "\\draw{} {} -- {};",
            path.options.render_brackets(),
            path.from.render(),
            path.to.render()
        )),
        TikzPathOperation::To => out.push_str(&format!(
            "\\draw {} to{} {};",
            path.from.render(),
            path.options.render_brackets(),
            path.to.render()
        )),
    }
}

fn render_raw_into(raw: &str, out: &mut String, indent: usize) {
    let pad = " ".repeat(indent);
    for line in raw.lines() {
        out.push_str(&pad);
        out.push_str(line);
        out.push('\n');
    }
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

fn small_transfer_system_picture<A>(
    system: &TransferSystem<A>,
    options: &TransferSystemGlyphOptions,
) -> TikzPicture {
    let highlighted_edges = system.edges(false);
    let coords = ranked_layout(
        system.lattice().as_poset(),
        options.x_spacing,
        options.y_spacing,
    );
    let mut picture = TikzPicture::with_options(options.picture_options());
    for edge in system.lattice().as_poset().proper_relations_iter() {
        let edge_options = if highlighted_edges.contains(&edge) {
            options.highlighted_edge_options.clone()
        } else {
            options.dim_edge_options.clone()
        };
        picture.push(TikzPath {
            from: coords[&edge.from].into(),
            to: coords[&edge.to].into(),
            operation: TikzPathOperation::Line,
            options: edge_options,
        });
    }
    for id in 0..system.lattice().size() {
        picture.push(TikzCircle {
            center: coords[&id].into(),
            radius: options.dot_radius,
            command: TikzDrawCommand::Fill,
            options: options.dot_options.clone(),
        });
    }
    picture
}

#[cfg(test)]
mod tests {
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
            "\\begin{tikzpicture}\n\\draw (a) to[bend left=18] (b);\n\\end{tikzpicture}\n"
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
    fn escape_tikz_escapes_latex_special_characters() {
        assert_eq!(
            escape_tikz("\\{}$&%#_^~"),
            "\\textbackslash{}\\{\\}\\$\\&\\%\\#\\_\\^{}\\~{}"
        );
    }

    #[test]
    fn ranked_layout_orders_same_rank_elements_by_element_id() {
        let poset = Poset::from_edges(vec![0i32, 1, 2], []).unwrap();
        let picture = poset_to_tikz_with(&poset, &PosetTikzOptions::default(), |_id, element| {
            TikzLabel::escaped(element.to_string())
        });

        let rendered = picture.render();
        assert!(
            rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p0) at (-1.800,0.000) {0};")
        );
        assert!(
            rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p1) at (0.000,0.000) {1};")
        );
        assert!(
            rendered.contains("\\node[circle, draw, inner sep=1.5pt] (p2) at (1.800,0.000) {2};")
        );
    }
}
