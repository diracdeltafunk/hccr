//! TikZ rendering for finite posets, lattices, and transfer-system diagrams.
//!
//! The module provides a small typed TikZ abstract syntax tree together with
//! convenience renderers for Hasse diagrams.  The default layout uses centered
//! feasible heights and layered crossing reduction, retaining the best
//! straight-line geometry found among the ranked and optimized candidates.
//! Cover relations are drawn unless full relations are requested.

#[cfg(feature = "groups")]
use crate::g_lattice::GTransferLattice;
use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};
use crate::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use bitvec::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// A type that can be rendered as a TikZ picture.
///
/// Each renderable type chooses its own associated option type.  Call
/// [`ToTikz::to_tikz`] for the mathematical default, or
/// [`ToTikz::to_tikz_with`] to control layout and styling explicitly.
pub trait ToTikz {
    /// The options accepted by [`ToTikz::to_tikz_with`].
    type Options: Default;

    /// Renders `self` as a TikZ picture with explicit options.
    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture;

    /// Renders `self` using [`Default::default`] for its associated options.
    fn to_tikz(&self) -> TikzPicture {
        self.to_tikz_with(&Self::Options::default())
    }
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

/// The algorithm used to assign coordinates to elements of a finite poset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PosetLayoutAlgorithm {
    /// Place elements at their earliest rank and order each rank by element id.
    Ranked,
    /// Center feasible heights and reduce crossings in a proper layered graph.
    #[default]
    CrossingReduced,
}

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
    /// The algorithm used to lay out the underlying lattice glyph.
    pub layout_algorithm: PosetLayoutAlgorithm,
    /// Whether to bend relations that would overlap glyph nodes or relations.
    pub bend_colinear_edges: bool,
    /// Bend angle used when a glyph relation must be curved.
    pub bend_angle: f64,
    /// Numerical tolerance used in glyph collinearity checks.
    pub colinear_tolerance: f64,
    /// Which relations of the full ambient lattice are drawn in the background.
    ///
    /// By default, every non-identity relation is drawn.
    pub ambient_relations: RelationDisplay,
    /// Which relations of the transfer-system suborder are highlighted.
    ///
    /// By default, every non-identity relation belonging to the transfer
    /// system is drawn.
    pub highlighted_relations: RelationDisplay,
    /// Whether ambient-lattice elements are shown as dots or supplied labels.
    pub node_display: GlyphNodeDisplay,
    /// Options for relations that belong to the transfer system.
    pub highlighted_edge_options: TikzOptions,
    /// Options for proper lattice relations not in the transfer system.
    pub dim_edge_options: TikzOptions,
    /// Options for element dots.
    pub dot_options: TikzOptions,
    /// Radius of element dots.
    pub dot_radius: f64,
    /// Options for nodes when [`GlyphNodeDisplay::Labels`] is selected.
    pub label_node_options: TikzOptions,
}

/// The content shown for each element of a miniature suborder diagram.
///
/// A label vector must have one entry for every element of the ambient lattice,
/// in `ElementId` order. This lets a single transfer-system rendering option be
/// reused for every node of an outer transfer-system lattice.
#[derive(Debug, Clone)]
pub enum GlyphNodeDisplay {
    /// Show the compact filled dots used by default.
    Dots,
    /// Show a supplied label at each ambient-lattice element.
    Labels(Vec<TikzLabel>),
}

impl GlyphNodeDisplay {
    /// Constructs label nodes from plain text, escaping TeX special characters.
    pub fn escaped(labels: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self::Labels(labels.into_iter().map(TikzLabel::escaped).collect())
    }

    /// Constructs label nodes from TeX math or TikZ markup.
    ///
    /// Callers are responsible for ensuring that every label is valid TeX.
    /// [`GlyphNodeDisplay::escaped`] is the appropriate constructor for plain
    /// text.
    pub fn raw(labels: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self::Labels(
            labels
                .into_iter()
                .map(|label| TikzLabel::raw(label.as_ref()))
                .collect(),
        )
    }
}

/// The amount of order-relation data displayed in a suborder glyph.
///
/// When used for the ambient lattice, [`RelationDisplay::Covers`] means its
/// ordinary Hasse edges.  When used for a transfer system, it means covers in
/// the transfer-system partial order itself, which can differ from covers in
/// the ambient lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationDisplay {
    /// Draw only cover relations, producing a Hasse-style diagram.
    Covers,
    /// Draw every non-identity relation, including transitive ones.
    AllProperRelations,
}

impl Default for TransferSystemGlyphOptions {
    fn default() -> Self {
        Self {
            scale: 0.35,
            baseline: "-.5ex".to_string(),
            x_spacing: 1.0,
            y_spacing: 0.9,
            layout_algorithm: PosetLayoutAlgorithm::default(),
            bend_colinear_edges: true,
            bend_angle: 18.0,
            colinear_tolerance: 1e-6,
            ambient_relations: RelationDisplay::AllProperRelations,
            highlighted_relations: RelationDisplay::AllProperRelations,
            node_display: GlyphNodeDisplay::Dots,
            highlighted_edge_options: TikzOptions::new(["line width=.75pt", "orange"]),
            dim_edge_options: TikzOptions::new(["black!30"]),
            dot_options: TikzOptions::new(["black"]),
            dot_radius: 0.1,
            label_node_options: TikzOptions::new(["inner sep=.5pt", "font=\\tiny"]),
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

/// Renders the containment lattice of transfer systems as a TikZ picture.
pub fn transfer_system_lattice_to_tikz<A>(lattice: &TransferLattice<A>) -> TikzPicture {
    lattice.to_tikz()
}

/// Renders the containment lattice of transfer systems with custom options.
pub fn transfer_system_lattice_to_tikz_with<A>(
    lattice: &TransferLattice<A>,
    options: &TransferSystemTikzOptions,
) -> TikzPicture {
    lattice.to_tikz_with(options)
}

/// Renders a poset of transfer systems as a TikZ picture.
pub fn transfer_system_order_to_tikz<A>(order: &TransferPoset<A>) -> TikzPicture {
    order.to_tikz()
}

/// Renders a poset of transfer systems with custom options.
pub fn transfer_system_order_to_tikz_with<A>(
    order: &TransferPoset<A>,
    options: &TransferSystemTikzOptions,
) -> TikzPicture {
    order.to_tikz_with(options)
}

/// Renders the containment lattice of G-transfer systems as a TikZ picture.
///
/// Each node is shown as a suborder of the full underlying lattice.  This
/// function is available only with the `groups` feature; it is also exposed by
/// the feature-gated [`ToTikz`] implementation for [`GTransferLattice`].
#[cfg(feature = "groups")]
pub fn g_transfer_system_lattice_to_tikz<A>(lattice: &GTransferLattice<A>) -> TikzPicture {
    lattice.to_tikz()
}

/// Renders the containment lattice of G-transfer systems with custom options.
#[cfg(feature = "groups")]
pub fn g_transfer_system_lattice_to_tikz_with<A>(
    lattice: &GTransferLattice<A>,
    options: &TransferSystemTikzOptions,
) -> TikzPicture {
    lattice.to_tikz_with(options)
}

impl<A> ToTikz for TransferLattice<A> {
    type Options = TransferSystemTikzOptions;

    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture {
        transfer_system_order_picture(
            self.as_poset(),
            self.universe().lattice().as_ref(),
            options,
            |node_id, glyph_renderer| {
                self.system(node_id)
                    .map(|system| glyph_renderer.render_transfer_system(&system))
                    .expect("transfer-system lattice node id should be valid")
            },
        )
    }
}

impl<A> ToTikz for TransferPoset<A> {
    type Options = TransferSystemTikzOptions;

    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture {
        transfer_system_order_picture(
            self.raw_poset(),
            self.universe().lattice().as_ref(),
            options,
            |node_id, glyph_renderer| {
                self.system(node_id)
                    .map(|system| glyph_renderer.render_transfer_system(&system))
                    .expect("transfer-system poset node id should be valid")
            },
        )
    }
}

#[cfg(feature = "groups")]
impl<A> ToTikz for GTransferLattice<A> {
    type Options = TransferSystemTikzOptions;

    fn to_tikz_with(&self, options: &Self::Options) -> TikzPicture {
        transfer_system_order_picture(
            self.as_poset(),
            self.universe().lattice().as_ref(),
            options,
            |node_id, glyph_renderer| {
                let system = self
                    .system(node_id)
                    .expect("G-transfer-system lattice node id should be valid");
                glyph_renderer.render_with(|relation| system.contains_relation(relation))
            },
        )
    }
}

fn transfer_system_order_picture<A, R, F>(
    order: &Poset<R>,
    underlying_lattice: &Lattice<A>,
    options: &TransferSystemTikzOptions,
    mut glyph_for_node: F,
) -> TikzPicture
where
    F: FnMut(ElementId, &SuborderGlyphRenderer<'_, A>) -> TikzPicture,
{
    let glyph_renderer = SuborderGlyphRenderer::new(underlying_lattice, &options.glyph);
    poset_to_tikz_with(order, &options.poset, |node_id, _raw| {
        TikzLabel::raw(glyph_for_node(node_id, &glyph_renderer).render_inline())
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
            "{pad}\\draw{} {} to {};\n",
            path.options.render_brackets(),
            path.from.render(),
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
            "\\draw{} {} to {};",
            path.options.render_brackets(),
            path.from.render(),
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

#[cfg(test)]
fn default_layout<A>(
    poset: &Poset<A>,
    x_spacing: f64,
    y_spacing: f64,
) -> HashMap<ElementId, (f64, f64)> {
    let mut covers: Vec<_> = poset.cover_relations().into_iter().collect();
    covers.sort_unstable();
    layout_with_covers(
        poset.size(),
        &covers,
        x_spacing,
        y_spacing,
        PosetLayoutAlgorithm::default(),
    )
}

fn layout_with_covers(
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

/// A proper layered graph made by subdividing every Hasse edge at each
/// intervening half-rank. Vertices below `real_count` are poset elements; the
/// rest are temporary dummy vertices used only by crossing reduction.
struct LayeredCoverGraph {
    real_count: usize,
    real_edges: Vec<Edge>,
    levels: Vec<usize>,
    layers: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    outgoing: Vec<Vec<usize>>,
}

impl LayeredCoverGraph {
    fn new(real_levels: &[usize], covers: &[Edge]) -> Self {
        let real_count = real_levels.len();
        let layer_count = real_levels
            .iter()
            .copied()
            .max()
            .map_or(0, |maximum| maximum + 1);
        let mut graph = Self {
            real_count,
            real_edges: covers.to_vec(),
            levels: real_levels.to_vec(),
            layers: vec![Vec::new(); layer_count],
            incoming: vec![Vec::new(); real_count],
            outgoing: vec![Vec::new(); real_count],
        };

        for (id, &level) in real_levels.iter().enumerate() {
            graph.layers[level].push(id);
        }

        for &edge in covers {
            let from_level = graph.levels[edge.from];
            let to_level = graph.levels[edge.to];
            debug_assert!(from_level < to_level);

            let mut previous = edge.from;
            for level in from_level + 1..to_level {
                let dummy = graph.levels.len();
                graph.levels.push(level);
                graph.layers[level].push(dummy);
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
        debug_assert_eq!(self.levels[from] + 1, self.levels[to]);
        self.outgoing[from].push(to);
        self.incoming[to].push(from);
    }

    fn reduce_crossings(&mut self) -> StraightGeometryDefects {
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
        let mut routed_crossings = 0u64;
        let mut horizontal_length = 0u64;

        for level in 0..self.layers.len().saturating_sub(1) {
            let upper_width = self.layers[level + 1].len();
            let mut targets_seen = FenwickTree::new(upper_width);
            let mut seen_count = 0u64;

            for &source in &self.layers[level] {
                // Query the whole source group before inserting it, so edges
                // with a shared endpoint are not counted as crossings.
                for &target in &self.outgoing[source] {
                    let target_position = positions[target];
                    routed_crossings = routed_crossings
                        .saturating_add(seen_count - targets_seen.prefix_sum(target_position + 1));
                }
                for &target in &self.outgoing[source] {
                    targets_seen.add(positions[target], 1);
                    seen_count += 1;

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

    fn straight_geometry_defects(&self) -> StraightGeometryDefects {
        let positions = self.vertex_positions();
        straight_geometry_defects(
            &self.real_edges,
            &self.real_grid_points_with_positions(&positions),
        )
    }

    fn real_grid_points_with_positions(&self, positions: &[usize]) -> Vec<GridPoint> {
        (0..self.real_count)
            .map(|id| {
                let level = self.levels[id];
                GridPoint {
                    x: doubled_centered_slot(positions[id], self.layers[level].len()),
                    y: level as i64,
                }
            })
            .collect()
    }

    fn vertex_positions(&self) -> Vec<usize> {
        let mut positions = vec![0usize; self.levels.len()];
        for layer in &self.layers {
            for (position, &vertex) in layer.iter().enumerate() {
                positions[vertex] = position;
            }
        }
        positions
    }

    fn real_coordinates(&self, x_spacing: f64, y_spacing: f64) -> HashMap<ElementId, (f64, f64)> {
        let positions = self.vertex_positions();
        (0..self.real_count)
            .map(|id| {
                let level = self.levels[id];
                let width = self.layers[level].len().saturating_sub(1) as f64;
                let x = (positions[id] as f64 - width / 2.0) * x_spacing;
                let y = level as f64 * y_spacing / 2.0;
                (id, (x, y))
            })
            .collect()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GridPoint {
    x: i64,
    y: i64,
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
struct StraightGeometryDefects {
    proper_crossings: u64,
    node_intersections: u64,
    collinear_overlaps: u64,
}

impl StraightGeometryDefects {
    fn is_clean(self) -> bool {
        self.proper_crossings == 0 && self.node_intersections == 0 && self.collinear_overlaps == 0
    }

    fn replace_contribution(self, before: Self, after: Self) -> Self {
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

fn straight_geometry_defects(edges: &[Edge], points: &[GridPoint]) -> StraightGeometryDefects {
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
fn straight_geometry_contribution(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrderingScore {
    geometry: StraightGeometryDefects,
    routed_crossings: u64,
    horizontal_length: u64,
}

fn doubled_centered_slot(position: usize, layer_width: usize) -> i64 {
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
            index += index & index.wrapping_neg();
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

/// Cached geometry and drawing data for miniature spanning suborders.
///
/// One renderer is created for the ambient lattice of an outer transfer-system
/// diagram and reused for every node.  This keeps all glyphs in the same
/// coordinates and avoids recomputing the ranked layout for each transfer
/// system.
struct SuborderGlyphRenderer<'a, A> {
    lattice: &'a Lattice<A>,
    coordinates: HashMap<ElementId, (f64, f64)>,
    proper_edges: Vec<Edge>,
    ambient_edges: Vec<Edge>,
    options: &'a TransferSystemGlyphOptions,
}

impl<'a, A> SuborderGlyphRenderer<'a, A> {
    fn new(lattice: &'a Lattice<A>, options: &'a TransferSystemGlyphOptions) -> Self {
        if let GlyphNodeDisplay::Labels(labels) = &options.node_display {
            assert_eq!(
                labels.len(),
                lattice.size(),
                "suborder glyph received {} labels for a lattice with {} elements",
                labels.len(),
                lattice.size(),
            );
        }
        let proper_edges = lattice
            .as_poset()
            .proper_relations_iter()
            .collect::<Vec<_>>();
        let mut covers = lattice
            .as_poset()
            .cover_relations()
            .into_iter()
            .collect::<Vec<_>>();
        covers.sort_unstable();
        let ambient_edges = match options.ambient_relations {
            RelationDisplay::Covers => covers.clone(),
            RelationDisplay::AllProperRelations => proper_edges.clone(),
        };
        Self {
            lattice,
            coordinates: layout_with_covers(
                lattice.size(),
                &covers,
                options.x_spacing,
                options.y_spacing,
                options.layout_algorithm,
            ),
            proper_edges,
            ambient_edges,
            options,
        }
    }

    fn render_transfer_system(&self, system: &TransferSystem<A>) -> TikzPicture {
        self.render_with(|relation| system.contains_relation(relation))
    }

    fn render_with<F>(&self, mut contains_relation: F) -> TikzPicture
    where
        F: FnMut(Edge) -> bool,
    {
        let selected = self
            .proper_edges
            .iter()
            .copied()
            .map(&mut contains_relation)
            .collect::<Vec<_>>();
        let mut highlighted_edges = match self.options.highlighted_relations {
            RelationDisplay::Covers => {
                selected_cover_relations(self.lattice.size(), &self.proper_edges, &selected)
            }
            RelationDisplay::AllProperRelations => self
                .proper_edges
                .iter()
                .copied()
                .zip(selected.iter().copied())
                .filter_map(|(edge, is_selected)| is_selected.then_some(edge))
                .collect(),
        };
        highlighted_edges.sort_unstable();
        highlighted_edges.dedup();

        // Compute one bend decision per relation in the union. Each relation
        // is emitted once, with either its highlighted or dimmed style.
        let mut visible_edges = self.ambient_edges.clone();
        visible_edges.extend(highlighted_edges.iter().copied());
        visible_edges.sort_unstable();
        visible_edges.dedup();
        let bend_edges = if self.options.bend_colinear_edges {
            edge_bend_decisions(
                &visible_edges,
                &self.coordinates,
                self.options.colinear_tolerance,
            )
        } else {
            vec![false; visible_edges.len()]
        };
        let make_path = |index: usize, edge: Edge, mut options: TikzOptions| {
            let operation = if bend_edges[index] {
                let direction = if index.is_multiple_of(2) {
                    "left"
                } else {
                    "right"
                };
                options.push(format!("bend {direction}={}", self.options.bend_angle));
                TikzPathOperation::To
            } else {
                TikzPathOperation::Line
            };
            TikzPath {
                from: self.coordinates[&edge.from].into(),
                to: self.coordinates[&edge.to].into(),
                operation,
                options,
            }
        };

        let mut picture = TikzPicture::with_options(self.options.picture_options());
        let mut highlighted_index = 0;
        for (index, &edge) in visible_edges.iter().enumerate() {
            while highlighted_index < highlighted_edges.len()
                && highlighted_edges[highlighted_index] < edge
            {
                highlighted_index += 1;
            }
            let is_highlighted = highlighted_edges.get(highlighted_index) == Some(&edge);
            let options = if is_highlighted {
                self.options.highlighted_edge_options.clone()
            } else {
                self.options.dim_edge_options.clone()
            };
            picture.push(make_path(index, edge, options));
        }
        match &self.options.node_display {
            GlyphNodeDisplay::Dots => {
                for id in 0..self.lattice.size() {
                    picture.push(TikzCircle {
                        center: self.coordinates[&id].into(),
                        radius: self.options.dot_radius,
                        command: TikzDrawCommand::Fill,
                        options: self.options.dot_options.clone(),
                    });
                }
            }
            GlyphNodeDisplay::Labels(labels) => {
                for (id, label) in labels.iter().enumerate() {
                    picture.push(TikzNode {
                        name: None,
                        at: self.coordinates[&id].into(),
                        label: label.clone(),
                        options: self.options.label_node_options.clone(),
                    });
                }
            }
        }
        picture
    }
}

fn selected_cover_relations(
    lattice_size: usize,
    proper_edges: &[Edge],
    selected: &[bool],
) -> Vec<Edge> {
    let mut relation: Vec<BitVec> = vec![BitVec::repeat(false, lattice_size); lattice_size];
    for (id, row) in relation.iter_mut().enumerate() {
        row.set(id, true);
    }
    for (&edge, &is_selected) in proper_edges.iter().zip(selected) {
        if is_selected {
            relation[edge.from].set(edge.to, true);
        }
    }

    proper_edges
        .iter()
        .copied()
        .zip(selected.iter().copied())
        .filter_map(|(edge, is_selected)| {
            (is_selected
                && !(0..lattice_size).any(|middle| {
                    middle != edge.from
                        && middle != edge.to
                        && relation[edge.from][middle]
                        && relation[middle][edge.to]
                }))
            .then_some(edge)
        })
        .collect()
}

#[cfg(test)]
fn small_transfer_system_picture<A>(
    system: &TransferSystem<A>,
    options: &TransferSystemGlyphOptions,
) -> TikzPicture {
    SuborderGlyphRenderer::new(system.lattice().as_ref(), options).render_transfer_system(system)
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
}
