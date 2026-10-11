//! Drawings of finite posets: positions, node contents, and edges, as data.

use super::edge_routing::edge_bend_decisions;
use super::layout::{LayoutAlgorithm, PosetLayout};
use super::syntax::*;
use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};

/// What can be drawn at the node of a [`PosetDrawing`].
///
/// Implemented for plain labels ([`TikzLabel`]), for transfer-system glyphs
/// ([`super::Glyph`]), and for whole drawings, which are drawn inline.
/// Implement it to put anything else at the nodes.
pub trait NodeContent {
    /// The TikZ content of the node.
    fn to_label(&self) -> TikzLabel;
}

impl NodeContent for TikzLabel {
    fn to_label(&self) -> TikzLabel {
        self.clone()
    }
}

impl<N: NodeContent> NodeContent for PosetDrawing<N> {
    fn to_label(&self) -> TikzLabel {
        TikzLabel::raw(self.render(false).render_inline())
    }
}

/// One element of a drawing: what is drawn there, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Node<N> {
    /// The content of the node.
    pub content: N,
    /// TikZ options for the node.
    pub options: TikzOptions,
}

/// A segment drawn between two elements of a drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawnEdge {
    /// The element at which the edge starts.
    pub from: ElementId,
    /// The element at which the edge ends.
    pub to: ElementId,
    /// TikZ options for the edge.
    pub options: TikzOptions,
    /// How the edge is routed.
    pub route: Route,
}

impl DrawnEdge {
    /// The pair of elements the edge joins.
    pub fn relation(&self) -> Edge {
        Edge::new(self.from, self.to)
    }
}

/// How an edge is routed between its endpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Route {
    /// Straight, unless it would pass through another node or lie along a
    /// shorter edge; then it bends by [`PosetDrawing::bend_angle`]. Decided
    /// when the drawing is rendered, from the positions at that time.
    #[default]
    Auto,
    /// Always straight.
    Straight,
    /// Bend by this many degrees: to the left of the direction of travel if
    /// positive, to the right if negative.
    Bend(f64),
}

/// A drawing of a finite poset: a position and a node for each element, and
/// a list of edges.
///
/// A drawing is plain data. Every element of the poset has an entry in
/// [`positions`](Self::positions) and [`nodes`](Self::nodes), indexed by its
/// [`ElementId`], and every field can be edited freely before rendering with
/// [`to_tikz`](Self::to_tikz). Positions are in layout units (one unit
/// between neighbouring elements of a level, one unit per height step), and
/// are scaled by [`spacing`](Self::spacing) when rendered.
///
/// The builder methods only set data and never move a node, except for
/// [`layout`](Self::layout), which recomputes every position.
///
/// ```
/// use hccr::prelude::*;
/// use hccr::tikz::PosetLayout;
///
/// let square = product([chain(1), chain(1)]);
/// let picture = square
///     .drawing_with(PosetLayout::TEXTBOOK)
///     .edges(square.proper_relations())
///     .spacing(1.0, 1.0)
///     .to_tikz();
/// assert!(picture.render().contains("\\draw"));
/// ```
#[derive(Debug, Clone)]
pub struct PosetDrawing<N = TikzLabel> {
    poset: Poset,
    /// The position of each element, in layout units.
    pub positions: Vec<(f64, f64)>,
    /// The node of each element.
    pub nodes: Vec<Node<N>>,
    /// The edges, drawn in this order.
    pub edges: Vec<DrawnEdge>,
    /// The size in TikZ units of one layout unit, horizontally and vertically.
    pub spacing: (f64, f64),
    /// Options for the whole picture.
    pub options: TikzOptions,
    /// Options given to edges added by [`edges`](Self::edges).
    pub edge_options: TikzOptions,
    /// The angle, in degrees, by which [`Route::Auto`] edges bend.
    pub bend_angle: f64,
    /// Whether to mark each node with its [`ElementId`], for debugging.
    pub show_ids: bool,
}

impl<N> PosetDrawing<N> {
    /// Lays out `poset` with `layout`, puts `content(id)` at each element,
    /// and draws the cover relations.
    ///
    /// Nodes and edges start with no options. [`Draw::drawing`] gives the
    /// styled default drawing of each kind of poset.
    pub fn new(
        poset: &Poset,
        layout: impl LayoutAlgorithm,
        content: impl FnMut(ElementId) -> N,
    ) -> Self {
        let nodes = poset
            .ids()
            .map(content)
            .map(|content| Node {
                content,
                options: TikzOptions::default(),
            })
            .collect();
        Self {
            poset: poset.clone(),
            positions: Vec::new(),
            nodes,
            edges: Vec::new(),
            spacing: (1.8, 1.4),
            options: TikzOptions::default(),
            edge_options: TikzOptions::default(),
            bend_angle: 18.0,
            show_ids: false,
        }
        .layout(layout)
        .edges(poset.covers())
    }

    /// The poset being drawn.
    pub fn poset(&self) -> &Poset {
        &self.poset
    }

    /// Recomputes every position with `layout`, discarding any edits to the
    /// positions.
    ///
    /// # Panics
    ///
    /// Panics if `layout` does not return one position per element.
    pub fn layout(mut self, layout: impl LayoutAlgorithm) -> Self {
        self.positions = layout.positions(&self.poset);
        assert_eq!(
            self.positions.len(),
            self.poset.size(),
            "a layout returned {} positions for a poset with {} elements",
            self.positions.len(),
            self.poset.size(),
        );
        self
    }

    /// Replaces the edges with segments between the given pairs of elements,
    /// styled with [`edge_options`](Self::edge_options).
    ///
    /// The pairs need not be relations of the poset; for example,
    /// `poset.proper_relations()` draws every relation and `poset.covers()`
    /// draws the Hasse diagram.
    ///
    /// # Panics
    ///
    /// Panics if a pair names an element outside the poset.
    pub fn edges(mut self, edges: impl IntoIterator<Item = Edge>) -> Self {
        self.replace_edges(edges);
        self
    }

    pub(super) fn replace_edges(&mut self, edges: impl IntoIterator<Item = Edge>) {
        let size = self.poset.size();
        self.edges = edges
            .into_iter()
            .map(|edge| {
                assert!(
                    edge.from < size && edge.to < size,
                    "edge {edge} names an element outside a poset with {size} elements"
                );
                DrawnEdge {
                    from: edge.from,
                    to: edge.to,
                    options: self.edge_options.clone(),
                    route: Route::Auto,
                }
            })
            .collect();
    }

    /// Sets the size in TikZ units of one layout unit.
    pub fn spacing(mut self, x: f64, y: f64) -> Self {
        self.spacing = (x, y);
        self
    }

    /// Sets the options of the whole picture.
    pub fn options(mut self, options: TikzOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets the options of every node.
    pub fn node_options(mut self, options: TikzOptions) -> Self {
        for node in &mut self.nodes {
            node.options = options.clone();
        }
        self
    }

    /// Sets the options of every edge, and of edges added later by
    /// [`edges`](Self::edges).
    pub fn edge_options(mut self, options: TikzOptions) -> Self {
        for edge in &mut self.edges {
            edge.options = options.clone();
        }
        self.edge_options = options;
        self
    }

    /// Sets whether each node is marked with its [`ElementId`].
    pub fn show_ids(mut self, show: bool) -> Self {
        self.show_ids = show;
        self
    }

    /// Reflects the drawing left to right.
    pub fn mirror(mut self) -> Self {
        for position in &mut self.positions {
            position.0 = -position.0;
        }
        self
    }

    /// Replaces the content of each node by `f(id, content)`.
    pub fn map_nodes<M>(self, mut f: impl FnMut(ElementId, N) -> M) -> PosetDrawing<M> {
        PosetDrawing {
            poset: self.poset,
            positions: self.positions,
            nodes: self
                .nodes
                .into_iter()
                .enumerate()
                .map(|(id, node)| Node {
                    content: f(id, node.content),
                    options: node.options,
                })
                .collect(),
            edges: self.edges,
            spacing: self.spacing,
            options: self.options,
            edge_options: self.edge_options,
            bend_angle: self.bend_angle,
            show_ids: self.show_ids,
        }
    }
}

impl<N: NodeContent> PosetDrawing<N> {
    /// Renders the drawing as a TikZ picture.
    pub fn to_tikz(&self) -> TikzPicture {
        self.render(true)
    }

    /// With `named`, nodes are named `p0`, `p1`, ... and edges run between
    /// node borders. Without, nothing is named and edges run between node
    /// centres, so that a drawing can be nested in a node of another without
    /// its names clashing.
    pub(super) fn render(&self, named: bool) -> TikzPicture {
        let points: Vec<(f64, f64)> = self
            .positions
            .iter()
            .map(|&(x, y)| (x * self.spacing.0, y * self.spacing.1))
            .collect();
        let relations: Vec<Edge> = self.edges.iter().map(DrawnEdge::relation).collect();
        let automatic_bends = edge_bend_decisions(&relations, &points, BEND_TOLERANCE);

        let mut picture = TikzPicture::with_options(self.options.clone());
        let node_name = |id: ElementId| format!("p{id}");
        let endpoint = |id: ElementId| {
            if named {
                TikzCoord::named(node_name(id))
            } else {
                points[id].into()
            }
        };
        let nodes = self.nodes.iter().enumerate().map(|(id, node)| TikzNode {
            name: named.then(|| node_name(id)),
            at: points[id].into(),
            label: node.content.to_label(),
            options: node.options.clone(),
        });
        let edges = self.edges.iter().enumerate().map(|(index, edge)| {
            let bend = match edge.route {
                Route::Auto if automatic_bends[index] => Some(if index % 2 == 0 {
                    self.bend_angle
                } else {
                    -self.bend_angle
                }),
                Route::Auto | Route::Straight => None,
                Route::Bend(angle) => Some(angle),
            };
            let mut options = edge.options.clone();
            let operation = match bend {
                Some(angle) => {
                    let side = if angle >= 0.0 { "left" } else { "right" };
                    options.push(format!("bend {side}={}", angle.abs()));
                    TikzPathOperation::To
                }
                None => TikzPathOperation::Line,
            };
            TikzPath {
                from: endpoint(edge.from),
                to: endpoint(edge.to),
                operation,
                options,
            }
        });

        // Named edges must follow the nodes they refer to; unnamed edges go
        // first so that nodes are drawn over them.
        if named {
            nodes.for_each(|node| picture.push(node));
            edges.for_each(|edge| picture.push(edge));
        } else {
            edges.for_each(|edge| picture.push(edge));
            nodes.for_each(|node| picture.push(node));
        }

        if self.show_ids && named {
            for id in self.poset.ids() {
                picture.push(TikzNode {
                    name: None,
                    at: TikzCoord::raw(format!("({}.north east)", node_name(id))),
                    label: TikzLabel::raw(format!("$ {id} $")),
                    options: TikzOptions::new([
                        "circle",
                        "draw=red!70!black",
                        "fill=white",
                        "inner sep=.6pt",
                        "font=\\scriptsize",
                    ]),
                });
            }
        }
        picture
    }
}

/// How close, in TikZ units, a node must be to an edge for [`Route::Auto`]
/// to bend the edge around it.
const BEND_TOLERANCE: f64 = 1e-6;

/// Something that can be drawn as a poset.
///
/// [`drawing`](Self::drawing) gives the default drawing, laid out with
/// [`default_layout`](Self::default_layout), and
/// [`drawing_with`](Self::drawing_with) the same drawing with any other
/// layout, including a closure. The result is a [`PosetDrawing`] to adjust
/// before rendering, or [`to_tikz`](Self::to_tikz) renders the default
/// directly.
pub trait Draw {
    /// What is drawn at each element.
    type Node: NodeContent;

    /// The layout used by [`drawing`](Self::drawing).
    fn default_layout(&self) -> PosetLayout;

    /// The default drawing, laid out by `layout` instead.
    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Self::Node>;

    /// The default drawing.
    fn drawing(&self) -> PosetDrawing<Self::Node> {
        self.drawing_with(self.default_layout())
    }

    /// Renders the default drawing.
    fn to_tikz(&self) -> TikzPicture {
        self.drawing().to_tikz()
    }
}

fn labelled_drawing(poset: &Poset, layout: impl LayoutAlgorithm) -> PosetDrawing {
    PosetDrawing::new(poset, layout, |id| {
        TikzLabel::escaped(poset.label(id).to_string())
    })
    .node_options(TikzOptions::new(["circle", "draw", "inner sep=1.5pt"]))
}

/// Posets are drawn as Hasse diagrams of their labels, with crossings
/// reduced.
impl Draw for Poset {
    type Node = TikzLabel;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::CROSSING_REDUCED
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing {
        labelled_drawing(self, layout)
    }
}

/// Lattices are drawn as Hasse diagrams of their labels, laid out
/// additively.
impl Draw for Lattice {
    type Node = TikzLabel;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::ADDITIVE
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing {
        labelled_drawing(self.as_poset(), layout)
    }
}
