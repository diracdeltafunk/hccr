//! Drawings of posets of transfer systems, with a glyph at each node.

use super::drawing::{Draw, NodeContent, PosetDrawing};
use super::layout::{LayoutAlgorithm, PosetLayout};
use super::syntax::*;
use crate::cotransfer_lattice::CotransferLattice;
#[cfg(feature = "groups")]
use crate::g_cotransfer_lattice::GCotransferLattice;
#[cfg(feature = "groups")]
use crate::g_transfer_lattice::GTransferLattice;
use crate::lattice::Lattice;
use crate::poset::{Edge, EdgeSet, ElementId, Poset};
use crate::transfer_lattice::{TransferLattice, TransferPoset};

/// A small drawing of a lattice with a suborder of it highlighted, such as a
/// transfer system.
///
/// The edges of [`drawing`](Self::drawing) are styled by membership: an edge
/// is drawn with [`highlight_options`](Self::highlight_options) if its
/// relation belongs to the suborder and with
/// [`dim_options`](Self::dim_options) otherwise.
#[derive(Debug, Clone)]
pub struct Glyph {
    /// The drawing of the lattice.
    pub drawing: PosetDrawing,
    relations: EdgeSet,
    /// Options for edges whose relation belongs to the suborder.
    pub highlight_options: TikzOptions,
    /// Options for edges whose relation does not.
    pub dim_options: TikzOptions,
}

impl Glyph {
    /// A glyph of the suborder with the given non-identity relations, drawn on
    /// a copy of `drawing`.
    pub fn new(drawing: PosetDrawing, relations: EdgeSet) -> Self {
        let mut glyph = Self {
            drawing,
            relations,
            highlight_options: TikzOptions::new(["line width=.75pt", "orange"]),
            dim_options: TikzOptions::new(["black!30"]),
        };
        glyph.restyle();
        glyph
    }

    /// Whether `relation` belongs to the suborder.
    pub fn contains(&self, relation: Edge) -> bool {
        relation.is_identity() || self.relations.contains(&relation)
    }

    /// The non-identity relations of the suborder.
    pub fn relations(&self) -> &EdgeSet {
        &self.relations
    }

    /// Draws these pairs of elements, each styled by membership.
    pub fn set_edges(&mut self, edges: impl IntoIterator<Item = Edge>) {
        self.drawing.replace_edges(edges);
        self.restyle();
    }

    /// Restyles every edge by membership, discarding other changes to the
    /// edge options.
    pub fn restyle(&mut self) {
        for edge in &mut self.drawing.edges {
            edge.options = if self.relations.contains(&edge.relation()) {
                self.highlight_options.clone()
            } else {
                self.dim_options.clone()
            };
        }
    }
}

impl NodeContent for Glyph {
    fn to_label(&self) -> TikzLabel {
        self.drawing.to_label()
    }
}

impl PosetDrawing<Glyph> {
    /// Makes every glyph draw the given pairs of elements of the lattice,
    /// each highlighted if it belongs to that glyph's suborder.
    ///
    /// For example, `edges = lattice.covers()` draws each glyph on the Hasse
    /// diagram of the lattice and highlights the covers of the lattice that
    /// belong to the transfer system.
    pub fn glyph_edges(mut self, edges: impl IntoIterator<Item = Edge>) -> Self {
        let edges: Vec<Edge> = edges.into_iter().collect();
        for node in &mut self.nodes {
            node.content.set_edges(edges.iter().copied());
        }
        self
    }

    /// Lays out the lattice in every glyph with `layout`, computing the
    /// layout once.
    pub fn glyph_layout(mut self, layout: impl LayoutAlgorithm) -> Self {
        let Some(first) = self.nodes.first() else {
            return self;
        };
        let positions = first.content.drawing.clone().layout(layout).positions;
        for node in &mut self.nodes {
            node.content.drawing.positions = positions.clone();
        }
        self
    }

    /// Calls `f` on the glyph of every element.
    ///
    /// For example, to label the elements of the lattice in every glyph
    /// instead of drawing dots:
    ///
    /// ```
    /// use hccr::prelude::*;
    /// use hccr::tikz::{TikzLabel, TikzOptions};
    ///
    /// let lattice = chain(2);
    /// let drawing = lattice.transfer_systems().drawing().for_each_glyph(|_, glyph| {
    ///     for (id, node) in glyph.drawing.nodes.iter_mut().enumerate() {
    ///         node.content = TikzLabel::raw(format!("${id}$"));
    ///         node.options = TikzOptions::new(["inner sep=.5pt", "font=\\tiny"]);
    ///     }
    /// });
    /// assert!(drawing.to_tikz().render().contains("$2$"));
    /// ```
    pub fn for_each_glyph(mut self, mut f: impl FnMut(ElementId, &mut Glyph)) -> Self {
        for (id, node) in self.nodes.iter_mut().enumerate() {
            f(id, &mut node.content);
        }
        self
    }
}

/// Draws `order` with a glyph of `base` at each element, highlighting the
/// relations `relations(id)`.
fn glyph_drawing(
    order: &Poset,
    base: &Lattice,
    layout: impl LayoutAlgorithm,
    mut relations: impl FnMut(ElementId) -> EdgeSet,
) -> PosetDrawing<Glyph> {
    let template = PosetDrawing::new(base.as_poset(), base.default_layout(), |_| {
        TikzLabel::escaped("")
    })
    .edges(base.proper_relations())
    .spacing(1.0, 0.9)
    .options(TikzOptions::new(["scale=0.35", "baseline=-.5ex"]))
    .node_options(TikzOptions::new([
        "circle",
        "fill",
        "inner sep=0pt",
        "minimum size=0.7mm",
    ]));
    PosetDrawing::new(order, layout, |id| {
        Glyph::new(template.clone(), relations(id))
    })
    .node_options(TikzOptions::new(["draw", "thick", "inner sep=2pt"]))
    .spacing(2.6, 2.6)
}

/// Lattices of transfer systems are laid out additively, with a glyph of the
/// transfer system at each node.
impl Draw for TransferLattice {
    type Node = Glyph;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::ADDITIVE
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Glyph> {
        glyph_drawing(self, self.base_lattice(), layout, |id| {
            self.system(id).proper_relations()
        })
    }
}

/// Other orders on transfer systems need not be lattices, so they keep the
/// crossing-reduced layout.
impl Draw for TransferPoset {
    type Node = Glyph;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::CROSSING_REDUCED
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Glyph> {
        glyph_drawing(self, self.base_lattice(), layout, |id| {
            self.system(id).proper_relations()
        })
    }
}

impl Draw for CotransferLattice {
    type Node = Glyph;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::ADDITIVE
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Glyph> {
        glyph_drawing(self, self.base_lattice(), layout, |id| {
            self.system(id).proper_relations()
        })
    }
}

#[cfg(feature = "groups")]
impl Draw for GTransferLattice {
    type Node = Glyph;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::ADDITIVE
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Glyph> {
        glyph_drawing(self, self.base_lattice(), layout, |id| {
            self.system(id).proper_relations()
        })
    }
}

#[cfg(feature = "groups")]
impl Draw for GCotransferLattice {
    type Node = Glyph;

    fn default_layout(&self) -> PosetLayout {
        PosetLayout::ADDITIVE
    }

    fn drawing_with(&self, layout: impl LayoutAlgorithm) -> PosetDrawing<Glyph> {
        glyph_drawing(self, self.base_lattice(), layout, |id| {
            self.system(id).proper_relations()
        })
    }
}
