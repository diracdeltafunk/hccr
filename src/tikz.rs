//! TikZ drawings of finite posets, lattices, and posets of transfer systems.
//!
//! # Overview
//!
//! A drawing is made in stages, and each stage hands you a value you can
//! inspect and change before the next one runs:
//!
//! ```text
//!  what you draw              layout                 drawing                 output
//! ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐
//! │ Poset, Lattice,    │  │ LayoutAlgorithm: │  │ PosetDrawing<N>  │  │ TikzPicture   │
//! │ TransferLattice, … │─▶│ a PosetLayout or │─▶│ positions, nodes,│─▶│ (TikZ syntax  │─▶ String
//! │ or any &Poset with │  │ any closure      │  │ edges, spacing … │  │  tree)        │
//! │ your node content  │  └──────────────────┘  └──────────────────┘  └───────────────┘
//! └────────────────────┘                          ▲ edit freely │         ▲ push more
//!                                                 └─────────────┘
//! ```
//!
//! 1. A [`LayoutAlgorithm`] gives every element a position in *layout units*:
//!    neighbouring elements of a level are one unit apart, and each height
//!    step is one unit. It is a [`PosetLayout`] (see its presets) or any
//!    closure `Fn(&Poset) -> Vec<(f64, f64)>`.
//! 2. A [`PosetDrawing<N>`](PosetDrawing) holds those positions together with a
//!    [`Node`] for each element (its content, of type `N`, and its TikZ
//!    options) and a list of [`DrawnEdge`]s. It is plain data: every field is
//!    public.
//! 3. [`PosetDrawing::to_tikz`] renders the drawing as a [`TikzPicture`], a
//!    small TikZ syntax tree, which renders as text with
//!    [`TikzPicture::render`], [`TikzPicture::render_inline`] or
//!    [`TikzPicture::to_standalone_document`].
//!
//! ```
//! use hccr::prelude::*;
//! use hccr::tikz::PosetLayout;
//!
//! let tr = chain(2).transfer_systems();
//!
//! // The default drawing, straight to TikZ.
//! let picture = tr.to_tikz();
//!
//! // A different layout and size, with each glyph drawn on the Hasse
//! // diagram of the base lattice.
//! let picture = tr
//!     .drawing_with(PosetLayout::BALANCED)
//!     .spacing(3.0, 3.0)
//!     .glyph_edges(tr.base_lattice().covers())
//!     .to_tikz();
//! println!("{}", picture.to_standalone_document());
//! ```
//!
//! # Getting a drawing
//!
//! Everything drawable implements [`Draw`], whose only required methods are
//! [`Draw::default_layout`] and [`Draw::drawing_with`].
//!
//! | Call | Gives | Layout | Styling |
//! |------|-------|--------|---------|
//! | `x.to_tikz()` | a [`TikzPicture`] directly | the type's default | the type's default |
//! | `x.drawing()` | `PosetDrawing<X::Node>` | the type's default | the type's default |
//! | `x.drawing_with(layout)` | `PosetDrawing<X::Node>` | yours | the type's default |
//! | [`PosetDrawing::new`]`(&poset, layout, content)` | `PosetDrawing<N>` for any `N` | yours | none: empty options, covers as edges |
//! | `drawing.`[`map_nodes`](PosetDrawing::map_nodes)`(f)` | `PosetDrawing<M>` | kept | kept; only node contents change |
//! | [`Glyph::new`]`(drawing, relations)` | one [`Glyph`] | the drawing's | edges styled by membership |
//!
//! What each type draws by default:
//!
//! | Type | Node | Default layout | Nodes | Spacing | Edges |
//! |------|------|----------------|-------|---------|-------|
//! | [`Poset`](crate::poset::Poset) | [`TikzLabel`] | [`PosetLayout::CROSSING_REDUCED`] | its label, in a circle | 1.8 × 1.4 | covers |
//! | [`Lattice`](crate::lattice::Lattice) | [`TikzLabel`] | [`PosetLayout::ADDITIVE`] | its label, in a circle | 1.8 × 1.4 | covers |
//! | lattices of transfer, cotransfer, G-transfer and G-cotransfer systems | [`Glyph`] | [`PosetLayout::ADDITIVE`] | a glyph, in a thick box | 2.6 × 2.6 | covers |
//! | [`TransferPoset`](crate::transfer_lattice::TransferPoset) | [`Glyph`] | [`PosetLayout::CROSSING_REDUCED`] | a glyph, in a thick box | 2.6 × 2.6 | covers |
//! | the base lattice inside each glyph | [`TikzLabel`] (empty) | the base lattice's, [`PosetLayout::ADDITIVE`] | small dots | 1.0 × 0.9, scaled by 0.35 | every proper relation, highlighted if in the system |
//!
//! Types that dereference to [`Lattice`](crate::lattice::Lattice), such as
//! subgroup lattices, are drawn as lattices.
//!
//! # Changing a drawing
//!
//! There are three levels, and they combine.
//!
//! **Builder methods** take the drawing by value and return it, so they chain.
//! Only [`layout`](PosetDrawing::layout) and [`mirror`](PosetDrawing::mirror)
//! move nodes.
//!
//! | Method | Effect |
//! |--------|--------|
//! | [`layout(l)`](PosetDrawing::layout) | recomputes every position, discarding edits to them |
//! | [`mirror()`](PosetDrawing::mirror) | reflects the drawing left to right |
//! | [`edges(es)`](PosetDrawing::edges) | draws these pairs of elements instead, e.g. `poset.covers()` or `poset.proper_relations()` |
//! | [`spacing(x, y)`](PosetDrawing::spacing) | sets the TikZ size of one layout unit |
//! | [`options(o)`](PosetDrawing::options) | sets the picture's options |
//! | [`node_options(o)`](PosetDrawing::node_options) | sets every node's options |
//! | [`edge_options(o)`](PosetDrawing::edge_options) | sets every edge's options, and those of edges added later |
//! | [`show_ids(b)`](PosetDrawing::show_ids) | marks each node with its element id |
//! | [`map_nodes(f)`](PosetDrawing::map_nodes) | replaces each node's content, possibly by another type |
//!
//! On a `PosetDrawing<Glyph>` only, these act on every glyph:
//!
//! | Method | Effect |
//! |--------|--------|
//! | [`glyph_edges(es)`](PosetDrawing::glyph_edges) | every glyph draws these pairs of base-lattice elements, each highlighted iff in its transfer system |
//! | [`glyph_layout(l)`](PosetDrawing::glyph_layout) | lays out the base lattice once and uses it in every glyph |
//! | [`for_each_glyph(f)`](PosetDrawing::for_each_glyph) | calls `f(id, &mut glyph)` for each element |
//!
//! **Direct edits** reach everything else: move an element with
//! `drawing.positions[id]`, restyle or replace one node through
//! `drawing.nodes[id]`, and edit `drawing.edges` as a vector, setting each
//! edge's options or [`Route`] ([`Route::Auto`], [`Route::Straight`] or
//! [`Route::Bend`]). Inside a glyph, `glyph.drawing` is an ordinary drawing,
//! and [`Glyph::set_edges`], [`Glyph::restyle`] and [`Glyph::contains`] manage
//! its highlighting.
//!
//! **After rendering**, [`TikzPicture::push`] and [`TikzPicture::push_raw`]
//! append anything to the picture.
//!
//! ```
//! use hccr::prelude::*;
//! use hccr::tikz::{Route, TikzLabel};
//!
//! let lattice = boolean(2);
//!
//! // A layout of your own: here, every element on a diagonal.
//! let diagonal = |p: &Poset| p.ids().map(|id| (id as f64, id as f64)).collect();
//! let mut drawing = lattice.drawing_with(diagonal);
//!
//! // Nudge one element and draw one edge straight.
//! drawing.positions[1].0 -= 1.0;
//! drawing.edges[0].route = Route::Straight;
//!
//! // In every glyph, mark the elements of the base lattice with bullets.
//! let tr = lattice.transfer_systems();
//! let glyphs = tr.drawing().for_each_glyph(|_id, glyph| {
//!     for node in &mut glyph.drawing.nodes {
//!         node.content = TikzLabel::raw("$\\bullet$");
//!     }
//! });
//! assert!(glyphs.to_tikz().render().contains("\\bullet"));
//! ```
//!
//! # Rendering
//!
//! [`PosetDrawing::to_tikz`]:
//!
//! 1. multiplies every position by [`spacing`](PosetDrawing::spacing), giving
//!    TikZ coordinates;
//! 2. bends each [`Route::Auto`] edge that passes through another node or lies
//!    along a shorter edge, by [`bend_angle`](PosetDrawing::bend_angle),
//!    alternating left and right. This is decided from the final positions,
//!    so moving nodes by hand still gives the right bends;
//! 3. emits nodes named `p0`, `p1`, … and then edges between those names, so
//!    edges stop at node borders;
//! 4. renders each node's content with [`NodeContent::to_label`]. A glyph, or
//!    any drawing used as node content, renders inline as `\tikz[…]{…}`, with
//!    unnamed nodes and edges between node centres, so its names cannot clash
//!    with the outer picture.
//!
//! Implement [`NodeContent`] to put any other kind of content at the nodes.
//!
//! A drawing always shows its whole poset: `positions` and `nodes` keep one
//! entry per element, and every edge names elements of the poset.
//! [`layout`](PosetDrawing::layout) and [`edges`](PosetDrawing::edges) check
//! this; a direct edit that breaks it panics when rendered.

mod drawing;
mod edge_routing;
mod glyph;
mod layout;
mod syntax;

pub use drawing::{Draw, DrawnEdge, Node, NodeContent, PosetDrawing, Route};
pub use glyph::Glyph;
pub use layout::{LayoutAlgorithm, LayoutCoordinates, LayoutLevels, LayoutOrdering, PosetLayout};
pub use syntax::{
    TikzCircle, TikzCoord, TikzDrawCommand, TikzItem, TikzLabel, TikzNode, TikzOptions, TikzPath,
    TikzPathOperation, TikzPicture, TikzScope, TikzStyle, escape_tikz,
};
