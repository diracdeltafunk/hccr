//! TikZ drawings of finite posets, lattices, and posets of transfer systems.
//!
//! Drawing goes through three stages, each of which can be stopped at and
//! adjusted:
//!
//! 1. a [`LayoutAlgorithm`] places the elements: a [`PosetLayout`], or any
//!    closure from a poset to positions;
//! 2. the result is a [`PosetDrawing`], plain data listing the position and
//!    node of each element and the edges to draw, which can be edited freely;
//! 3. [`PosetDrawing::to_tikz`] renders it as a [`TikzPicture`].
//!
//! Everything drawable implements [`Draw`], which supplies a default drawing:
//!
//! ```
//! use hccr::prelude::*;
//! use hccr::tikz::PosetLayout;
//!
//! let tr = chain(2).transfer_systems();
//! // The default drawing.
//! let picture = tr.to_tikz();
//! // A different layout, wider spacing, and glyphs drawn on the Hasse
//! // diagram of the base lattice.
//! let picture = tr
//!     .drawing_with(PosetLayout::BALANCED)
//!     .spacing(3.0, 3.0)
//!     .glyph_edges(tr.base_lattice().covers())
//!     .to_tikz();
//! ```
//!
//! Plain posets and lattices put their labels at the nodes; posets of transfer
//! systems put a [`Glyph`], a small drawing of the base lattice with the
//! transfer system highlighted.

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
