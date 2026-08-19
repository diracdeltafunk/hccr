//! TikZ rendering for finite posets, lattices, and transfer-system diagrams.
//!
//! The module provides a small typed TikZ abstract syntax tree together with
//! convenience renderers for Hasse diagrams. The default layout uses centered
//! feasible heights and layered crossing reduction, retaining the best
//! straight-line geometry found among the ranked and optimized candidates.
//! Cover relations are drawn unless full relations are requested.

mod edge_routing;
mod layout;
mod poset;
mod syntax;
mod transfer;

pub use layout::PosetLayoutAlgorithm;
pub use poset::{PosetTikzOptions, poset_to_tikz_with};
pub use syntax::{
    TikzCircle, TikzCoord, TikzDrawCommand, TikzItem, TikzLabel, TikzNode, TikzOptions, TikzPath,
    TikzPathOperation, TikzPicture, TikzScope, TikzStyle, escape_tikz,
};
pub use transfer::{
    GlyphNodeDisplay, RelationDisplay, TransferSystemGlyphOptions, TransferSystemTikzOptions,
    transfer_system_lattice_to_tikz, transfer_system_lattice_to_tikz_with,
    transfer_system_order_to_tikz, transfer_system_order_to_tikz_with,
    transfer_system_tikz_options,
};
#[cfg(feature = "groups")]
pub use transfer::{g_transfer_system_lattice_to_tikz, g_transfer_system_lattice_to_tikz_with};

/// A type that can be rendered as a TikZ picture.
///
/// Each renderable type chooses its own associated option type. Call
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
