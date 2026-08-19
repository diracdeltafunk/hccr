use super::ToTikz;
use super::edge_routing::edge_bend_decisions;
use super::layout::{PosetLayoutAlgorithm, layout_with_covers};
use super::poset::{PosetTikzOptions, poset_to_tikz_with};
use super::syntax::*;
#[cfg(feature = "groups")]
use crate::g_lattice::GTransferLattice;
use crate::lattice::Lattice;
use crate::poset::{Edge, ElementId, Poset};
use crate::transfer_lattice::{TransferLattice, TransferPoset, TransferSystem};
use bitvec::prelude::*;
use std::collections::HashMap;

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

/// Cached geometry and drawing data for miniature spanning suborders.
///
/// One renderer is created for the ambient lattice of an outer transfer-system
/// diagram and reused for every node.  This keeps all glyphs in the same
/// coordinates and avoids recomputing the ranked layout for each transfer
/// system.
pub(in crate::tikz) struct SuborderGlyphRenderer<'a, A> {
    lattice: &'a Lattice<A>,
    coordinates: HashMap<ElementId, (f64, f64)>,
    proper_edges: Vec<Edge>,
    ambient_edges: Vec<Edge>,
    options: &'a TransferSystemGlyphOptions,
}

impl<'a, A> SuborderGlyphRenderer<'a, A> {
    pub(in crate::tikz) fn new(
        lattice: &'a Lattice<A>,
        options: &'a TransferSystemGlyphOptions,
    ) -> Self {
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

    pub(in crate::tikz) fn render_with<F>(&self, mut contains_relation: F) -> TikzPicture
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
