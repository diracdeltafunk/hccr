use std::fmt;

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
