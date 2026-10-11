# hccr [![Rust](https://github.com/diracdeltafunk/hccr/actions/workflows/rust.yml/badge.svg)](https://github.com/diracdeltafunk/hccr/actions/workflows/rust.yml) [![Rust with GAP](https://github.com/diracdeltafunk/hccr/actions/workflows/gap.yml/badge.svg)](https://github.com/diracdeltafunk/hccr/actions/workflows/gap.yml)

**H**omotopical **C**ombinatorics **C**omputations in **R**ust.

`hccr` is a research-oriented Rust library for finite order-theoretic
calculations arising in homotopical combinatorics. It constructs and studies:

- finite posets and lattices;
- transfer and cotransfer systems;
- model structures and weak factorization systems;
- maps induced by monotone and lattice morphisms;
- equivariant transfer systems for finite groups, via GAP;
- concept lattices of formal contexts, via [`fcars`](https://crates.io/crates/fcars); and
- TikZ drawings of the resulting orders and lattices.

The project is under active development. The API may change as the mathematical
interface settles.

## Getting started

Until the crate is published, depend on the GitHub repository:

```toml
[dependencies]
hccr = { git = "https://github.com/diracdeltafunk/hccr" }
```

Everything needed for everyday work comes from `hccr::prelude`:

```rust
use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    // The horizontal join [3] * [3] * [3] has 298 transfer systems.
    let c3 = chain(3);
    let joined = horizontal_join([&c3, &c3, &c3])?;
    assert_eq!(joined.transfer_systems().size(), 298);

    // Lattices can also be given by labels and cover relations.
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )?;
    let t = pentagon.transfer_system_generated_by([pentagon.edge("a", "1")?])?;
    println!("{t}"); // {0 -> c, a -> b, a -> 1}
    Ok(())
}
```

A few things to know:

- **Elements have labels.** Every element has a printable [`Label`], unique in
  its lattice: an integer, a string, a tuple such as `(1, 2)`, a set such as
  `{0, 2}`, or a subgroup. Look elements up with `lattice.id(label)?`.
  Constructions label their elements predictably; for example, in a horizontal
  join the element `x` of the `i`th factor is `(i, x)`.
- **Objects are cheap to clone.** `Lattice`, `Poset`, transfer systems, maps,
  and so on are handles to shared data, so `.clone()` costs nothing and there
  is never a need for `Arc` or `Rc`.
- **One error type.** Every error converts into `hccr::Error`, so
  `fn main() -> hccr::Result<()>` with `?` works throughout.

## Drawing

Posets, lattices and lattices of transfer systems can be drawn as TikZ
pictures. A lattice of transfer systems shows each transfer system as a small
picture of the underlying lattice, with its relations highlighted:

```rust
use hccr::prelude::*;
use hccr::tikz::PosetLayout;

fn main() {
    let tr = boolean(2).transfer_systems();

    // The default drawing, as a complete LaTeX document.
    println!("{}", tr.to_tikz().to_standalone_document());

    // Drawings are plain data, so they can be adjusted before rendering:
    // here, a different layout, wider spacing, and element ids.
    let drawing = tr
        .drawing_with(PosetLayout::BALANCED)
        .spacing(3.0, 3.0)
        .show_ids(true);
    println!("{}", drawing.to_tikz());
}
```

The [`tikz`](src/tikz.rs) module documentation describes the available
layouts and every way to adjust a drawing.

## Documentation and examples

Until the crate is published, build the API documentation locally:

```console
cargo doc --open                    # add --features groups for the GAP parts
```

More complete programs are available in [`examples/`](examples), including
horizontal joins, transfer-system morphisms, model calculations for groups, and
TikZ output. Run one with:

```console
cargo run --example horizontal_join
```

## Equivariant calculations

The optional `groups` feature uses
[`gap-sys`](https://github.com/diracdeltafunk/gap-sys) and requires a working
GAP/libgap installation (on Debian or Ubuntu, `apt install gap libgap-dev`):

```toml
[dependencies]
hccr = { git = "https://github.com/diracdeltafunk/hccr", features = ["groups"] }
```

```rust
use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
    println!("{} transfer systems for S_3", s3.transfer_system_count());

    // Rubin's functors along the sign homomorphism S_3 -> C_2.
    let sign = SubgroupMaps::from_gap(
        "NaturalHomomorphismByNormalSubgroup(SymmetricGroup(3), AlternatingGroup(3))",
    )?;
    let inflated = sign.inflation(&sign.codomain().complete_transfer_system())?;
    println!("{inflated}");
    Ok(())
}
```

GAP itself is re-exported as `hccr::gap`, so no separate dependency is
needed. Run an example with:

```console
cargo run --features groups --example tr_S_3
```

The dependency currently follows an unreleased `gap-sys` development branch.

## Citation

If you use `hccr` in academic work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

[`Label`]: src/label.rs
