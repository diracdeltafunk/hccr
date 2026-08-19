# hccr [![Rust](https://github.com/diracdeltafunk/hccr/actions/workflows/rust.yml/badge.svg)](https://github.com/diracdeltafunk/hccr/actions/workflows/rust.yml) [![Rust with GAP](https://github.com/diracdeltafunk/hccr/actions/workflows/gap.yml/badge.svg)](https://github.com/diracdeltafunk/hccr/actions/workflows/gap.yml)

**H**omotopical **C**ombinatorics **C**omputations in **R**ust.

`hccr` is a research-oriented Rust library for finite order-theoretic
calculations arising in homotopical combinatorics. It constructs and studies:

- finite posets and lattices;
- transfer and cotransfer systems;
- model structures and weak factorization systems;
- maps induced by monotone and lattice morphisms;
- equivariant transfer systems for finite groups, via GAP; and
- TikZ representations of the resulting orders and lattices.

The project is under active development. The API may change as the mathematical
interface settles.

## Getting started

Until the crate is published, depend on the GitHub repository:

```toml
[dependencies]
hccr = { git = "https://github.com/diracdeltafunk/hccr" }
```

For example, the Boolean lattice of rank two has ten transfer systems:

```rust
use hccr::lattice::Lattice;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lattice = Arc::new(Lattice::boolean(2)?);
    let systems = lattice.transfer_systems_containment()?;

    assert_eq!(systems.size(), 10);
    Ok(())
}
```

More complete programs are available in [`examples/`](examples), including
transfer-system morphisms, model calculations for groups, and TikZ output. Run
an ordinary example with:

```console
cargo run --example transfer_morphisms
```

## Equivariant calculations

The optional `groups` feature uses
[`gap-sys`](https://github.com/diracdeltafunk/gap-sys) and requires a working
GAP/libgap installation:

```toml
[dependencies]
hccr = { git = "https://github.com/diracdeltafunk/hccr", features = ["groups"] }
```

```console
cargo run --features groups --example tr_S_3
```

The dependency currently follows an unreleased `gap-sys` development branch.

## Citation

If you use `hccr` in academic work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).
