use hccr::lattice::Lattice;
use hccr::poset::{Edge, Poset};
use hccr::tikz::{transfer_system_lattice_to_tikz_with, transfer_system_tikz_options};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pentagon = Arc::new(Lattice::new(Poset::from_edges(
        ["0", "a", "b", "c", "1"].to_vec(),
        [
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 4),
            Edge::new(0, 3),
            Edge::new(3, 4),
        ],
    )?)?);

    let containment_lattice = pentagon.transfer_systems_containment()?;
    let mut tikz_options = transfer_system_tikz_options();
    tikz_options.poset.debug_element_ids = true;
    let tikz = transfer_system_lattice_to_tikz_with(&containment_lattice, &tikz_options).render();
    let tex = format!(
        "\\documentclass[tikz,border=8pt]{{standalone}}\n\
         \\usepackage{{tikz}}\n\
         \\begin{{document}}\n\
         % Transfer systems on the pentagon lattice N_5, ordered by containment.\n\
         % There are {} transfer systems.\n\
         {}\n\
         \\end{{document}}\n",
        containment_lattice.size(),
        tikz
    );

    print!("{tex}");
    eprintln!("{} transfer systems", containment_lattice.size());
    Ok(())
}
