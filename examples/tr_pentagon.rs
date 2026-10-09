use hccr::lattice::Lattice;
use hccr::tikz::{transfer_system_lattice_to_tikz_with, transfer_system_tikz_options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )?;

    let containment_lattice = pentagon.transfer_systems();
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
