use hccr::lattice::Lattice;
use hccr::tikz::{transfer_system_order_to_tikz_with, transfer_system_tikz_options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let diamond = Lattice::from_covers(
        ["0", "a", "b", "1"],
        [("0", "a"), ("0", "b"), ("a", "1"), ("b", "1")],
    )?;

    let composition_closed_order = diamond.transfer_systems_composition_closed();
    let mut tikz_options = transfer_system_tikz_options();
    tikz_options.poset.debug_element_ids = true;
    let tikz =
        transfer_system_order_to_tikz_with(&composition_closed_order, &tikz_options).render();
    let tex = format!(
        "\\documentclass[tikz,border=8pt]{{standalone}}\n\
         \\usepackage{{tikz}}\n\
         \\begin{{document}}\n\
         % Transfer systems on the diamond lattice, ordered by the composition-closed relation.\n\
         % There are {} transfer systems.\n\
         {}\n\
         \\end{{document}}\n",
        composition_closed_order.size(),
        tikz
    );

    print!("{tex}");
    eprintln!("{} transfer systems", composition_closed_order.size());
    eprintln!(
        "{} cover relations in the composition-closed ordering",
        composition_closed_order.cover_relations().len()
    );
    Ok(())
}
