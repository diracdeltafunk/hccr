use hccr::g_lattice::SubgroupGLattice;
use hccr::tikz::{ToTikz, TransferSystemTikzOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let group = gap_sys::eval("CyclicGroup(4);")?;
    let subgroup_lattice = SubgroupGLattice::new(&group)?;
    let systems = subgroup_lattice.transfer_systems_containment()?;
    let mut options = TransferSystemTikzOptions::default();
    // The staggered middle branches remain legible with a tighter diagram.
    options.poset.y_spacing = 1.1;
    options.poset.debug_element_ids = true;
    let picture = systems.to_tikz_with(&options);
    let tex = format!(
        "\\documentclass[tikz,border=8pt]{{standalone}}\n\
         \\usepackage{{tikz}}\n\
         \\begin{{document}}\n\
         % Transfer systems for C_4, ordered by containment.\n\
         % There are {} transfer systems.\n\
         {}\n\
         \\end{{document}}\n",
        systems.size(),
        picture
    );

    print!("{tex}");
    eprintln!("{} transfer systems for C_4", systems.size());
    Ok(())
}
