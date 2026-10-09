use hccr::g_lattice::SubgroupGLattice;
use hccr::tikz::{ToTikz, TransferSystemTikzOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let group = gap_sys::eval("SymmetricGroup(3);")?;
    let subgroup_lattice = SubgroupGLattice::new(&group)?;
    let systems = subgroup_lattice.transfer_systems();
    let mut options = TransferSystemTikzOptions::default();
    // The staggered middle branches remain legible with a tighter diagram.
    options.poset.y_spacing = 2.2;
    let picture = systems.to_tikz_with(&options);
    let tex = format!(
        "\\documentclass[tikz,border=8pt]{{standalone}}\n\
         \\usepackage{{tikz}}\n\
         \\begin{{document}}\n\
         % Transfer systems for S_3, ordered by containment.\n\
         % There are {} transfer systems.\n\
         {}\n\
         \\end{{document}}\n",
        systems.size(),
        picture
    );

    print!("{tex}");
    eprintln!("{} transfer systems for S_3", systems.size());
    Ok(())
}
