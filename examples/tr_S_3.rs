use hccr::g_lattice::SubgroupGLattice;
use hccr::tikz::{GlyphNodeDisplay, ToTikz, TransferSystemTikzOptions};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let group = gap_sys::eval("SymmetricGroup(3);")?;
    let subgroup_lattice = SubgroupGLattice::new(&group)?;
    let systems = subgroup_lattice.transfer_systems_containment()?;
    let mut options = TransferSystemTikzOptions::default();
    options.glyph.node_display =
        GlyphNodeDisplay::raw(subgroup_lattice.subgroup_structure_descriptions_tex());
    let picture = systems.to_tikz_with(&options);
    let tex = format!(
        "\\documentclass[tikz,border=8pt]{{standalone}}\n\
         \\usepackage{{tikz}}\n\
         \\begin{{document}}\n\
         % Transfer systems on the diamond lattice, ordered by containment.\n\
         % There are {} transfer systems.\n\
         {}\n\
         \\end{{document}}\n",
        systems.size(),
        picture
    );

    fs::create_dir_all("target/tikz")?;
    fs::write("target/tikz/tr_S_3.tex", tex)?;

    println!("target/tikz/tr_S_3.tex");
    println!("{} transfer systems", systems.size());
    Ok(())
}
