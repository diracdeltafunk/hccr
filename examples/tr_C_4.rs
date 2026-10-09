use hccr::prelude::*;
use hccr::tikz::TransferSystemTikzOptions;

fn main() -> hccr::Result<()> {
    let c4 = SubgroupGLattice::from_gap("CyclicGroup(4)")?;
    let tr = c4.transfer_systems();

    let mut options = TransferSystemTikzOptions::default();
    // The staggered middle branches remain legible with a tighter diagram.
    options.poset.y_spacing = 1.1;
    options.poset.debug_element_ids = true;
    print!("{}", tr.to_tikz_with(&options).to_standalone_document());
    eprintln!("{} transfer systems for C_4", tr.size());
    Ok(())
}
