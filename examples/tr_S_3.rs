use hccr::prelude::*;
use hccr::tikz::TransferSystemTikzOptions;

fn main() -> hccr::Result<()> {
    let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
    let tr = s3.transfer_systems();

    let mut options = TransferSystemTikzOptions::default();
    // The staggered middle branches remain legible with a tighter diagram.
    options.poset.y_spacing = 2.2;
    print!("{}", tr.to_tikz_with(&options).to_standalone_document());
    eprintln!("{} transfer systems for S_3", tr.size());
    Ok(())
}
