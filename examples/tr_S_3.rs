use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let s3 = SubgroupGLattice::from_gap("SymmetricGroup(3)")?;
    let tr = s3.transfer_systems();

    print!("{}", tr.to_tikz().to_standalone_document());
    eprintln!("{} transfer systems for S_3", tr.size());
    Ok(())
}
