use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let c4 = SubgroupGLattice::from_gap("CyclicGroup(4)")?;
    let tr = c4.transfer_systems();

    let drawing = tr.drawing().show_ids(true);
    print!("{}", drawing.to_tikz().to_standalone_document());
    eprintln!("{} transfer systems for C_4", tr.size());
    Ok(())
}
