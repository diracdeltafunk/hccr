use hccr::prelude::*;
use hccr::tikz::TransferSystemTikzOptions;

fn main() -> hccr::Result<()> {
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )?;

    let tr = pentagon.transfer_systems();
    let mut options = TransferSystemTikzOptions::default();
    options.poset.debug_element_ids = true;
    print!("{}", tr.to_tikz_with(&options).to_standalone_document());
    eprintln!("{} transfer systems on the pentagon", tr.size());
    Ok(())
}
