use hccr::prelude::*;
use hccr::tikz::TransferSystemTikzOptions;

fn main() -> hccr::Result<()> {
    let diamond = boolean(2);

    let tr = diamond.transfer_systems();
    let mut options = TransferSystemTikzOptions::default();
    options.poset.debug_element_ids = true;
    print!("{}", tr.to_tikz_with(&options).to_standalone_document());
    eprintln!("{} transfer systems", tr.size());
    Ok(())
}
