use hccr::prelude::*;
use hccr::tikz::TransferSystemTikzOptions;

fn main() -> hccr::Result<()> {
    let diamond = boolean(2);

    let order = diamond.transfer_systems_composition_closed();
    let mut options = TransferSystemTikzOptions::default();
    options.poset.debug_element_ids = true;
    print!("{}", order.to_tikz_with(&options).to_standalone_document());
    eprintln!("{} transfer systems", order.size());
    eprintln!(
        "{} cover relations in the composition-closed ordering",
        order.cover_relations().len()
    );
    Ok(())
}
