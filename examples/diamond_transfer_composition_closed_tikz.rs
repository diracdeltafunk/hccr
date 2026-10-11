use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let diamond = boolean(2);

    let order = diamond.transfer_systems_composition_closed();
    let drawing = order.drawing().show_ids(true);
    print!("{}", drawing.to_tikz().to_standalone_document());
    eprintln!("{} transfer systems", order.size());
    eprintln!(
        "{} cover relations in the composition-closed ordering",
        order.covers().len()
    );
    Ok(())
}
