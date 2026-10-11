use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let diamond = boolean(2);

    let tr = diamond.transfer_systems();
    let drawing = tr.drawing().show_ids(true);
    print!("{}", drawing.to_tikz().to_standalone_document());
    eprintln!("{} transfer systems", tr.size());
    Ok(())
}
