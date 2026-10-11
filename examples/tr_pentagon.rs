use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    let pentagon = Lattice::from_covers(
        ["0", "a", "b", "c", "1"],
        [("0", "a"), ("a", "b"), ("b", "1"), ("0", "c"), ("c", "1")],
    )?;

    let tr = pentagon.transfer_systems();
    let drawing = tr.drawing().show_ids(true);
    print!("{}", drawing.to_tikz().to_standalone_document());
    eprintln!("{} transfer systems on the pentagon", tr.size());
    Ok(())
}
