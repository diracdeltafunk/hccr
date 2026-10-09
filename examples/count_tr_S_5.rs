use hccr::prelude::*;
use std::time::Instant;

fn main() -> hccr::Result<()> {
    let started = Instant::now();
    let s5 = SubgroupGLattice::from_gap("SymmetricGroup(5)")?;
    let setup_elapsed = started.elapsed();

    let started_counting = Instant::now();
    let count = s5.transfer_system_count();
    let counting_elapsed = started_counting.elapsed();

    println!("{count} transfer systems for S_5");
    println!("subgroup-lattice and action setup: {setup_elapsed:.3?}");
    println!("transfer-system enumeration:      {counting_elapsed:.3?}");
    println!(
        "total:                            {:.3?}",
        started.elapsed()
    );
    Ok(())
}
