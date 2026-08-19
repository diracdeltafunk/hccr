use hccr::g_lattice::SubgroupGLattice;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let group = gap_sys::eval("SymmetricGroup(5);")?;
    let subgroup_lattice = SubgroupGLattice::new(&group)?;
    let setup_elapsed = started.elapsed();

    let started_counting = Instant::now();
    let count = subgroup_lattice.transfer_system_count();
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
