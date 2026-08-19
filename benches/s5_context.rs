use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hccr::g_lattice::SubgroupGLattice;
use std::time::Duration;

fn bench_s5_transfer_context(c: &mut Criterion) {
    c.bench_function("s5_group_to_transfer_context", |b| {
        b.iter(|| {
            let group = gap_sys::eval("SymmetricGroup(5);").expect("GAP constructs S_5");
            let subgroup_lattice =
                SubgroupGLattice::new(&group).expect("construct the subgroup G-lattice of S_5");
            black_box(subgroup_lattice.g_lattice().transfer_context())
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(15));
    targets = bench_s5_transfer_context
}
criterion_main!(benches);
