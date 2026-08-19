// GAP values are deliberately retained inside the crate's Arc-backed
// mathematical presentations even though the GAP runtime is single-threaded.
#![allow(clippy::arc_with_non_send_sync)]

use hccr::g_lattice::SubgroupGLattice;
use hccr::g_transfer_morphism::{fixed_points, image_pullback, image_pushforward, inflation};
use hccr::subgroup_morphism::SubgroupMaps;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = gap_sys::eval(
        "(function() local g, n, h, f;
            g := SymmetricGroup(3);
            n := DerivedSubgroup(g);
            f := NaturalHomomorphismByNormalSubgroup(g, n);
            h := Range(f);
            return [g, h, f];
        end)();",
    )?;
    let (group, quotient, homomorphism) = {
        let gap = gap_sys::global()?;
        (
            gap.list_get(&values, 0)?,
            gap.list_get(&values, 1)?,
            gap.list_get(&values, 2)?,
        )
    };

    let subgroups = Arc::new(SubgroupGLattice::new(&group)?);
    let quotient_subgroups = Arc::new(SubgroupGLattice::new(&quotient)?);
    let f = SubgroupMaps::new(
        &homomorphism,
        Arc::clone(&subgroups),
        Arc::clone(&quotient_subgroups),
    )?;

    let tr_g = subgroups.transfer_systems_containment()?;
    let tr_quotient = quotient_subgroups.transfer_systems_containment()?;
    let top_g = tr_g.system(tr_g.top()).expect("top should exist");
    let top_quotient = tr_quotient
        .system(tr_quotient.top())
        .expect("top should exist");

    // Rubin's two adjunctions, applied pointwise. For this quotient the second
    // pair has the familiar names inflation and fixed points.
    let left_image = image_pushforward(&f, &top_g, tr_quotient.universe())?;
    let right_inverse = image_pullback(&f, &top_quotient, tr_g.universe())?;
    let inflated = inflation(&f, &top_quotient, tr_g.universe())?;
    let fixed = fixed_points(&f, &top_g, tr_quotient.universe())?;

    println!(
        "f_L: {}, f_R^-1: {}, inflation: {}, fixed points: {} relation orbits",
        left_image.raw().orbit_arrows().count_ones(),
        right_inverse.raw().orbit_arrows().count_ones(),
        inflated.raw().orbit_arrows().count_ones(),
        fixed.raw().orbit_arrows().count_ones(),
    );
    Ok(())
}
