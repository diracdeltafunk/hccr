use hccr::g_lattice::SubgroupGLattice;
use hccr::subgroup_morphism::SubgroupMaps;

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

    let subgroups = SubgroupGLattice::new(&group)?;
    let quotient_subgroups = SubgroupGLattice::new(&quotient)?;
    let f = SubgroupMaps::new(&homomorphism, &subgroups, &quotient_subgroups)?;

    let top_g = subgroups.complete_transfer_system();
    let top_quotient = quotient_subgroups.complete_transfer_system();

    // Rubin's two adjunctions, applied pointwise. For this quotient the second
    // pair has the familiar names inflation and fixed points.
    let left_image = f.image_pushforward(&top_g)?;
    let right_inverse = f.image_pullback(&top_quotient)?;
    let inflated = f.inflation(&top_quotient)?;
    let fixed = f.fixed_points(&top_g)?;

    println!("f_L:          {left_image}");
    println!("f_R^-1:       {right_inverse}");
    println!("inflation:    {inflated}");
    println!("fixed points: {fixed}");
    Ok(())
}
