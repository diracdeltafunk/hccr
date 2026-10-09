use hccr::prelude::*;

fn main() -> hccr::Result<()> {
    // The sign homomorphism f: S_3 -> C_2, together with the subgroup
    // lattices of its source and range.
    let f = SubgroupMaps::from_gap(
        "NaturalHomomorphismByNormalSubgroup(SymmetricGroup(3), AlternatingGroup(3))",
    )?;
    let top_g = f.domain().complete_transfer_system();
    let top_quotient = f.codomain().complete_transfer_system();

    // Rubin's two adjunctions, applied pointwise. For this quotient the second
    // pair has the familiar names inflation and fixed points.
    println!("f_L:          {}", f.image_pushforward(&top_g)?);
    println!("f_R^-1:       {}", f.image_pullback(&top_quotient)?);
    println!("inflation:    {}", f.inflation(&top_quotient)?);
    println!("fixed points: {}", f.fixed_points(&top_g)?);
    Ok(())
}
