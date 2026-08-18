use hccr::lattice::Lattice;
use hccr::morphism::LatticeMap;
use hccr::poset::Edge;
use hccr::transfer_morphism::{
    pullback, pullback_containment_map, pushforward, pushforward_containment_map,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The quotient C3 -> C2 that identifies the bottom and middle elements.
    let c3 = Arc::new(Lattice::chain(2)?);
    let c2 = Arc::new(Lattice::chain(1)?);
    let quotient = LatticeMap::new(Arc::clone(&c3), Arc::clone(&c2), vec![0, 0, 1])?;

    let c3_transfers = Arc::clone(&c3).transfer_universe();
    let c2_transfers = Arc::clone(&c2).transfer_universe();
    let system = c3_transfers.generated_by([Edge::new(1, 2)])?;

    let image = pushforward(&quotient, &system, &c2_transfers)?;
    let inverse_image = pullback(&quotient, &image, &c3_transfers)?;
    println!("pushforward: {:?}", image.edges(false));
    println!("pullback: {:?}", inverse_image.edges(false));

    // Enumerate Tr(C3) and Tr(C2) only when an actual map of their
    // containment posets is wanted.
    let tr_c3 = c3_transfers.containment_lattice()?;
    let tr_c2 = c2_transfers.containment_lattice()?;
    let push = pushforward_containment_map(&quotient, &tr_c3, &tr_c2)?;
    let pull = pullback_containment_map(&quotient, &tr_c2, &tr_c3)?;

    let system_id = tr_c3
        .systems()
        .position(|candidate| candidate == system)
        .expect("an enumerated transfer system has an element id");
    let image_id = push
        .apply(system_id)
        .expect("the induced map is defined on every element id");
    assert_eq!(push.codomain().element(image_id), Some(&image));

    // The two containment maps satisfy f_*(R) <= S iff R <= f^*(S).
    for source in 0..tr_c3.size() {
        for target in 0..tr_c2.size() {
            assert_eq!(
                tr_c2.as_poset().leq(push.apply(source).unwrap(), target),
                tr_c3.as_poset().leq(source, pull.apply(target).unwrap())
            );
        }
    }

    Ok(())
}
