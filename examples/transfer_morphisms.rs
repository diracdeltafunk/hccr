use hccr::lattice::Lattice;
use hccr::morphism::{LatticeMap, PosetMap};
use hccr::poset::Edge;
use hccr::transfer_morphism::{
    generated_inverse_image, pullback, pullback_containment_map, pushforward,
    pushforward_containment_map,
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

    // A merely monotone map uses the same API. Here the atoms and top of B2
    // all map to the top of C2, so the map does not preserve meets.
    let b2 = Arc::new(Lattice::boolean(2)?);
    let monotone = PosetMap::between_lattices(&b2, &c2, vec![0, 1, 1, 1])?;
    let b2_transfers = Arc::clone(&b2).transfer_universe();
    let target_bottom = c2_transfers.generated_by(std::iter::empty::<Edge>())?;

    // `pullback` is the greatest transfer system inside the raw inverse image
    // and remains right adjoint to pushforward.
    let right_adjoint = pullback(&monotone, &target_bottom, &b2_transfers)?;
    assert!(right_adjoint.edges(false).is_empty());

    // The separately named operation closes above the raw inverse image.
    let generated = generated_inverse_image(&monotone, &target_bottom, &b2_transfers)?;
    assert_eq!(generated.edges(false).len(), 5);

    Ok(())
}
