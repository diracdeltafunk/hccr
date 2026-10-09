use hccr::lattice::Lattice;
use hccr::morphism::{LatticeMap, PosetMap};
use hccr::transfer_morphism::{pullback_containment_map, pushforward_containment_map};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The quotient [2] -> [1] that identifies the bottom and middle elements.
    let c3 = Lattice::chain(2);
    let c2 = Lattice::chain(1);
    let quotient = LatticeMap::new(&c3, &c2, vec![0, 0, 1])?;

    let system = c3.transfer_system_generated_by([(1, 2)])?;
    let image = system.pushforward(&quotient)?;
    let inverse_image = image.pullback(&quotient)?;
    println!("system:      {system}");
    println!("pushforward: {image}");
    println!("pullback:    {inverse_image}");

    // Enumerate Tr(C3) and Tr(C2) only when an actual map of their
    // containment lattices is wanted.
    let tr_c3 = c3.transfer_systems();
    let tr_c2 = c2.transfer_systems();
    let push = pushforward_containment_map(&quotient, &tr_c3, &tr_c2)?;
    let pull = pullback_containment_map(&quotient, &tr_c2, &tr_c3)?;

    let system_id = tr_c3
        .id_of(&system)
        .expect("an enumerated transfer system has an element id");
    assert_eq!(tr_c2.system(push.apply(system_id)), &image);

    // The two containment maps satisfy f_*(R) <= S iff R <= f^*(S).
    for source in tr_c3.ids() {
        for target in tr_c2.ids() {
            assert_eq!(
                tr_c2.leq(push.apply(source), target),
                tr_c3.leq(source, pull.apply(target))
            );
        }
    }

    // A merely monotone map uses the same API. Here the atoms and top of B2
    // all map to the top of C2, so the map does not preserve meets.
    let b2 = Lattice::boolean(2);
    let monotone = PosetMap::new(&b2, &c2, vec![0, 1, 1, 1])?;
    let target_bottom = c2.trivial_transfer_system();

    // `pullback` is the greatest transfer system inside the raw inverse image
    // and remains right adjoint to pushforward.
    let right_adjoint = target_bottom.pullback(&monotone)?;
    assert!(right_adjoint.edges(false).is_empty());

    // The separately named operation closes above the raw inverse image.
    let generated = target_bottom.generated_inverse_image(&monotone)?;
    assert_eq!(generated.edges(false).len(), 5);
    println!("generated inverse image on B2: {generated}");

    Ok(())
}
