#![cfg(feature = "groups")]
// GAP values are deliberately retained inside the crate's Arc-backed
// mathematical presentations even though the GAP runtime is single-threaded.
#![allow(clippy::arc_with_non_send_sync)]

use gap_sys::GapValue;
use hccr::g_lattice::SubgroupGLattice;
use hccr::subgroup_morphism::SubgroupMaps;
use std::error::Error;
use std::sync::Arc;

#[test]
fn subgroup_image_and_preimage_maps_are_mathematically_correct() -> Result<(), Box<dyn Error>> {
    let (group, quotient, homomorphism) = cyclic_four_to_cyclic_two()?;
    let domain = Arc::new(SubgroupGLattice::new(&group)?);
    let codomain = Arc::new(SubgroupGLattice::new(&quotient)?);

    let maps = SubgroupMaps::new(&homomorphism, Arc::clone(&domain), Arc::clone(&codomain))?;

    assert!(!maps.is_injective());
    assert!(maps.is_surjective());

    let image_orders = (0..domain.lattice().size())
        .map(|subgroup| {
            let image = maps.image_map().apply(subgroup).unwrap();
            subgroup_order(codomain.subgroup(image).unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(image_orders, vec![1, 1, 2]);

    let preimage_orders = (0..codomain.lattice().size())
        .map(|subgroup| {
            let preimage = maps.preimage_map().apply(subgroup).unwrap();
            subgroup_order(domain.subgroup(preimage).unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(preimage_orders, vec![2, 4]);

    check_preimage_map_for_nonsurjective_inclusion()?;
    check_map_that_is_neither_injective_nor_surjective()?;

    Ok(())
}

fn check_preimage_map_for_nonsurjective_inclusion() -> Result<(), Box<dyn Error>> {
    let (group, overgroup, inclusion) = cyclic_two_into_symmetric_three()?;
    let domain = Arc::new(SubgroupGLattice::new(&group)?);
    let codomain = Arc::new(SubgroupGLattice::new(&overgroup)?);

    let maps = SubgroupMaps::new(&inclusion, domain, Arc::clone(&codomain))?;
    assert!(maps.is_injective());
    assert!(!maps.is_surjective());

    let mut preimage_orders = (0..codomain.lattice().size())
        .map(|subgroup| {
            let preimage = maps.preimage_map().apply(subgroup).unwrap();
            subgroup_order(maps.domain().subgroup(preimage).unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;
    preimage_orders.sort_unstable();

    // Only the image subgroup itself and S3 contain the chosen transposition.
    assert_eq!(preimage_orders, vec![1, 1, 1, 1, 2, 2]);

    Ok(())
}

fn check_map_that_is_neither_injective_nor_surjective() -> Result<(), Box<dyn Error>> {
    let values = gap_sys::eval(
        "(function() local g, f;
            g := Group((1,2,3,4));
            f := GroupHomomorphismByImages(
                g, g, [(1,2,3,4)], [(1,3)(2,4)]
            );
            return [g, f];
        end)();",
    )?;
    let (group, homomorphism) = {
        let gap = gap_sys::global()?;
        (gap.list_get(&values, 0)?, gap.list_get(&values, 1)?)
    };
    let lattice = Arc::new(SubgroupGLattice::new(&group)?);
    let maps = SubgroupMaps::new(&homomorphism, Arc::clone(&lattice), lattice)?;
    assert!(!maps.is_injective());
    assert!(!maps.is_surjective());

    let image_orders = (0..maps.domain().lattice().size())
        .map(|subgroup| {
            let image = maps.image_map().apply(subgroup).unwrap();
            subgroup_order(maps.codomain().subgroup(image).unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(image_orders, vec![1, 1, 2]);

    let preimage_orders = (0..maps.codomain().lattice().size())
        .map(|subgroup| {
            let preimage = maps.preimage_map().apply(subgroup).unwrap();
            subgroup_order(maps.domain().subgroup(preimage).unwrap())
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(preimage_orders, vec![2, 4, 4]);

    Ok(())
}

fn cyclic_four_to_cyclic_two() -> Result<(GapValue, GapValue, GapValue), Box<dyn Error>> {
    let values = gap_sys::eval(
        "(function() local g, h, f;
            g := Group((1,2,3,4));
            h := Group((1,2));
            f := GroupHomomorphismByImages(g, h, [(1,2,3,4)], [(1,2)]);
            return [g, h, f];
        end)();",
    )?;
    let gap = gap_sys::global()?;
    Ok((
        gap.list_get(&values, 0)?,
        gap.list_get(&values, 1)?,
        gap.list_get(&values, 2)?,
    ))
}

fn cyclic_two_into_symmetric_three() -> Result<(GapValue, GapValue, GapValue), Box<dyn Error>> {
    let values = gap_sys::eval(
        "(function() local g, h, f;
            g := Group((1,2));
            h := SymmetricGroup(3);
            f := GroupHomomorphismByImages(g, h, [(1,2)], [(1,2)]);
            return [g, h, f];
        end)();",
    )?;
    let gap = gap_sys::global()?;
    Ok((
        gap.list_get(&values, 0)?,
        gap.list_get(&values, 1)?,
        gap.list_get(&values, 2)?,
    ))
}

fn subgroup_order(subgroup: &GapValue) -> Result<usize, Box<dyn Error>> {
    let gap = gap_sys::global()?;
    let order = gap.call_global("Size", &[subgroup])?;
    Ok(gap.to_usize(&order)?)
}
