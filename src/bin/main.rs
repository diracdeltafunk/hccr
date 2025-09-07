// use bitvec::prelude::*;
use hccr::fca::*;

fn main() {
    let file = std::fs::File::open("examples/[13].dat").expect("Couldn't open .dat file!");
    let mut context = FormalContext::from_dat(file);
    // println!("Context:\n{}", context);
    context.reduce();
    // println!("Reduced Context:\n{}", context);
    // let concepts = context.all_concepts();
    println!("Found {} concepts", context.num_concepts());
    // for concept in concepts {
    //     println!("{}", concept);
    // }
}
