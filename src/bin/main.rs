use hccr::cong_lattice;

fn main() {
    // Create a total order lattice with 5 elements
    let lattice = cong_lattice::CongLattice::total_order(5);

    // Print the lattice structure
    println!("{:?}", lattice);

    // Additional operations can be performed on the lattice here
}
