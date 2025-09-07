use fcars::*;

// use plotly::{ImageFormat, Plot, Scatter};

// fn main() {
//     let num_objs = 15;
//     let num_attrs = 15;
//     let mut plot = Plot::new();
//     let mut result_unreduced_x: Vec<f64> = vec![];
//     let mut result_unreduced_y: Vec<usize> = vec![];
//     let mut result_reduced_x: Vec<f64> = vec![];
//     let mut result_reduced_y: Vec<usize> = vec![];
//     let mut result_combined_x: Vec<f64> = vec![];
//     let mut result_combined_y: Vec<usize> = vec![];
//     for density_halfpct in 1..200 {
//         for _ in 0..100 {
//             let density = f64::from(density_halfpct) * 0.005;
//             let context = FormalContext::random_with_density(num_objs, num_attrs, density);
//             if context.is_reduced() {
//                 result_reduced_x.push(context.density());
//                 result_reduced_y.push(context.num_concepts());
//             } else {
//                 result_unreduced_x.push(context.density());
//                 result_unreduced_y.push(context.num_concepts());
//             }
//             result_combined_x.push(context.density());
//             result_combined_y.push(context.num_concepts());
//         }
//     }
//     // plot.add_trace(
//     //     Scatter::new(result_reduced_x, result_reduced_y)
//     //         .name("Reduced")
//     //         .mode(plotly::common::Mode::Markers),
//     // );
//     // plot.add_trace(
//     //     Scatter::new(result_unreduced_x, result_unreduced_y)
//     //         .name("Unreduced")
//     //         .mode(plotly::common::Mode::Markers),
//     // );
//     plot.add_trace(
//         Scatter::new(result_combined_x, result_combined_y).mode(plotly::common::Mode::Markers),
//     );
//     let _ = plot.write_image("out.png", ImageFormat::PNG, 1280, 720, 1.0);
//     // plot.show();
// }

fn main() {
    let context = FormalContext::random_with_density(10, 12, 0.8);
    println!("Context:\n{}", context);
    let concepts = context.all_concepts();
    println!("Reduced? {}", context.is_reduced());
    for concept in concepts {
        println!("{}", concept);
    }
}
