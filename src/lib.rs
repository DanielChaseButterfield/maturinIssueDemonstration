use pyo3::prelude::*;
use ndarray::prelude::*;

// Declare all submodules here
pub mod filters;
pub mod contours;
pub mod quad_fitting;
pub mod tag_detection;
pub mod tag_parameters;

#[pyclass]
struct SonarData {
    #[pyo3(get, set)]
    data: Vec<Vec<i32>>
}

#[pymethods]
impl SonarData {
    #[new]
    fn new(data: Vec<Vec<i32>>) -> Self {
        SonarData {
            data
        }
    }
}

// #[pyclass]
// struct AcTag {
//     #[pyo3(get, set)]
//     min_range: f64,
//     max_range: f64,
//     horizontal_aperture: f64,
//     tag_family: String,
//     tag_size: f64,
//     median_filter_kernel_radius: u64,
//     adaptive_threshold_kernel_radius: u64,
//     adaptive_threshold_offset: f64,
//     contours_tag_size_tolerance: f64,
//     contours_tag_area_tolerance: f64,
//     contours_min_tag_area_ratio: f64,
//     contours_reject_black_shapes: bool, 
//     contours_reject_white_shapes: bool, 
//     contours_reject_by_tag_size: bool,
//     contours_reject_by_area: bool,
//     quads_rng: String,
//     quads_points_per_line_ratio: f64,
//     quads_dist_for_inlier: f64,
//     quads_desired_inlier_ratio: f64,
//     quads_required_inlier_ratio: f64,
//     quads_parallel_threshold: f64,
//     decoding_num_bit_corrections: u64,
//     cpus_for_multithreading: Option<usize>,
// }

// impl Default for AcTag {
//     fn default() -> AcTag {
//         AcTag {
//             min_range: Default::default(),
//             max_range: Default::default(),
//             horizontal_aperture: Default::default(),
//             tag_family: Default::default(),
//             tag_size: Default::default(),
//             median_filter_kernel_radius: Default::default(),
//             adaptive_threshold_kernel_radius: Default::default(),
//             adaptive_threshold_offset: Default::default(),
//             contours_tag_size_tolerance: 0.2,
//             contours_tag_area_tolerance: 0.2,
//             contours_min_tag_area_ratio: 0.1,
//             contours_reject_black_shapes: true, 
//             contours_reject_white_shapes: false, 
//             contours_reject_by_tag_size: true,
//             contours_reject_by_area: true,
//             quads_rng: "Uniform".to_string(),
//             quads_points_per_line_ratio: 0.1,
//             quads_dist_for_inlier: 2.5,
//             quads_desired_inlier_ratio: 0.85,
//             quads_required_inlier_ratio: 0.8,
//             quads_parallel_threshold: 0.9,
//             decoding_num_bit_corrections: 4,
//             cpus_for_multithreading: None
//         }
//     }
// }

// #[pymethods]
// impl AcTag {
//     #[new]
//     fn new(min_range: f64,
//         max_range: f64,
//         horizontal_aperture: f64,
//         tag_family: String,
//         tag_size: f64,
//         median_filter_kernel_radius: u64,
//         adaptive_threshold_kernel_radius: u64,
//         adaptive_threshold_offset: f64,
//         contours_tag_size_tolerance: Option<f64>,
//         contours_tag_area_tolerance: f64,
//         contours_min_tag_area_ratio: f64,
//         contours_reject_black_shapes: bool, 
//         contours_reject_white_shapes: bool, 
//         contours_reject_by_tag_size: bool,
//         contours_reject_by_area: bool,
//         quads_rng: String,
//         quads_points_per_line_ratio: f64,
//         quads_dist_for_inlier: f64,
//         quads_desired_inlier_ratio: f64,
//         quads_required_inlier_ratio: f64,
//         quads_parallel_threshold: f64,
//         decoding_num_bit_corrections: u64,
//         cpus_for_multithreading: Option<usize>) -> Self {
//             let actagStruct: AcTag = Default::default();
//         // AcTag {
//         //     min_range,
//         //     max_range,
//         //     horizontal_aperture,
//         //     vertical_aperture,
//         //     tag_family,
//         //     tag_size,
//         //     median_filter_kernel_radius,
//         //     adaptive_threshold_kernel_radius,
//         //     adaptive_threshold_offset,
//         //     contours_tag_size_tolerance,
//         //     contours_tag_area_tolerance,
//         //     contours_min_tag_area_ratio,
//         //     contours_reject_black_shapes, 
//         //     contours_reject_white_shapes, 
//         //     contours_reject_by_tag_size,
//         //     contours_reject_by_area,
//         //     quads_rng,
//         //     quads_points_per_line_ratio,
//         //     quads_dist_for_inlier,
//         //     quads_desired_inlier_ratio,
//         //     quads_required_inlier_ratio,
//         //     quads_parallel_threshold,
//         //     decoding_num_bit_corrections,
//         //     cpus_for_multithreading
//         // }
//             actagStruct
//     }

//     fn run_detection(&self, data: Vec<Vec<i32>>) -> PyResult<Vec<Vec<f64>>> {

//         // Find the shape of the input image
//         let img_shape: (i64, i64) = (data.len().try_into().expect("Failed conversion from usize to i32."), 
//                                      data[0].len().try_into().expect("Failed conversion from usize to i32."));

//         // Filter the input image
//         let filtered: Vec<Vec<i32>> = filters::median_filter_multithread(data, 
//                                 self.median_filter_kernel_radius.try_into().expect("Failed conversion from u32 to usize."),
//                             self.cpus_for_multithreading);
//         let filtered_duplicate = filtered.clone();

//         // Binarize the filtered data
//         let binarized: Vec<Vec<i32>> = filters::adaptive_threshold_multithread(filtered, 
//                                         self.adaptive_threshold_kernel_radius.try_into().expect("Failed conversion fromm u32 to usize."), self.adaptive_threshold_offset, 
//                                     self.cpus_for_multithreading);

//         // Find contours
//         let contours: Vec<Vec<usize>> = contours::border_follow_haig_1992(binarized, true, self.min_range, 
//                                         self.max_range, self.tag_size, self.horizontal_aperture as f64, self.contours_tag_area_tolerance, 
//                                         self.contours_tag_size_tolerance);

//         // Convert usize in each contour to i64
//         let contours_for_quads: Vec<Vec<i64>> = contours.iter().map(|contour| {
//             contour.iter().map(|&x| x.try_into().expect("Failed conversion from usize to i64.")).collect()
//         }).collect();
                                     
//         // Fit quadrilaterals to contours
//         let quads: Vec<Vec<(i64, i64)>> = quad_fitting::fit_quadrilaterals_to_contours(contours_for_quads, 
//                                           img_shape, true, self.quads_points_per_line_ratio, 
//                                           self.quads_desired_inlier_ratio, self.quads_required_inlier_ratio,
//                                           self.quads_parallel_threshold, self.quads_dist_for_inlier);

//         // Detect tags from the family
//         let detected_tags: Vec<Vec<u64>> = tag_detection::detect_tags_from_family(filtered_duplicate, quads, 
//                                             self.tag_family.clone(), self.decoding_num_bit_corrections, 
//                                                         self.horizontal_aperture, self.min_range, self.max_range);

//         // Get the range and azimuth locations for each tag
//         let detected_tags_range_azi: Vec<Vec<f64>> = tag_detection::convert_tag_detections_to_range_azi_locations(
//                                                             detected_tags, img_shape, self.min_range, self.max_range, 
//                                                             self.horizontal_aperture);

//         // Return the result
//         Ok(detected_tags_range_azi)
//     }
// }

// Convenience function to convert from a vector of vectors to a ndarray
fn vec_tup_to_arr2_f32(vec: Vec<(f64, f64)>) -> ndarray::Array2<f64> {
    let mut arr = ndarray::Array2::zeros((vec.len(), 2));
    for i in 0..vec.len() {
        arr[[i, 0]] = vec[i].0;
        arr[[i, 1]] = vec[i].1;
    }
    arr
}

// Convenience function to convert from a ref vector of vectors to a ndarray
fn vec_tup_to_arr2_f32_ref(vec: &Vec<(f64, f64)>) -> ndarray::Array2<f64> {
    let mut arr = ndarray::Array2::zeros((vec.len(), 2));
    for i in 0..vec.len() {
        arr[[i, 0]] = vec[i].0;
        arr[[i, 1]] = vec[i].1;
    }
    arr
}

// // Convenience function to convert from a ndarray to a vector of vectors
// fn arr2_to_vec2_f32(arr: Array2<f32>) -> Vec<Vec<f32>> {
//     let mut vec = Vec::new();
//     for i in 0..arr.shape()[0] {
//         let mut row = Vec::new();
//         for j in 0..arr.shape()[1] {
//             row.push(arr[[i, j]]);
//         }
//         vec.push(row);
//     }
//     vec
// }

// Convenience function to convert from a vector of vectors to a ndarray
pub fn vec2_to_arr2_i32(vec: Vec<Vec<i32>>) -> Array2<i32> {
    let mut arr = Array2::zeros((vec.len(), vec[0].len()));
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            arr[[i, j]] = vec[i][j];
        }
    }
    arr
}

// Convenience function to convert from a ndarray to a vector of vectors
pub fn arr2_to_vec2_i32(arr: Array2<i32>) -> Vec<Vec<i32>> {
    let mut vec = Vec::new();
    for i in 0..arr.shape()[0] {
        let mut row = Vec::new();
        for j in 0..arr.shape()[1] {
            row.push(arr[[i, j]]);
        }
        vec.push(row);
    }
    vec
}

// Convenience function to convert from a ndarray to a vector of vectors
pub fn arr2_to_vec2_i32_ref(arr: &Array2<i32>) -> Vec<Vec<i32>> {
    let mut vec = Vec::new();
    for i in 0..arr.shape()[0] {
        let mut row = Vec::new();
        for j in 0..arr.shape()[1] {
            row.push(arr[[i, j]]);
        }
        vec.push(row);
    }
    vec
}

// Flood fill a four connected area based around a seed pixel. Changes the values to a new value.
#[pyfunction]
pub fn flood_fill(mut image: Vec<Vec<i32>>, sr: i32, sc: i32, new_val: i32) -> Vec<Vec<i32>> {
    use std::collections::VecDeque;
    use std::convert::TryFrom;

    let sr = usize::try_from(sr).unwrap();
    let sc = usize::try_from(sc).unwrap();

    let initial_val = image[sr][sc];

    if initial_val == new_val {
        return image;
    }

    let height = image.len();
    let width = image[0].len();

    let mut cells: VecDeque<(usize, usize)> = VecDeque::new();
    cells.push_back((sr, sc));

    while let Some((sr, sc)) = cells.pop_front() {
        let cell = &mut image[sr][sc];

        if *cell != initial_val {
            continue;
        }

        *cell = new_val;

        const OFFSETS: &[(usize, usize)] = &[(0, usize::MAX), (usize::MAX, 0), (0, 1), (1, 0)];

        for (delta_r, delta_c) in OFFSETS.iter().copied() {
            let new_r = sr.wrapping_add(delta_r);
            let new_c = sc.wrapping_add(delta_c);

            if new_r < height && new_c < width {
                cells.push_back((new_r, new_c));
            }
        }
    }

    image
}

// Convert a chain path into a vector of points
fn chain_path_to_vec(row_init: i32, col_init: i32, chain_path: Vec<i32>) -> Vec<(f32, f32)> {
    let mut coords = vec![];
    let mut r = row_init;
    let mut c = col_init;
    coords.push((r as f32, c as f32));
    // Decode path into coordinates
    for i in chain_path {
        match i {
            0 => c += 1,
            1 => {r -= 1; c += 1},
            2 => r -= 1,
            3 => {r -= 1; c -= 1},
            4 => c -= 1,
            5 => {r += 1; c -= 1},
            6 => r += 1,
            7 => {r += 1; c += 1}
            _ => panic!("Invalid chain path")
        }
        coords.push((r as f32, c as f32));
    };
    coords
}

#[pyfunction]
pub fn approx_poly(x: i32, y: i32, path: Vec<i32>, epsilon: f32) -> Vec<(i32, i32)> {
    use geo::geometry::LineString;
    use geo::algorithm::Simplify;

    let path_idxs = chain_path_to_vec(x, y, path);
    let path_linestring = LineString::from(path_idxs);
    let simplified_linestring = path_linestring.simplify(&epsilon);
    let simplified_coords = simplified_linestring.into_inner();
    let mut output_vec = vec![];
    for coord in simplified_coords {
        let (x, y) = coord.x_y();
        let (x, y) = (x as i32, y as i32);
        output_vec.push((x, y));
    }
    output_vec
}

/// A Python module implemented in Rust.
#[pymodule]
fn actag(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add the SonarData and AcTag classes
    // m.add_class::<SonarData>()?;
    // m.add_class::<AcTag>()?;

    // Add functions from filters module
    m.add_function(wrap_pyfunction!(filters::median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::median_filter_multithread, m)?)?;
    m.add_function(wrap_pyfunction!(filters::adaptive_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(filters::adaptive_threshold_multithread, m)?)?;
    m.add_function(wrap_pyfunction!(filters::bilateral_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::bilateral_filter_multithread, m)?)?;

    // Add functions from the contours module
    m.add_function(wrap_pyfunction!(contours::get_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::plot_contours, m)?)?;

    // Add functions from the quad_fitting module
    m.add_function(wrap_pyfunction!(quad_fitting::get_intersection_of_lines, m)?)?;
    m.add_function(wrap_pyfunction!(quad_fitting::least_squares_line_fit_python_wrap, m)?)?;
    m.add_function(wrap_pyfunction!(quad_fitting::get_random_point_and_fit_line_python_wrap, m)?)?;
    m.add_function(wrap_pyfunction!(quad_fitting::fit_quadrilaterals_to_contours, m)?)?;

    // Add functions from the tag_detection module
    m.add_function(wrap_pyfunction!(tag_detection::parse_apriltag_family, m)?)?;
    m.add_function(wrap_pyfunction!(tag_detection::get_data_bit_locations, m)?)?;
    m.add_function(wrap_pyfunction!(tag_detection::check_quad_for_tag_python_wrap, m)?)?;
    m.add_function(wrap_pyfunction!(tag_detection::decode_tags, m)?)?;
    m.add_function(wrap_pyfunction!(tag_detection::convert_tag_detections_to_range_azi_locations, m)?)?;

    // Return Ok
    Ok(())
}