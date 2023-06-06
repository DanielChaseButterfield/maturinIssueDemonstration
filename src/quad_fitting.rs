use pyo3::prelude::*;
use ndarray;
use randomkit::{Rng, Sample}; // randomkit is approximately equivalent to Numpy's numpy.random module
use randomkit::dist::Uniform;
use nalgebra::{self as na, OMatrix, OVector, U2};
use pyo3::exceptions::{PyValueError};
use rug::float::Round;
use rug::Float;

/// Rust equivalent of :func:`quad_fitting.get_intersection_of_lines`.
/// 
/// :param line_1: Slope and intercept of the first line.
/// :type line_1: (f64, f64)
/// :param line_2: Slope and intercept of the second line.
/// :type line_2: (f64, f64)
/// :param line_1_row: ``true`` if line 1 is in ``row`` form, or ``false`` for ``column`` form.
/// :type line_1_row: bool
/// :param line_2_row: ``true`` if line 2 is in ``row`` form, or ``false`` for ``column``.
/// :type line_2_row: bool
/// :return: The intersection of the two lines in the form ``(row, column)``, rounded to the nearest row and column index.
/// :rtype: PyResult<(i64, i64)>
/// :raises ValueError: Raised if the two lines have the same slope, resulting in either 0 or infinite intersections.
///
#[pyfunction]
pub fn get_intersection_of_lines(line_1: (f64, f64), line_2: (f64, f64), line_1_row: bool, line_2_row: bool) -> PyResult<(i64, i64)> {
    // Initalize variables
    let row;
    let column;
    let slope_1 = line_1.0;
    let slope_2 = line_2.0;
    let intercept_1 = line_1.1;
    let intercept_2 = line_2.1;

    // Check if there is no intersection point or infinite. If so, then quad detection should
    // fail, so return an itersection point outside the bounds of the image.
    if slope_1 == slope_2 {
        return Err(PyValueError::new_err("0 or infinite intersections for lines with identical slope".to_string()));
    }

    // Both equations are in the form column = slope row + intercept (y = m x + b)
    if line_1_row == false && line_2_row == false {
        row = (intercept_2 - intercept_1) / (slope_1 - slope_2);
        column = slope_1 * row + intercept_1;
    }
    // Both equations are in the form row = slope column + intercept (x = m y + b)
    else if line_1_row == true && line_2_row == true {
        column = (intercept_2 - intercept_1) / (slope_1 - slope_2);
        row = slope_1 * column + intercept_1;
    }
    // The first equation is in the form column = slope row + intercept (y = m x + b) 
    // while the second is in the form row = slope column + intercept (x = m y + b)
    else if line_1_row == false && line_2_row == true {
        column = (slope_1 * intercept_2 + intercept_1) / (1.0 - slope_1 * slope_2);
        row = slope_2 * column + intercept_2;
    }
    // The first equation is in the form row = slope column + intercept (x = m y + b)
    // while the second is in the form column = slope row + intercept (y = m x + b)
    else {
        row = (slope_1 * intercept_2 + intercept_1) / (1.0 - slope_1 * slope_2);
        column = slope_2 * row + intercept_2;
    }

    // Return the intersection point. Round to nearest even, to match Python code.
    let row_rounded = Float::with_val(64, row);
    let column_rounded = Float::with_val(64, column);
    Ok((row_rounded.to_i32_saturating_round(Round::Nearest).unwrap() as i64, 
        column_rounded.to_i32_saturating_round(Round::Nearest).unwrap() as i64))
}


/// This function finds the smallest distance between a point and a line segment.
/// 
/// Arguments: point - the row and column number of the point
///            line_point_1 - the row and column number of the first point of the line segment
///            line_point_2 - the row and column number of the second point of the line segment
///
/// Returns: f64 - The smallest distance between the point and the line segment
///
pub fn distance_between_point_and_line_segment(point: (f64, f64), line_point_1: (f64, f64), line_point_2: (f64, f64)) -> f64 {
    // Initalize variables
    let rp = point.0;
    let cp = point.1;
    let r1 = line_point_1.0;
    let c1 = line_point_1.1;
    let r2 = line_point_2.0;
    let c2 = line_point_2.1;

    // Vector for the line segment
    let mut vec_12 = ndarray::Array1::zeros(2);
    vec_12[0] = r2 - r1; 
    vec_12[1] = c2 - c1;

    // Vector for line_point_1 to point
    let mut vec_1p = ndarray::Array1::zeros(2);
    vec_1p[0] = rp - r1;
    vec_1p[1] = cp - c1;

    // Vector for line_point_2 to point
    let mut vec_2p = ndarray::Array1::zeros(2);
    vec_2p[0] = rp - r2; 
    vec_2p[1] = cp - c2;

    // Dot products
    let dp_12_1p = vec_12.dot(&vec_1p);
    let dp_12_2p = vec_12.dot(&vec_2p);

    // Handle when closest point is line_point_2
    let dist;
    if dp_12_2p > 0.0 {
        dist = (vec_2p[0].powi(2) + vec_2p[1].powi(2)).sqrt();
    }
    // Handle when the closest point is line_point_1
    else if dp_12_1p < 0.0 {
        dist = (vec_1p[0].powi(2) + vec_1p[1].powi(2)).sqrt();
    }
    // Handle when the closest point is between line_point_1 and line_point_2
    else {
        dist = (vec_12[0] * vec_1p[1] - vec_12[1] * vec_1p[0]).abs() / (vec_12[0].powi(2) + vec_12[1].powi(2)).sqrt();
    }

    // Return the distance
    dist
}

/// Same as above, but with integers for the line_points
pub fn distance_between_point_and_line_segment_i64(point: (f64, f64), line_point_1: (i64, i64), line_point_2: (i64, i64)) -> f64 {
    distance_between_point_and_line_segment(point, (line_point_1.0 as f64, line_point_1.1 as f64), 
                                        (line_point_2.0 as f64, line_point_2.1 as f64))
}

/// This function finds the smallest distance between a point and a line.
/// 
/// Arguments: point - the row and column number of the point
///            slope - the slope of the line
///            intercept - the intercept of the line
///            is_row_form - true if the line is in 'row" form, or false for 'column'
///
/// Returns: f64 - The smallest distance between the point and the line
///
pub fn distance_between_point_and_line(point: (f64, f64), slope: f64, intercept: f64, is_row_form: bool) -> f64{
    // Initalize variables
    let r = point.0;
    let c = point.1;

    // For r = slope c + intercept (x = m y + b)
    if is_row_form == true {
        return (intercept + slope * c - r).abs() / (1.0 + slope.powi(2)).sqrt();
    }
    // For c = slope r + intercept (y = m x + b)
    else {
        return (intercept + slope * r - c).abs() / (1.0 + slope.powi(2)).sqrt();
    }
}

/// Fit a line to the points using least squares, trying forms y = m x + b and 
/// x = m y + b to see which is the better fit. This is necessary because linear 
/// regression for vertical lines is very poor with the form y = m x + b; likewise 
/// for horizontal lines we do not want to use the form x = m y + b.
/// 
/// :param line_points: The points used to fit a line to.
/// :type line_points: ndarray::Array2<f64>
/// :return: A vector containing the fitted line, containing the form (1 for 'row', 
/// 0 for 'column), the slope, and the intercept.
/// :rtype: Vec<f64>
///
pub fn least_squares_line_fit(line_points: ndarray::Array2<f64>) -> Vec<f64> {
    // Make necessary transformations
    let mut vec_a1: Vec<f64> = line_points.clone().into_raw_vec();
    for i in 0..vec_a1.len() {
        if i % 2 != 0 {
            vec_a1[i] = 1.0;
        }
    }

    let mut vec_a2: Vec<f64> = line_points.clone().into_raw_vec();
    for i in 0..vec_a2.len() {
        if i % 2 == 0 {
            vec_a2[i] = vec_a2[i+1];
        } else {
            vec_a2[i] = 1.0;
        }
    }

    let mut vec_b1 = Vec::new();
    for (_i, val) in line_points.column(1).into_iter().enumerate() {
        vec_b1.push(val.clone());
    }

    let mut vec_b2 = Vec::new();
    for (_i, val) in line_points.column(0).into_iter().enumerate() {
        vec_b2.push(val.clone());
    }
    
    // y = m x + b
    let a1 = OMatrix::<f64, na::Dynamic, U2>::from_row_slice(&vec_a1);
    let b1 = OVector::<f64, na::Dynamic>::from_row_slice(&vec_b1);
    let results1 = lstsq::lstsq(&a1, &b1, 1e-30).unwrap();
    let slope_x = results1.solution[0];
    let intercept_x = results1.solution[1];

    // x = m y + b
    let a2 = OMatrix::<f64, na::Dynamic, U2>::from_row_slice(&vec_a2);
    let b2 = OVector::<f64, na::Dynamic>::from_row_slice(&vec_b2);
    let results2 = lstsq::lstsq(&a2, &b2, 1e-30).unwrap();
    let slope_y = results2.solution[0];
    let intercept_y = results2.solution[1];

    // Calculate sum of squares for each line approximation
    let mut err_x = 0.0;
    let mut err_y = 0.0;
    for i in 0..line_points.shape()[0] {
        err_x += (line_points[[i, 1]] - (slope_x * line_points[[i, 0]] + intercept_x)).powi(2);
        err_y += (line_points[[i, 0]] - (slope_y * line_points[[i, 1]] + intercept_y)).powi(2);
    }

    // Round to 8 decimal places, so that the functionality is identical to the Python code
    err_x = (err_x * 1e8 as f64).round() / 1e8;
    err_y = (err_y * 1e8 as f64).round() / 1e8;
    
    // Determine which line is a better approximation, and set the output values appropriately
    let mut results: Vec<f64> = Vec::new();
    if err_x < err_y {
        results.push(0.0); // Column form
        results.push(slope_x);
        results.push(intercept_x);
    }
    else {
        results.push(1.0);
        results.push(slope_y);
        results.push(intercept_y);
    }

    // Return the line parameters
    results
}

/// Rust equivalent of :func:`quad_fitting.least_squares_line_fit`.
/// 
/// :param line_points: The points used to fit a line.
/// :type line_points: Vec<(f64, f64)>
/// :return: A vector containing the fitted line in the following format: ``[equation form, slope, intercept]``. The ``equation form`` will be 1 for ``row`` form or 0 for ``column`` form.
/// :rtype: Vec<f64>
///
#[pyfunction]
pub fn least_squares_line_fit_python_wrap(line_points: Vec<(f64, f64)>) -> Vec<f64> {
    least_squares_line_fit(crate::vec_tup_to_arr2_f32(line_points))
}

/// Randomly sample 'points_for_line' points from 'shape_points' with the provided random number generator, 
/// and then fit a line to these points using least squares. If 'use_same_random_vals' is true, roll to the 
/// 'supplied_roll_value' instead of randomly.
/// 
/// :param shape_points: All of the points in the contour, to randomly sample from.
/// :type shape_points: &mut Vec<(f64, f64)>
/// :param points_for_line: The number of points to sample for a line.
/// :type points_for_line: u64
/// :param use_same_random_vals: If true, sample points starting at the 'supplied_roll_value' instead of randomly.
/// :type use_same_random_vals: bool
/// :param rand: A random number generator used to sample randomly.
/// :type rand: &mut Rng
/// :param supplied_roll_value: If 'use_same_random_vals' is true, then sample points starting from this value.
/// :type supplied_roll_value: u32
/// :return: A vector containing the fitted line and the leftover points. The first three values represent the fitted line, being the form (1 for 'row', 
/// 0 for 'column), the slope, and the intercept. The remaining values are the row and column values of leftover points from shape_points that 
/// were not used to fit this line.
/// :rtype: Vec<f64>
///
pub fn get_random_point_and_fit_line(shape_points: &mut Vec<(f64, f64)>, points_for_line: u64, 
                                     use_same_random_vals: bool, rand: &mut Rng, supplied_roll_value: u32) -> Vec<f64> {
    // Get the rows/number of points left in the shape
    let rows = shape_points.len() as u64;

    // If there are not enough points to fit a line, return an empty list, and a line at y = 0
    if rows < points_for_line {
        return Vec::from([0.0, 0.0, 0.0]); // "Column form", slope of 0, and intercept of 0
    }

    // Randomly roll the points and take the first 'points_for_line' points.
    let roll_value: u64;
    if !use_same_random_vals {
        let uniform = Uniform::new(0.0, rows as f64).unwrap(); // Samples from the Uniform distribution
        roll_value = uniform.sample(rand) as u64; // This turns the sampling into a discrete uniform distribution
    } else {
        roll_value = supplied_roll_value as u64 % rows;
    }
    for _i in 0..roll_value {
        let temp = shape_points.pop().unwrap();
        shape_points.insert(0, temp);
    }

    // Make line_points and leftover_points
    let mut line_points = Vec::new();
    let mut leftover_points = Vec::new();
    for i in 0..shape_points.len() {
        if (i as u64) < points_for_line {
            line_points.push(shape_points[i]);
        } else {
            leftover_points.push(shape_points[i]);
        }
    }

    // Get approximate line for the points
    let mut results = least_squares_line_fit(crate::vec_tup_to_arr2_f32(line_points));

    // Return the line parameters and the leftover points.
    for point in leftover_points {
        results.push(point.0);
        results.push(point.1);
    }
    results
}


/// Rust equivalent of :func:`quad_fitting.get_random_point_and_fit_line`. Randomly sample ``points_for_line``
/// points from ``shape_points`` and then fit a line to these points using least squares. If ``use_same_random_vals``
/// is true, roll to the ``supplied_roll_value`` instead of randomly.
/// 
/// :param shape_points: All of the points in the contour to randomly sample from.
/// :type shape_points: &mut Vec<(f64, f64)>
/// :param points_for_line: The number of points to sample for a line.
/// :type points_for_line: u64
/// :param use_same_random_vals: If true, sample points starting at the ``supplied_roll_value`` instead of randomly.
/// :type use_same_random_vals: bool
/// :param supplied_roll_value: If ``use_same_random_vals`` is true, then sample points starting from this value.
/// :type supplied_roll_value: u32
/// :return: A vector containing the fitted line and the leftover points. The first three values are the equation form of the line (1 for ``row``, 0 for ``column``), the slope, and the intercept. The remaining values are the row and column values of leftover points from ``shape_points`` that were not used to fit this line.
/// :rtype: Vec<f64>
///
#[pyfunction]
pub fn get_random_point_and_fit_line_python_wrap(shape_points: Vec<(f64, f64)>, points_for_line: u64,
                                                 use_same_random_vals: bool, supplied_roll_value: u32) -> Vec<f64> {
    let mut rand = Rng::from_seed(123456);
    let mut shape_points_mut = shape_points.clone();
    get_random_point_and_fit_line(&mut shape_points_mut, points_for_line, use_same_random_vals, &mut rand, supplied_roll_value)

}

/// Computes the vector of two points.
///  
/// Arguments: p1 - First point to use
///            p2 - Second point to use
/// 
/// Returns: (f64, f64) - The vector of the two points
///
fn compute_vector(p1: (i64, i64), p2: (i64, i64)) -> (f64, f64) {
    let v = (p1.0 - p2.0, p1.1 - p2.1);
    let den = ((v.0.pow(2) + v.1.pow(2)) as f64).sqrt();
    ((v.0 as f64) / den, (v.1 as f64) / den)
}

/// Rust equivalent of :func:`quad_fitting.fit_quadrilaterals_to_contours`.
/// 
/// :param contour_list: A list of contours, with each contour having the form ``[layer_num, row1, col1, row2, col2, ...]``.
/// :type contour_list: Vec<Vec<i64>>
/// :param img_shape: The shape of the image, in the form ``(num_rows, num_cols)``.
/// :type img_shape: (i64, i64)
/// :param random_seed: If specified, the discrete uniform random number generator will be seeded with this value. If not specified, the random number generator will have a random seed, leading to new values each time it is run. 
/// :type random_seed: Option<u32>
/// :param points_per_line_ratio: The ratio of the number of points within a contour to use for each line fit.
/// :type points_per_line_ratio: f64
/// :param dist_for_inlier: The distance threshold for a point to be considered an inlier.
/// :type dist_for_inlier: f64
/// :param desired_inlier_percentage: The desired ratio of inliers within the total number of points.
/// :type desired_inlier_percentage: f64
/// :param required_inlier_percentage: The required ratio of inliers within the total number of points.
/// :type required_inlier_percentage: f64
/// :param parallel_threshold: The value used for determining whether or not two lines are parallel, when checking if the quad is a parallelogram.
/// :type parallel_threshold: f64
/// :param use_same_random_vals: Set to true to use repeatable random values, starting with the ``starting_roll_val``. Used to syncronize random values between Python & Rust versions of this function (for testing purposes).
/// :type use_same_random_vals: bool
/// :param starting_roll_val: The starting roll value for when ``use_same_random_vals`` is set to true.
/// :type starting_roll_val: u32
/// :return: A vector of quadrilaterals. Each quadrilateral contains four corners, each of which is a tuple of the form ``(row, column)``.
/// :rtype: Vec<Vec<(i64, i64)>>
/// 
#[pyfunction]
pub fn fit_quadrilaterals_to_contours(contour_list: Vec<Vec<i64>>, img_shape: (i64, i64), random_seed: Option<u32>, 
                                      points_per_line_ratio: f64, dist_for_inlier: f64, desired_inlier_percentage: f64, 
                                      required_inlier_percentage: f64, parallel_threshold: f64, use_same_random_vals: bool, starting_roll_val: u32) -> Vec<Vec<(i64, i64)>> {   

    // Now we can get into the meat of the algorithm by first setting up useful variables
    let mut quad_list = Vec::new();

    // Create the random number generator
    let mut rand = match random_seed {
        Some(seed) => Rng::from_seed(seed),
        None => Rng::new().unwrap()
    };
    
    // TODO: Calculate the number of iterations for RANSAC
    let n = 500;

    // Iterate through the number of contours
    for (_i, contour) in contour_list.into_iter().enumerate() {

        // Set up values for rolls if not random
        let mut curr_roll_val: u32 = starting_roll_val;

        // Extract the contour points from the contour
        let mut contour_points = Vec::new();
        for i in 1..contour.len() {
            if i % 2 == 0 {
                continue
            } else {
                contour_points.push((contour[i] as f64, contour[i+1] as f64));
            }
        }

        // Get the number of points in the contour
        let num_points = contour_points.len();

        // Calculate how many points are used for each line fit, ensuring 
        // that at least 2 points are used, as that is the minimum needed
        let points_for_line_float = Float::with_val(64, points_per_line_ratio * (num_points as f64));
        let mut points_for_line = points_for_line_float.to_u32_saturating_round(Round::Nearest).unwrap() as u64; // This round matches the python side
        if points_for_line < 2 { points_for_line = 2; }

        // Set up variables to track the best solution from RANSAC
        let mut num_iterations = 0;
        let mut best_inliers = Vec::from([Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
        let mut best_inlier_count = 0;
        let mut best_quad = Vec::new();
        let mut attempt_improvement = false;

        // Run RANSAC, fitting a quadrilateral using a random sample of points
        while num_iterations <= n { 
            let mut lines = Vec::new();
            let mut slopes = Vec::new();
            let mut temp_points = contour_points.to_owned();

            for i in 0..4 {
                // Normal case, randomly select points for the lines
                let results;
                if !attempt_improvement || best_inliers[i].is_empty() {                       
                    results = get_random_point_and_fit_line(&mut temp_points, points_for_line, use_same_random_vals, &mut rand, curr_roll_val);
                    
                    // Extract the leftover point and put them back into the temp_points
                    temp_points = Vec::new();
                    for j in (3..(results.len()-1)).step_by(2) {
                        temp_points.push((results[j], results[j+1]));
                    }
                    
                    curr_roll_val = curr_roll_val + 1;
                } else { // Special case, on the last run of this loop we want to try to refine the lines based on the best inlier sets found
                    results = least_squares_line_fit(crate::vec_tup_to_arr2_f32_ref(&best_inliers[i]));
                }
                // Round slopes and intercepts to 8 decimal places to ensure identical functionality to Python code
                let slope = (results[1] * 1e8).round() / 1e8;
                let intercept = (results[2] * 1e8).round() / 1e8;
                
                let line = Vec::from([results[0], slope, intercept]);
                lines.push(line);

                // Handle equations of the form row = slope column + intercept (x = m y + b)
                if results[0] == 1.0 { // Row form
                    // Set slope to a very large number when the line is vertical
                    if slope.abs() < 1e-6 {
                        slopes.push(1e6);
                    }
                    // Otherwise just invert the slope
                    else {
                        slopes.push(1.0 / slope);
                    }
                }
                // For column = slope row + intercept (y = m x + b) just append the slope as is
                else {
                    slopes.push(slope);
                }
            }

            // Sort based on slope to identify which lines are used to calculate the intersections
            for i in 0..3 {
                for j in (i+1)..4 {
                    if slopes[j] < slopes[i] {
                        // Swap the slopes
                        let temp = slopes[i];
                        slopes[i] = slopes[j];
                        slopes[j] = temp;

                        // Also swap the corresponding lines
                        lines.swap(i, j);
                    }
                }
            }

            // If any of the slopes are identical, make sure to sort by intercept as well
            // THIS seems unnecessary, but required to guaranteed identical results with Python code
            for i in 0..3 {
                for j in (i+1)..4 {
                    if slopes[j] == slopes[i] {
                        if lines[j][2] < lines[i][2] {
                            // Swap the slopes
                            let temp = slopes[i];
                            slopes[i] = slopes[j];
                            slopes[j] = temp;

                            // Also swap the corresponding lines
                            lines.swap(i, j);
                        }
                    }
                }
            }

            // Finally, if any of the slopes and intercepts are identical, sort by row/column form
            // THIS seems unnecessary, but required to guaranteed identical results with Python code
            for i in 0..3 {
                for j in (i+1)..4 {
                    if slopes[j] == slopes[i] && lines[j][2] == lines[i][2] {
                        if lines[j][0] < lines[i][0] {
                            // Swap the slopes
                            let temp = slopes[i];
                            slopes[i] = slopes[j];
                            slopes[j] = temp;

                            // Also swap the corresponding lines
                            lines.swap(i, j);
                        }
                    }
                }
            }

            // Solve for the intersections, getting the four corners of the quadrilateral
            let mut intersections = Vec::new();

            // The order is such that we should get the four points in either clockwise or counterclockwise order around the quadrilateral
            let intersect = match get_intersection_of_lines((lines[0][1], lines[0][2]), (lines[2][1], lines[2][2]), 
            lines[0][0] == 1.0, lines[2][0] == 1.0) { 
                Result::Ok(val) => val, 
                Result::Err(_) => {
                    num_iterations += 1;
                    continue;
                }
            };
            intersections.push(intersect);

            let intersect = match get_intersection_of_lines((lines[0][1], lines[0][2]), (lines[3][1], lines[3][2]), 
            lines[0][0] == 1.0, lines[3][0] == 1.0) { 
                Result::Ok(val) => val, 
                Result::Err(_) => {
                    num_iterations += 1;
                    continue;
                }
            };
            intersections.push(intersect);

            let intersect = match get_intersection_of_lines((lines[1][1], lines[1][2]), (lines[3][1], lines[3][2]), 
            lines[1][0] == 1.0, lines[3][0] == 1.0) { 
                Result::Ok(val) => val, 
                Result::Err(_) => {
                    num_iterations += 1;
                    continue;
                }
            };
            intersections.push(intersect);

            let intersect = match get_intersection_of_lines((lines[1][1], lines[1][2]), (lines[2][1], lines[2][2]), 
            lines[1][0] == 1.0, lines[2][0] == 1.0) { 
                Result::Ok(val) => val, 
                Result::Err(_) => {
                    num_iterations += 1;
                    continue;
                }
            };
            intersections.push(intersect);

            // Move on to the next iteration if any of the intersection points lie outside of the image bounds
            let mut skip = false;
            for (_i, intersection) in intersections.iter().enumerate() {
                if intersection.0 < 0 || intersection.0 >= img_shape.0 || intersection.1 < 0 || intersection.1 >= img_shape.1 {
                    if !skip {
                        num_iterations += 1;
                        skip = true;
                    }
                }
            }
            if skip { 
                num_iterations += 1;
                continue; 
            }

            // If any of the intersections are the same point, this isn't a quadrilateral.
            // Go on to the next iteration
            let mut intersections_match = false;
            for i in 0..4 {
                for j in (i+1)..4 {
                    if intersections[i].0 == intersections[j].0 && intersections[i].1 == intersections[j].1 {
                        intersections_match = true;
                    }
                }
            } if intersections_match {
                num_iterations += 1;
                continue;
            }

            // Calculate the distance between each point and each of the four sides of the quadrilateral
            let mut distances = ndarray::Array2::zeros((num_points, 4));
            for (i, point) in contour_points.iter().enumerate() {
                distances[[i, 0]] = distance_between_point_and_line_segment_i64(*point, intersections[0], intersections[1]);
                distances[[i, 1]] = distance_between_point_and_line_segment_i64(*point, intersections[1], intersections[2]);
                distances[[i, 2]] = distance_between_point_and_line_segment_i64(*point, intersections[2], intersections[3]);
                distances[[i, 3]] = distance_between_point_and_line_segment_i64(*point, intersections[3], intersections[0]);
            }

            // Determine which line segment is the closest for each point, and determine the inliers
            let mut inliers: Vec<Vec<(f64, f64)>> = Vec::from([Vec::new(), Vec::new(), Vec::new(), Vec::new()]);
            let mut num_inliers = 0;
        
            // Determine the number of inliers, and split them up according to which line segment they belong to
            for i in 0..distances.shape()[0] {
                // Find minimum distance
                let mut min = distances[[i, 0]];
                let mut min_index = 0;
                for j in 1..4 {
                    if distances[[i, j]] < min {
                        min = distances[[i, j]];
                        min_index = j;
                    }
                }

                if min <= dist_for_inlier {
                    inliers[min_index].push(contour_points[i]);
                    num_inliers += 1;
                }
            }
            
            // Only accept solutions that contain enough points for four unique lines (a valid quadrilateral)
            for (_i, inlier) in inliers.iter().enumerate() {
                if inlier.len() < 2 {
                    num_inliers = 0;
                }
            }
            
            //Update tracking of best solution if this is a better solution
            if num_inliers > best_inlier_count {
                best_inlier_count = num_inliers;
                best_quad = intersections.clone();
                best_inliers = inliers.clone();

                // Continue attempting to improve the quadrilateral until we no longer see an increase in the number of inliers
                if attempt_improvement {
                    num_iterations = n - 1;
                }
            }
            
            // Increment the number of iterations
            num_iterations += 1;

            // Check for early termination criteria
            if ((num_inliers as f64) / (num_points as f64)) >= desired_inlier_percentage && attempt_improvement == false {
                attempt_improvement = true;
                num_iterations = n;

            } // If we reach the maximum number of iterations, begin attempting improvement on the best solution found thus far
            else if num_iterations == n {
                if !attempt_improvement {
                    attempt_improvement = true;
                }
            }
        }
        
        // Check if the best fit we found meets the criteria to be a good enough fit to call it a probable quadrilateral
        if (best_inlier_count as f64 / num_points as f64) >= required_inlier_percentage {
            // Check if the quadrilateral is a parallelogram by computing the dot product of the two sets
            // of unit vectors that should be parallel
            let v1 = compute_vector(best_quad[0], best_quad[1]);
            let v2 = compute_vector(best_quad[3], best_quad[2]);
            let v3 = compute_vector(best_quad[1], best_quad[2]);
            let v4 = compute_vector(best_quad[0], best_quad[3]);
            let v1v2 = (v1.0*v2.0 + v1.1*v2.1).abs();
            let v3v4 = (v3.0*v4.0 + v3.1*v4.1).abs();

            // Only accept the quadrilateral if the dot product is above the specified threshold that we consider to be a parallelogram
            if v1v2 >= parallel_threshold && v3v4 >= parallel_threshold {
                // Check if the points are in clockwise or counterclockwise order around the quadrilateral (sum over edges (r2-r1)(c2+c1))
                let mut temp_quad = best_quad.clone();
                let temp_value = temp_quad.remove(0);
                temp_quad.push(temp_value);
                let mut other_temp_quad = Vec::new();
                for i in 0..4 {
                    other_temp_quad.push((temp_quad[i].0 - best_quad[i].0, temp_quad[i].1 + best_quad[i].1));
                }
                let mut sum = 0;
                for i in 0..4 {
                    sum += other_temp_quad[i].0 * other_temp_quad[i].1;
                }

                // If the sum is negative, the points are in counter-clockwise order, and we need to reorder them to be clockwise
                if sum < 0 {
                    best_quad.swap(1, 3);
                }
                // Append the quadrilateral to the list of quadrilaterals
                quad_list.push(best_quad);
            }
        }
    }
    // Return the list of any quadrilaterals found
    quad_list
}