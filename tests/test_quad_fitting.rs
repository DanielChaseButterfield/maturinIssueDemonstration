use actag::quad_fitting::*;
use float_cmp::approx_eq;

// Test the get_intersection_of_lines function.
#[test]
fn test_get_intersection_of_lines() {
    // ========== Test 1 ==========
    let result = get_intersection_of_lines((19.3, 32.0), (32.0, 14.0), true, false);
    assert_eq!(result.unwrap(), (0, -2));

    // ========== Test 2 ==========
    let result = get_intersection_of_lines((-25.0, 32.0), (32.0, 14.0), true, false);
    assert_eq!(result.unwrap(), (0, 1));

    // ========== Test 3 ==========
    let result = get_intersection_of_lines((-25.0, 32.0), (32.0, 14.0), false, true);
    assert_eq!(result.unwrap(), (1, 0));

    // ========== Test 4 ==========
    let result = get_intersection_of_lines((-25.0, 32.0), (0.9, 32.9), false, true);
    assert_eq!(result.unwrap(), (3, -34));
}

// Test the distance_between_point_and_line_segment function.
#[test]
fn test_distance_between_point_and_line_segment() {
    // ========== Test 1 ==========
    let result = distance_between_point_and_line_segment((0.0, 0.0), (0.0, 0.0), (1.0, 1.0));
    assert_eq!(result, 0.0);

    // ========== Test 2 ==========
    let result = distance_between_point_and_line_segment((10.0, 0.0), (-5.0, 0.0), (1.0, 200.0));
    assert_eq!(result, 14.993254552835502);

    // ========== Test 3 ==========
    let result = distance_between_point_and_line_segment((10.0, 10.0), (-5.0, -4.4), (23.0, 200.0));
    assert_eq!(result, 12.906859904821516);
}

// Test the distance_between_point_and_line function.
#[test]
fn test_distance_between_point_and_line() {
    // ========== Test 1 ==========
    let result = distance_between_point_and_line((15.6, -2.0), 24.5, -1098.9, true);
    assert_eq!(result, 47.45028692989885);

    // ========== Test 2 ==========
    let result = distance_between_point_and_line((15.6, 14.0), 24.5, -198.9, true);
    assert_eq!(result, 5.2405344825887425);

    // ========== Test 3 ==========
    let result = distance_between_point_and_line((15.6, 14.0), 24.5, -198.9, false);
    assert_eq!(result, 6.904455158772561);
}

// Test the least_squares_line_fit function.
#[test]
fn test_least_squares_line_fit() {
    // ========== Test 1 ==========
    let results = least_squares_line_fit(ndarray::Array2::from(vec![[72.0, 170.0],
        [ 71.0, 171.0], [ 72.0, 172.0], [ 72.0, 171.0], [ 72.0, 170.0], [ 76.0, 167.0], [ 76.0, 168.0], 
        [ 77.0, 168.0], [ 78.0, 169.0], [ 78.0, 170.0], [ 78.0, 171.0], [ 79.0, 170.0], [ 80.0, 169.0],
        [ 82.0, 167.0], [ 83.0, 167.0], [ 84.0, 168.0], [ 84.0, 169.0], [ 85.0, 170.0], [ 91.0, 169.0],
        [ 91.0, 168.0], [ 92.0, 167.0], [ 94.0, 167.0], [ 94.0, 166.0], [ 95.0, 165.0]]));
    assert!( approx_eq!(f64, results[0], 0.0, ulps = 10));
    assert!( approx_eq!(f64, results[1], -0.16276595744680838, epsilon = 1e-13));
    assert!( approx_eq!(f64, results[2], 181.9737588652481, epsilon = 1e-10));
}