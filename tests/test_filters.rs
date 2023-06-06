use actag::filters::*;

#[test]
// This function will test the adaptive_threshold() function and
// the adaptive_threadhold_multithread() function
fn test_adaptive_thresholds() {
    // ========== Test 1 ==========
    // Create a test vector
    let mut vec_test = Vec::new();
    let row1: Vec<i32> = Vec::from([  1,   2, 255,  40, 255]);
    let row2: Vec<i32> = Vec::from([  6,   7,  20, 255,  10]);
    let row3: Vec<i32> = Vec::from([255, 100,  13, 255,  15]);
    vec_test.push(row1);
    vec_test.push(row2);
    vec_test.push(row3);
    let vec_test_2 = vec_test.clone();

    // Run the adaptive threshold algorithm
    let result = adaptive_threshold(vec_test, 8, 1.0);
    let result_mt = adaptive_threshold_multithread(vec_test_2, 8, 1.0, None);

    // Create vector with desired results
    let mut vec_desired = Vec::new();
    let row1: Vec<i32> = Vec::from([0, 0, 1, 0, 1]);
	let row2: Vec<i32> = Vec::from([0, 0, 0, 1, 0]);
	let row3: Vec<i32> = Vec::from([1, 0, 0, 1, 0]);
    vec_desired.push(row1);
    vec_desired.push(row2);
    vec_desired.push(row3);

    // Make sure the output is correct.
    assert_eq!(result, vec_desired);
    assert_eq!(result_mt, vec_desired);

    // ========== Test 2 ==========
    // Create a test vector
    let mut vec_test = Vec::new();
    let row1: Vec<i32> = Vec::from([155, 89, 159, 92, 134, 154, 149, 161, 177, 85]);
	let row2: Vec<i32> = Vec::from([165, 70, 113, 73, 166, 67, 96, 161, 113, 152]);
	let row3: Vec<i32> = Vec::from([77, 176, 112, 80, 155, 85, 130, 106, 187, 78]);
	let row4: Vec<i32> = Vec::from([63, 74, 104, 106, 70, 93, 99, 66, 178, 139]);
	let row5: Vec<i32> = Vec::from([115, 130, 183, 177, 135, 79, 110, 147, 148, 127]);
	let row6: Vec<i32> = Vec::from([175, 65, 134, 92, 125, 158, 128, 98, 139, 75]);
	let row7: Vec<i32> = Vec::from([95, 136, 116, 136, 124, 128, 140, 91, 80, 123]);
	let row8: Vec<i32> = Vec::from([187, 144, 108, 72, 175, 183, 74, 183, 181, 147]);
	let row9: Vec<i32> = Vec::from([78, 80, 108, 100, 66, 123, 190, 176, 101, 99]);
	let row10: Vec<i32> = Vec::from([84, 122, 169, 70, 189, 174, 139, 151, 184, 101]);
	vec_test.push(row1);
    vec_test.push(row2);
    vec_test.push(row3);
    vec_test.push(row4);
    vec_test.push(row5);
    vec_test.push(row6);
    vec_test.push(row7);
    vec_test.push(row8);
    vec_test.push(row9);
    vec_test.push(row10);
    let vec_test_2 = vec_test.clone();

    // Run the filter
    let result = adaptive_threshold(vec_test, 8, 1.0);
    let result_mt = adaptive_threshold_multithread(vec_test_2, 8, 1.0, None);

    // Create vector with desired results - These values were calculated by MATLAB.
    let mut vec_desired = Vec::new();
    let row1: Vec<i32> = Vec::from([1, 0, 1, 0, 1, 1, 1, 1, 1, 0]);
	let row2: Vec<i32> = Vec::from([1, 0, 0, 0, 1, 0, 0, 1, 0, 1]);
	let row3: Vec<i32> = Vec::from([0, 1, 0, 0, 1, 0, 1, 0, 1, 0]);
	let row4: Vec<i32> = Vec::from([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]);
	let row5: Vec<i32> = Vec::from([0, 1, 1, 1, 1, 0, 0, 1, 1, 1]);
	let row6: Vec<i32> = Vec::from([1, 0, 1, 0, 0, 1, 1, 0, 1, 0]);
	let row7: Vec<i32> = Vec::from([0, 1, 0, 1, 0, 1, 1, 0, 0, 0]);
	let row8: Vec<i32> = Vec::from([1, 1, 0, 0, 1, 1, 0, 1, 1, 1]);
	let row9: Vec<i32> = Vec::from([0, 0, 0, 0, 0, 0, 1, 1, 0, 0]);
	let row10: Vec<i32> = Vec::from([0, 0, 1, 0, 1, 1, 1, 1, 1, 0]);
    vec_desired.push(row1);
    vec_desired.push(row2);
    vec_desired.push(row3);
    vec_desired.push(row4);
    vec_desired.push(row5);
    vec_desired.push(row6);
    vec_desired.push(row7);
    vec_desired.push(row8);
    vec_desired.push(row9);
    vec_desired.push(row10);

    // Make sure the output is correct.
    assert_eq!(result, vec_desired);
    assert_eq!(result_mt, vec_desired);
}

#[test]
// This function will test the bilateral_filter() function and
// the bilateral_filter_multithread() function.
fn test_bilateral_filters() {

    // ========== Test 1 ==========
    // Create a test vector
    let mut vec_test = Vec::new();
    let row1: Vec<i32> = Vec::from([  1,   2, 255,  40, 255]);
    let row2: Vec<i32> = Vec::from([  6,   7,  20, 255,  10]);
    let row3: Vec<i32> = Vec::from([255, 100,  13, 255,  15]);
    vec_test.push(row1);
    vec_test.push(row2);
    vec_test.push(row3);
    let vec_test_2 = vec_test.clone();

    // Run the filter
    let result = bilateral_filter(vec_test, 20, 50, 1);
    let result_mt = bilateral_filter_multithread(vec_test_2, 20, 50, 1, None);

    // Create vector with desired results - These values were calculated by MATLAB.
    let mut vec_desired = Vec::new();
    let row1: Vec<i32> = Vec::from([4, 6, 255, 29, 255]);
	let row2: Vec<i32> = Vec::from([4, 8, 16, 255, 16]);
	let row3: Vec<i32> = Vec::from([255, 100, 13, 255, 13]);
    vec_desired.push(row1);
    vec_desired.push(row2);
    vec_desired.push(row3);

    // Make sure the output is correct.
    assert_eq!(result, vec_desired);
    assert_eq!(result_mt, vec_desired);

    // ========== Test 2 ==========
    // Create a test vector
    let mut vec_test = Vec::new();
    let row1: Vec<i32> = Vec::from([202, 62, 197, 68, 151, 190, 181, 211, 253, 48]);
	let row2: Vec<i32> = Vec::from([211, 18, 104, 25, 202, 9, 73, 198, 102, 176]);
	let row3: Vec<i32> = Vec::from([41, 229, 108, 45, 183, 54, 139, 95, 255, 37]);
	let row4: Vec<i32> = Vec::from([4, 32, 89, 95, 19, 70, 81, 2, 223, 148]);
	let row5: Vec<i32> = Vec::from([116, 138, 228, 221, 144, 40, 102, 166, 165, 128]);
	let row6: Vec<i32> = Vec::from([227, 10, 139, 68, 128, 184, 132, 75, 148, 28]);
	let row7: Vec<i32> = Vec::from([78, 148, 111, 146, 126, 133, 153, 61, 32, 121]);
	let row8: Vec<i32> = Vec::from([253, 167, 102, 27, 221, 237, 30, 242, 236, 165]);
	let row9: Vec<i32> = Vec::from([39, 45, 100, 86, 6, 129, 251, 225, 79, 79]);
	let row10: Vec<i32> = Vec::from([52, 127, 204, 15, 251, 219, 155, 177, 238, 80]);
	vec_test.push(row1);
    vec_test.push(row2);
    vec_test.push(row3);
    vec_test.push(row4);
    vec_test.push(row5);
    vec_test.push(row6);
    vec_test.push(row7);
    vec_test.push(row8);
    vec_test.push(row9);
    vec_test.push(row10);
    let vec_test_2 = vec_test.clone();

    // Run the filter
    let result = bilateral_filter(vec_test, 80, 20, 6);
    let result_mt = bilateral_filter_multithread(vec_test_2, 80, 20, 6, None);

    // Create vector with desired results - These values were calculated by MATLAB.
    let mut vec_desired = Vec::new();
    let row1: Vec<i32> = Vec::from([155, 89, 159, 92, 134, 154, 149, 161, 177, 85]);
	let row2: Vec<i32> = Vec::from([165, 70, 113, 73, 166, 67, 96, 161, 113, 152]);
	let row3: Vec<i32> = Vec::from([77, 176, 112, 80, 155, 85, 130, 106, 187, 78]);
	let row4: Vec<i32> = Vec::from([63, 74, 104, 106, 70, 93, 99, 66, 178, 139]);
	let row5: Vec<i32> = Vec::from([115, 130, 183, 177, 135, 79, 110, 147, 148, 127]);
	let row6: Vec<i32> = Vec::from([175, 65, 134, 92, 125, 158, 128, 98, 139, 75]);
	let row7: Vec<i32> = Vec::from([95, 136, 116, 136, 124, 128, 140, 91, 80, 123]);
	let row8: Vec<i32> = Vec::from([187, 144, 108, 72, 175, 183, 74, 183, 181, 147]);
	let row9: Vec<i32> = Vec::from([78, 80, 108, 100, 66, 123, 190, 176, 101, 99]);
	let row10: Vec<i32> = Vec::from([84, 122, 169, 70, 189, 174, 139, 151, 184, 101]);
    vec_desired.push(row1);
    vec_desired.push(row2);
    vec_desired.push(row3);
    vec_desired.push(row4);
    vec_desired.push(row5);
    vec_desired.push(row6);
    vec_desired.push(row7);
    vec_desired.push(row8);
    vec_desired.push(row9);
    vec_desired.push(row10);

    // Make sure the output is correct.
    assert_eq!(result, vec_desired);
    assert_eq!(result_mt, vec_desired);
}