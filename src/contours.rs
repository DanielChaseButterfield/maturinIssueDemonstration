use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use itertools_num::linspace;

/// Rust equivalent of :func:`contour_identification.get_contours`.
///
/// :param img: the image of which the contours will be found.
/// :type img: Vec<Vec<i32>>
/// :param min_range: Minimum range of the sonar in meters
/// :type min_range: f64
/// :param max_range: Maximum range of the sonar in meters
/// :type max_range: f64
/// :param horizontal_aperture: Horizontal aperture of the sonar in radians
/// :type horizontal_aperture: f64
/// :param tag_size: Side length of the white square on the inside of the tag in meters
/// :type tag_size: f64
/// :param tag_size_tolerance: The tolerance for rejecting contours that are too long in the range or azimuth axes
/// :type tag_size_tolerance: f64
/// :param tag_area_tolerance: The tolerance for rejecting contours that are too large or too small in pixel area
/// :type tag_area_tolerance: f64
/// :param min_tag_area_ratio: The ratio of the smallest size of a tag when rotated to the full size of the tag
/// :type min_tag_area_ratio: f64
/// :param reject_black_shape: Flag to reject black contours (i.e. outer area is white, inner is black)
/// :type reject_black_shape: bool
/// :param reject_white_shapes: Flag to reject white contours (i.e. outer area is black, inner is white)
/// :type reject_white_shapes: bool
/// :param reject_by_tag_size: Flag to reject contours that are too long in range or azimuth axes
/// :type reject_by_tag_size: bool
/// :param reject_by_area: Flag to reject contours that are too large or too small in pixel area
/// :type reject_by_area: bool
/// :return: The vector containing all of the image contours.
/// :rtype: Vec<Vec<i32>>
/// 
#[pyfunction]
pub fn get_contours(img: Vec<Vec<i32>>, min_range: f64, max_range: f64, horizontal_aperture: f64,
                        tag_size: f64, tag_size_tolerance: f64, tag_area_tolerance: f64, min_tag_area_ratio: f64,
                        reject_black_shape: bool, reject_white_shapes: bool,
                        reject_by_tag_size: bool, reject_by_area: bool) -> Vec<Vec<usize>> {
    // Convert Vec<Vec<i32>> to Array2<i32>
    let img_arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = crate::vec2_to_arr2_i32(img);

    // Get the rows and columns
    let rows: usize = img_arr.shape()[0];
    let cols: usize = img_arr.shape()[1];

    // Calculate values for rejecting contours
    let range_pixel_resolution = (max_range - min_range) / (rows as f64); // Meters per pixel
    let tag_diagonal_length = (2.0 * tag_size.powi(2)).sqrt(); // Meters
    let max_tag_range_pixels = tag_diagonal_length / range_pixel_resolution;
    let ranges = linspace::<f64>(max_range, min_range, rows); // Max to min, to line up with image
    let mut azimuth_pixel_resolution = ndarray::Array::zeros(rows);
    for (i, val) in ranges.into_iter().enumerate() {
        azimuth_pixel_resolution[[i]] = val * horizontal_aperture / (cols as f64); // Meters per pixel
    }

    // Make a working image, which is a copy of the input image
    let mut working_img: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = ndarray::Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            working_img[[i, j]] = img_arr[[i, j]];
        }
    }

    // Contours are defined by: [layer_num, r1, c1, ... rn, cn]
    let mut contour_list = Vec::new();

    // Initialize other useful variables
    let mut layer_num = 0;

    // Define a flipped neighborhood
    let neigh_flip: [(isize, isize); 8] = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)];
    let neigh_flip_index = [7, 6, 5, 4, 3, 2, 1, 0];

    // Main algorithm loop, exit when the working image is all zeros
    let mut all_zeros: bool = false;
    while !all_zeros {
        // Increase the layer number
        layer_num += 1;


        // Skip any layers that the user has specified to be skipped
        if (layer_num % 2 == 0 && reject_black_shape) || (layer_num % 2 == 1 && reject_white_shapes) {

            // Flood fill the image
            working_img = crate::vec2_to_arr2_i32(crate::flood_fill(crate::arr2_to_vec2_i32(working_img), 0, 0, -1));

            // Invert the image
            for i in 0..working_img.shape()[0] {
                for j in 0..working_img.shape()[1] {
                    if working_img[[i, j]] == -1 {
                        working_img[[i, j]] = 1;
                    }
                    working_img[[i, j]] += 1;
                }
            }

            // Proceed to the next layer
            continue;
        }

        // Create an array to identify which pixels are searchable. Pixels part of,
        // or inside of, a contour are marked, and therefore are unsearchable in the current layer
        let mut searchable: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = ndarray::Array2::zeros((rows, cols));

        // Iterate through the rows of the image, looking for next contours
        for i in 0..(working_img.shape()[0]) {
            // Iterate through the pixels/columns of the row, looking for next contours
            for j in 0..(working_img.shape()[1]) {
                
                // Begin border following when we find a 1, and it is searchable
                if working_img[[i, j]] == 1 && searchable[[i, j]] == 0 {
                    // Initialize variables for the new contour
                    let mut keep_following = true;
                    let mut r_contour = i;
                    let mut c_contour = j;
                    let mut came_from = 4;

                    // Start the list of pixels in the contour
                    let mut chain = Vec::new();
                    chain.push(i);
                    chain.push(j);

                    // Create an array with the indices of each pixel
                    let mut idxs = Vec::new();
                    idxs.push((i, j));

                    // Starting at the topmost and leftmost pixel of the contour,
                    // we then follow around it working in a clockwise direction
                    while keep_following {
                        // Get order of neighboring pixels to check, working clockwise around the pixel we just came from
                        let mut ray_path = neigh_flip.to_vec();
                        let mut ray_path_index = neigh_flip_index.to_vec();
                        for _k in 0..came_from { // Roll "came_from" times
                            let temp = ray_path.pop();
                            ray_path.insert(0, temp.unwrap());
                            let temp_index = ray_path_index.pop();
                            ray_path_index.insert(0, temp_index.unwrap());
                        }

                        // Search each of the neighboring pixels around the current pixel
                        for k in 0..ray_path.len() {
                            // Set loop variables
                            let dir = ray_path_index[k];
                            let add_r  = ray_path[k].0;
                            let add_c = ray_path[k].1;

                            // If we've come back to where we started, we've traced
                            // the full contour
                            if r_contour == i && c_contour == j {
                                if dir == 4 {
                                    keep_following = false;
                                    
                                    // If the contour contains more than a single pixel, it will have a duplicate in the first
                                    // and last positions, so we can remove the last pixel, as it is redundant
                                    if chain.len() > 2 {
                                        chain.pop();
                                        chain.pop();
                                    }
                                    break;
                                }
                            }

                            // Skip any instances where we would be moving outside of the image
                            if ((r_contour as isize) + add_r < 0) || ((r_contour as isize) + add_r >= (rows as isize))
                                || ((c_contour as isize) + add_c < 0) || ((c_contour as isize) + add_c >= (cols as isize)) {
                                continue;
                            }
                            // Otherwise, if we encounter a 1, we found the next pixel in the contour
                            else if working_img[[((r_contour as isize) + add_r) as usize,
                                                 ((c_contour as isize) + add_c) as usize]] == 1 {
                                // Update indices and searchable array
                                r_contour = ((r_contour as isize) + add_r) as usize;
                                c_contour = ((c_contour as isize) + add_c) as usize;
                                searchable[[r_contour, c_contour]] = 1;

                                // Add onto chain path, and update temporary list index
                                chain.push(r_contour);
                                chain.push(c_contour);
                                idxs.push((r_contour, c_contour));

                                // Update came_from to indicate the ray value that points to the pixel we came from
                                came_from = (dir + 4) % 8;
                                break;
                            }
                        }
                    }

                    // After tracing the contour, we need to fill the area inside the contour in the searchable array
                    // to avoid retracing it, and waiting to trace any inner level contours until we reach the next layer
                    // To do this, we create a temporary array that is just large enough to hold the shape
                    let mut r_min = working_img.shape()[0];
                    let mut c_min = working_img.shape()[1];
                    let mut r_max = 0;
                    let mut c_max = 0;
                    let mut pixels_in_contour: i32 = (chain.len() as i32) / 2; // Start with the outer edge pixels
                    
                    if idxs.len() > 1 {
                        // Get the min and max values for the rows and columns
                        for k in 0..idxs.len() {
                            if idxs[k].0 < r_min {
                                r_min = idxs[k].0;
                            } else if idxs[k].0 > r_max {
                                r_max = idxs[k].0;
                            }
                            if idxs[k].1 < c_min {
                                c_min = idxs[k].1;
                            } else if idxs[k].1 > c_max {
                                c_max = idxs[k].1;
                            }
                        }

                        // Get just the contour area
                        let temp_array = searchable.slice_mut(s![r_min..(r_max+1), c_min..(c_max+1)]);
                        
                        // Pad around the contour area to ensure the flood fill will work as intended
                        let mut temp_area_padded: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>>=
                                                ndarray::Array2::zeros((temp_array.shape()[0]+2, temp_array.shape()[1]+2));
                        for k in 1..=(temp_array.shape()[0]) {
                            for n in 1..=(temp_array.shape()[1]) {
                                temp_area_padded[[k, n]] = temp_array[[k-1, n-1]];
                            }
                        }

                        // We will now fill the area inside the contour using the approach described at
                        // https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
                        // which is: flood fill the outer area of 0's with 1's, invert the resulting image,
                        // and then bitwise or the original image with the inverted image
                        let mut inverted_array = crate::vec2_to_arr2_i32(
                            crate::flood_fill(crate::arr2_to_vec2_i32_ref(&temp_area_padded), 0, 0, 1));
                        for k in 0..inverted_array.shape()[0] {
                            for n in 0..inverted_array.shape()[1] {
                                if inverted_array[[k, n]] == 1 {
                                    inverted_array[[k, n]] = -1;
                                }
                                inverted_array[[k, n]] += 1;
                            }
                        }

                        // Calculate total area of contour after filling the area inside
                        pixels_in_contour += inverted_array.iter().sum::<i32>();

                        // Combine temp_area_padded and inverted_array
                        for k in 0..inverted_array.shape()[0] {
                            for n in 0..inverted_array.shape()[1] {
                                if temp_area_padded[[k, n]] == 0 && inverted_array[[k, n]] != 0 {
                                    temp_area_padded[[k, n]] = inverted_array[[k, n]];
                                }
                            }
                        }
            
                        // Replace the searchable array with the filled array
                        for k in 1..(temp_area_padded.shape()[0] - 1) {
                            for n in 1..(temp_area_padded.shape()[1] - 1) {
                                searchable[[r_min+k-1, c_min+n-1]] = temp_area_padded[[k, n]];
                            }
                        }
                    } else { // If the contour is just a single pixel
                        searchable[[r_contour, c_contour]] = 1;
                        r_min = 1;
                        c_min = 1;
                        r_max = 1;
                        c_max = 1;
                        pixels_in_contour = 1;
                    }

                    // Calculate useful values for contour rejection
                    let range_pix = (r_max - r_min) as f64;
                    let azimuth_pix = (c_max - c_min) as f64;
                    let r_centroid = r_min + ((r_max - r_min) as f64 / 2.0).floor() as usize;
                    let max_tag_azimuth_pixels = tag_diagonal_length / azimuth_pixel_resolution[[r_centroid]];

                    // Filter out any contours that shouldn't be possible, given range information and known tag size
                    if reject_by_tag_size || reject_by_area {
                        if reject_by_tag_size {
                             // Reject contours that occupy too much of the range axis
                            if range_pix > (max_tag_range_pixels * (1.0 + tag_size_tolerance)) { continue; }

                            // Reject contours that occupy too much of the azimuth axis
                            if azimuth_pix > (max_tag_azimuth_pixels * (1.0 + tag_size_tolerance)) { continue; }
                        }
                        
                        if reject_by_area {
                            // Calculate number of pixels that a tag could occupy, given the location of the centroid
                            let max_num_pixels = max_tag_range_pixels * max_tag_azimuth_pixels;
                            let min_num_pixels = max_num_pixels* min_tag_area_ratio;

                            // Reject contours whose area is too small or too large
                            if ((pixels_in_contour as f64) > (max_num_pixels * (1.0 + tag_area_tolerance))) ||
                               ((pixels_in_contour as f64) < (min_num_pixels * (1.0 - tag_area_tolerance))) {
                                continue;
                            }
                        }
                    }

                    // Create the vector for the new contour
                    let mut new_contour: Vec<usize> = Vec::<usize>::new();
                    new_contour.push(layer_num.try_into().unwrap());
                    new_contour.append(&mut chain);

                    // Add the contour to the list
                    contour_list.push(new_contour);
                }
                // Set all values outside of a known shape to -1 in preparation for the next layer
                else if searchable[[i, j]] == 0 {
                    working_img[[i,j]] = -1;
                }
            }
        }

        // Invert the image to go to the next layer
        for i in 0..working_img.shape()[0] {
            for j in 0..working_img.shape()[1] {
                if working_img[[i, j]] == 1{
                    working_img[[i, j]] = -1;
                }
                working_img[[i, j]] += 1;
            }
        }

        // Check if all of the first layer values are 0
        all_zeros = true;
        for i in 0..rows {
            for j in 0..cols {
                if working_img[[i, j]] != 0 {
                    all_zeros = false;
                    break;
                }
            }
        }
    }
    // Return the list of all contours that were found
    contour_list
}

/// Visualize a contour list onto an binary image with the specified dimensions.
///
/// :param contour_list: A list of contours to visualize.
/// :type contour_list: Vec<Vec<i32>>
/// :param img_shape: The shape of the image to plot the contours onto.
/// :type img_shape: (usize, usize)
/// :return: An image with the contours drawn.
/// :rtype: Vec<Vec<i32>>
/// 
#[pyfunction]
pub fn plot_contours(contour_list: Vec<Vec<i32>>, img_shape: (usize, usize)) -> Vec<Vec<i32>> {
    // Create an empty image of the specified size
    let mut output_img: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = ndarray::Array2::ones((img_shape.0, img_shape.1)) * 0;

    // Iterate through the contours
    for (_k, contour) in contour_list.into_iter().enumerate() {
        // For each point in the contour
        for l in 0..((contour.len()-3) / 2) {
            // Get the next point position
            let i = contour[l * 2 + 3] as usize;
            let j = contour[l * 2 + 4] as usize;

            // Plot the contour at the current location
            if contour[0] % 2 == 0 {
                output_img[[i, j]] = 0;
            } else {
                output_img[[i, j]] = 255;
            }
        }
    }
    // Return the image with the contours drawn on it
    crate::arr2_to_vec2_i32(output_img)
}