use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray::s;
use std::cmp;
use std::f64::consts::PI;
use num::traits::checked_pow;
use libm::exp;

// This helper function pads a vector image according to the "edge" scheme
fn pad_edge_vector(data: &Vec<Vec<i32>>, radius: usize) -> Vec<Vec<i32>> {
    let mut padded_data = data.clone();

    let first_val = padded_data[0].clone();
    let last_val = padded_data[padded_data.len() - 1].clone();
    for _i in 0..radius {
        padded_data.insert(0, first_val.clone());
        padded_data.push(last_val.clone());
    }

    for i in 0..padded_data.len() {
        let first_val = padded_data[i][0];
        let last_val = padded_data[i][padded_data[i].len() - 1];
        for _j in 0..radius {
            padded_data[i].insert(0, first_val);
            padded_data[i].push(last_val);
        }
    }
    padded_data
}

// This helper function does the opposite of pad_edge_vector()
fn unpad_edge_vector(filtered_vec: &Vec<Vec<i32>>, radius: usize) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    for i in radius..(filtered_vec.len()-radius) {
        let mut result_row = Vec::new();
        for j in radius..(filtered_vec[0].len()-radius) {
            result_row.push(filtered_vec[i][j])
        }
        result.push(result_row);
    }
    result
}

/// Rust equivalent of :func:`median_filter.median_filter`.
///
/// :param data: 2D image.
/// :type img: Vec<Vec<i32>>
/// :param radius: Radius of the kernel.
/// :type radius: usize
/// :return: Filtered image.
/// :rtype: Vec<Vec<i32>>
///
#[pyfunction]
pub fn median_filter(data: Vec<Vec<i32>>, radius: usize) -> Vec<Vec<i32>> {
    // Pad the input data with the shape of radius
    let padded_data = pad_edge_vector(&data, radius);

    // Initialize input array and output array for the median filter
    let arr = crate::vec2_to_arr2_i32(padded_data);
    let mut filtered = ndarray::Array2::zeros((arr.shape()[0], arr.shape()[1]));

    // Iterate through each pixel
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            // Get slice indices, ensuring they don't go out of bounds
            let r1 = cmp::max(0, i as i32 - radius as i32) as usize;
            let r2 = cmp::min((arr.shape()[0] as i32) - 1, i as i32 + radius as i32) as usize;
            let c1 = cmp::max(0, j as i32 - radius as i32) as usize;
            let c2 = cmp::min((arr.shape()[1] as i32) - 1, j as i32 + radius as i32) as usize;
            // Get the window of appropriate radius size
            let window = arr.slice(s![r1..=r2, c1..=c2]);
            let mut window_1d: Vec<&i32> = window.iter().collect();
            window_1d.sort();
            // Get the median value and update the output array
            let median = match window_1d.len() % 2 {
                0 => (*window_1d[window_1d.len() / 2] as f32 + *window_1d[window_1d.len() / 2 - 1] as f32) / 2.,
                _ => *window_1d[window_1d.len() / 2] as f32
            };
            filtered[[i, j]] = median as i32;
        }
    }
    let filtered_vec = crate::arr2_to_vec2_i32(filtered);

    // Crop the image to remove the outer padding used for the filter
    unpad_edge_vector(&filtered_vec, radius)
}

/// Rust equivalent of :func:`median_filter.median_filter_multiprocessed`.
///
/// :param data: 2D image.
/// :type img: Vec<Vec<i32>>
/// :param radius: Radius of the kernel.
/// :type radius: usize
/// :param cpu_option: The number of CPUS to use. If None, all available CPUs will be used.
/// :type cpu_ooption: Option<usize>
/// :return: Filtered image.
/// :rtype: Vec<Vec<i32>>
///
#[pyfunction]
pub fn median_filter_multithread(data: Vec<Vec<i32>>, radius: usize, cpu_option: Option<usize>) -> Vec<Vec<i32>> {
    // Pad the input data with the shape of radius
    let padded_data = pad_edge_vector(&data, radius);

    // Initialize input array and output array
    let arr = crate::vec2_to_arr2_i32(padded_data);
    let mut filtered = ndarray::Array2::zeros((arr.shape()[0], arr.shape()[1]));

    // Get available cpus
    let available_cpus = num_cpus::get();

    // Determine number of threads to use
    let threads = match cpu_option {
        Some(cpus) => if cpus < available_cpus && cpus >= 1 { cpus } else { available_cpus },
        None => available_cpus
    };
    let rows_per_thread = (arr.shape()[0]/ threads) + 1;

    // Split the result image into bands
    let bands = filtered.axis_chunks_iter_mut(Axis(0), rows_per_thread);

    // Create a reference to the array to share across threads
    let arr_ref = &arr;

    // Run the median filter
    crossbeam::scope(|spawner| {
        // For each band...
        for (k, mut band) in bands.enumerate() {
            // Create the thread for this band
            spawner.spawn(
                move |_| {
                // Iterate through each pixel
                for i in 0..band.shape()[0] {
                    for j in 0..band.shape()[1] {
                        // Get slice indices, ensuring they don't go out of bounds
                        let r1 = cmp::max(0, (i+(k*rows_per_thread)) as i32 - radius as i32) as usize;
                        let r2 = cmp::min((arr_ref.shape()[0] as i32) - 1, (i+(k*rows_per_thread)) as i32 + radius as i32) as usize;
                        let c1 = cmp::max(0, j as i32 - radius as i32) as usize;
                        let c2 = cmp::min((arr_ref.shape()[1] as i32) - 1, j as i32 + radius as i32) as usize;
                        // Get the window of appropriate radius size
                        let window = arr_ref.slice(s![r1..=r2, c1..=c2]);
                        let mut window_1d: Vec<&i32> = window.iter().collect();
                        window_1d.sort();
                        // Get the median value and update the output array
                        let median = match window_1d.len() % 2 {
                            0 => (*window_1d[window_1d.len() / 2] as f32 + *window_1d[window_1d.len() / 2 - 1] as f32) / 2.,
                            _ => *window_1d[window_1d.len() / 2] as f32
                        };
                        band[[i, j]] = median as i32;
                    }
                }
            });
        }
    }).unwrap();

    // Return the filtered image
    let filtered_vec = crate::arr2_to_vec2_i32(filtered);

    // Crop the image to remove the outer padding used for the filter
    unpad_edge_vector(&filtered_vec, radius)
}

/// Rust equivalent of :func:`adaptive_threshold.adaptive_threshold`.
/// 
/// :param data: 2D image.
/// :type data: Vec<Vec<i32>>
/// :param radius: Radius of the kernel.
/// :type radius: usize
/// :param offset: Offset value.
/// :type offset: f64
/// :return: Binarized image.
/// :rtype: Vec<Vec<i32>>
/// 
#[pyfunction]
pub fn adaptive_threshold(data: Vec<Vec<i32>>, radius: usize, offset: f32) -> Vec<Vec<i32>> {
    // Initialize input and output arrays
    let arr = crate::vec2_to_arr2_i32(data);
    let mut binarized = ndarray::Array2::zeros((arr.shape()[0], arr.shape()[1]));

    // Iterate through each pixel
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            // Get slice indices, ensuring they don't go out of bounds
            let r1 = cmp::max(0, i as i32 - radius as i32) as usize;
            let r2 = cmp::min((arr.shape()[0] as i32) - 1, i as i32 + radius as i32) as usize;
            let c1 = cmp::max(0, j as i32 - radius as i32) as usize;
            let c2 = cmp::min((arr.shape()[1] as i32) - 1, j as i32 + radius as i32) as usize;
            // Get the window of appropriate radius size
            let window = arr.slice(s![r1..=r2, c1..=c2]);
            let window_1d: Vec<&i32> = window.iter().collect();
            // Threshold is the mean of the window's extrema, including the offset
            let threshold = ((*window_1d.iter().min().unwrap() + *window_1d.iter().max().unwrap()) as f32 / 2.) - offset;
            // Assign the output to a 0 or 1, if it is below or above the threshold, respectively
            binarized[[i, j]] = match arr[[i, j]] as f32 >= threshold {
                true => 1,
                false => 0
            };
        }
    }
    // Return the binary array
    crate::arr2_to_vec2_i32(binarized)
}

/// Rust equivalent of :func:`adaptive_threshold.adaptive_threshold_multiprocessed`.
/// 
/// :param data: 2D image.
/// :type data: Vec<Vec<i32>>
/// :param radius: Radius of the kernel.
/// :type radius: usize
/// :param offset: Offset value.
/// :type offset: f64
/// :param cpu_option: Number of CPUs to use. If ``None``, all available CPUs will be used.
/// :type cpu_option: Option<usize>
/// :return: Binarized image.
/// :rtype: Vec<Vec<i32>>
/// 
#[pyfunction]
pub fn adaptive_threshold_multithread(data: Vec<Vec<i32>>, radius: usize, offset: f64, cpu_option: Option<usize>) -> Vec<Vec<i32>> {

    // Initialize input and output arrays
    let arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = crate::vec2_to_arr2_i32(data);
    let mut binarized: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = ndarray::Array2::zeros((arr.shape()[0], arr.shape()[1]));

    // Get available cpus
    let available_cpus = num_cpus::get();

    // Determine number of threads to use
    let threads = match cpu_option {
        Some(cpus) => if cpus < available_cpus && cpus >= 1 { cpus } else { available_cpus },
        None => available_cpus
    };
    let rows_per_thread = (arr.shape()[0]/ threads) + 1;

    // Split the result image into bands
    let bands = binarized.axis_chunks_iter_mut(Axis(0), rows_per_thread);

    // Create a reference to the array to share across threads
    let arr_ref = &arr;

    // Run the adaptive_threshold algorithm 
    crossbeam::scope(|spawner| {
        // For each band...
        for (k, mut band) in bands.enumerate() {
            // Create the thread for this band
            spawner.spawn(move |_| {
                // Iterate through each pixel
                for i in 0..band.shape()[0] {
                    for j in 0..band.shape()[1] {
                        // Get slice indices, ensuring they don't go out of bounds
                        let r1 = cmp::max(0, (i+(k*rows_per_thread)) as i32 - radius as i32) as usize;
                        let r2 = cmp::min((arr_ref.shape()[0] as i32) - 1, (i+(k*rows_per_thread)) as i32 + radius as i32) as usize;
                        let c1 = cmp::max(0, j as i32 - radius as i32) as usize;
                        let c2 = cmp::min((arr_ref.shape()[1] as i32) - 1, j as i32 + radius as i32) as usize;
                        // Get the window of appropriate radius size
                        let window = arr_ref.slice(s![r1..=r2, c1..=c2]);
                        let window_1d: Vec<&i32> = window.iter().collect();
                        // Threshold is the mean of the window's extrema, including the offset
                        let threshold = ((*window_1d.iter().min().unwrap() + *window_1d.iter().max().unwrap()) as f64 / 2.) - offset;
                        // Assign the output to a 0 or 1, if it is below or above the threshold, respectively
                        band[[i, j]] = match arr_ref[[(i+(k*rows_per_thread)), j]] as f64 >= threshold {
                            true => 1,
                            false => 0
                        };
                    }
                }
            });
        }
    }).unwrap();

    // Return the binary array
    crate::arr2_to_vec2_i32(binarized)
}

/// Runs the bilateral filter on the input image, using a border type of
/// `BORDER_ISOLATED <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5>`_.
/// This implementation follows the guassian kernel and brighness kernel
/// implementations from this `Non-Linear Image Filters <https://www.youtube.com/watch?v=7FP7ndMEfsc&t=467s>`_ video.
///
/// :param data: 2D image.
/// :type data: Vec<Vec<i32>>
/// :param brightness_sigma: The sigma of the Brightness kernel, which smooths pixels based on their brightness similarity to the current pixel.
/// :type brightness_sigma: u64
/// :param spatial_sigma: The sigma of the Gaussian Kernel, which smooths pixels based on their proximity to the current pixel.
/// :type spatial_sigma: u64
/// :param radius: the radius of the pixel neighborhood considered in the filter
/// :type radius: u64
/// :return: Filtered image.
/// :rtype: Vec<Vec<i32>>
///
#[pyfunction]
pub fn bilateral_filter(data: Vec<Vec<i32>>, mut brightness_sigma: u64, mut spatial_sigma: u64, radius: u64) -> Vec<Vec<i32>> {

    // Convert &Vec<Vec<f32>> to Array2<i32>
    let data = crate::vec2_to_arr2_i32(data);

    // Initialize height and width values
    let height = data.shape()[0];
    let width = data.shape()[1];

    // Initialize the output array
    let mut result = ndarray::Array2::zeros((data.shape()[0], data.shape()[1]));

    // Correct invalid input values
    if let 0 = brightness_sigma {
        brightness_sigma = 1;
    }
    if let 0 = spatial_sigma {
        spatial_sigma = 1;
    }
    
    // Calculate function constants to save computation time
    let spatial_sigma_squared: u64 = checked_pow(spatial_sigma, 2).expect("Overflow in squaring spatial_sigma");
    let spatial_gaussian_const: f64 = 1.0 / (2.0 * PI * (spatial_sigma_squared as f64));
    let brightness_sigma_squared: u64 = checked_pow(brightness_sigma, 2).expect("Overflow in squaring brightness_sigma");
    let brightness_gaussian_const: f64 = 1.0 / ((2.0 * PI).sqrt() * (brightness_sigma as f64));

    // Precalculate Spatial Gaussian results 
    let mut spatial_gaussian_array = ndarray::Array2::zeros((height*2-1, width*2-1));
    for m in -(height as i64)+1i64..(height as i64) {
        for n in -(width as i64)+1i64..(width as i64) {
            spatial_gaussian_array[[(m+(height as i64)-1) as usize, (n+(width as i64)-1) as usize]] = 
                spatial_gaussian_const * exp( 
                (-0.5f64) * ((((checked_pow(m, 2).expect("Overflow in squaring m") as u64) + 
                               (checked_pow(n, 2).expect("Overflow in squaring n") as u64)) as f64)
                            / (spatial_sigma_squared as f64)));  
        }
    }

    // Pre-calculate Brightness Gaussian results
    let mut brightness_gaussian_array = ndarray::Array1::zeros(512);
    for k in -255i64..=255i64 {
        brightness_gaussian_array[(k+255) as usize] = 
            brightness_gaussian_const * exp(
            (-0.5)*((checked_pow(k, 2).expect("Overflow in squaring k") as f64)
            / (brightness_sigma_squared as f64)));
    }

    // Run the filter
    for i in 0i64..(height as i64) { // For each pixel in the image
        for j in 0i64..(width as i64) {
            let mut normalization_factor: f64 = 0.0;
            let mut sum: f64 = 0.0;

            // For each pixel in the neighborhood of the current pixel
            for m in ((i as i64)-(radius as i64))..=((i as i64)+(radius as i64)) {
                for n in ((j as i64)-(radius as i64))..=((j as i64)+(radius as i64)) {
                    // If this pixel is out of bounds, skip it
                    if m >= 0 && m < (height as i64) && n >= 0 && n < (width as i64) {
                        // Add to Normalization Factor and Sum
                        let spatial_and_brightness: f64 = 
                        spatial_gaussian_array[[(i - m + (height as i64) - 1) as usize, 
                                                (j - n + (width as i64) - 1) as usize]] 
                            * brightness_gaussian_array[((data[[m as usize,n as usize]] 
                                                 - data[[i as usize, j as usize]]) + 255) as usize];
                        normalization_factor += spatial_and_brightness;
                        sum += (data[[m as usize, n as usize]] as f64) * spatial_and_brightness;
                    }
                }
            }
            // Calculate result and save it in output 
            let final_num = (1.0 / normalization_factor) * sum;
            result[[i as usize, j as usize]] = final_num.round() as i32;
        }
    }

    // Return the result
    crate::arr2_to_vec2_i32(result)
}

/// Same as :func:`bilateral_filter`, but leverages multithreading in order to speed up computation time.
/// 
/// :param data: 2D image.
/// :type data: Vec<Vec<i32>>
/// :param brightness_sigma: The sigma of the Brightness kernel, which smooths pixels based on their brightness similarity to the current pixel.
/// :type brightness_sigma: u64
/// :param spatial_sigma: The sigma of the Gaussian Kernel, which smooths pixels based on their proximity to the current pixel.
/// :type spatial_sigma: u64
/// :param radius: the radius of the pixel neighborhood considered in the filter
/// :type radius: u64
/// :param cpu_option: Number of CPUs to use. If ``None``, all available CPUs will be used.
/// :type cpu_option: Option<usize>
/// :return: Filtered image.
/// :rtype: Vec<Vec<i32>>
///
#[pyfunction]
pub fn bilateral_filter_multithread(data: Vec<Vec<i32>>, mut brightness_sigma: u64, 
                                    mut spatial_sigma: u64, radius: u64, cpu_option: Option<usize>) -> Vec<Vec<i32>> {

    // Convert Vec<Vec<f32>> to Array2<i32>
    let data_arr: ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>> = crate::vec2_to_arr2_i32(data);

    // Initialize height and width values
    let height = data_arr.shape()[0];
    let width = data_arr.shape()[1];

    // Initialize the output array
    let mut result = ndarray::Array2::zeros((data_arr.shape()[0], data_arr.shape()[1]));

    // Correct invalid input values
    if let 0 = brightness_sigma {
        brightness_sigma = 1;
    }
    if let 0 = spatial_sigma {
        spatial_sigma = 1;
    }
    
    // Calculate function constants to save computation time
    let spatial_sigma_squared: u64 = checked_pow(spatial_sigma, 2).expect("Overflow in squaring spatial_sigma");
    let spatial_gaussian_const: f64 = 1.0 / (2.0 * PI * (spatial_sigma_squared as f64));
    let brightness_sigma_squared: u64 = checked_pow(brightness_sigma, 2).expect("Overflow in squaring brightness_sigma");
    let brightness_gaussian_const: f64 = 1.0 / ((2.0 * PI).sqrt() * (brightness_sigma as f64));

    // Precalculate Spatial Gaussian results 
    let mut spatial_gaussian_array: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = ndarray::Array2::zeros((height*2-1, width*2-1));
    for m in -(height as i64)+1i64..(height as i64) {
        for n in -(width as i64)+1i64..(width as i64) {
            spatial_gaussian_array[[(m+(height as i64)-1) as usize, (n+(width as i64)-1) as usize]] = 
                spatial_gaussian_const * exp( 
                (-0.5f64) * ((((checked_pow(m, 2).expect("Overflow in squaring m") as u64) + 
                               (checked_pow(n, 2).expect("Overflow in squaring n") as u64)) as f64)
                            / (spatial_sigma_squared as f64)));  
        }
    }

    // Pre-calculate Brightness Gaussian results
    let mut brightness_gaussian_array: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = ndarray::Array1::zeros(512);
    for k in -255i64..=255i64 {
        brightness_gaussian_array[(k+255) as usize] = 
            brightness_gaussian_const * exp(
            (-0.5)*((checked_pow(k, 2).expect("Overflow in squaring k") as f64)
            / (brightness_sigma_squared as f64)));
    }

    // Get available cpus
    let available_cpus = num_cpus::get();

    // Determine number of threads to use
    let threads = match cpu_option {
        Some(cpus) => if cpus < available_cpus && cpus >= 1 { cpus } else { available_cpus },
        None => available_cpus
    };
    let rows_per_thread = (data_arr.shape()[0]/ threads) + 1;

    // Split the result array into bands
    let bands = result.axis_chunks_iter_mut(Axis(0), rows_per_thread);

    // Create a reference to the array to share across threads
    let arr_ref = &data_arr;
    let spatial_gaussian_array_ref = &spatial_gaussian_array;
    let brightness_gaussian_array_ref = &brightness_gaussian_array;

    // Run the bilateral filter
    crossbeam::scope(|spawner| {
        // For each band...
        for (k, mut band) in bands.enumerate() {
            // Create the thread for this band
            spawner.spawn(move |_| {
                // For each pixel in the band
                for i in ((k*rows_per_thread) as i64)..((band.shape()[0] + (k*rows_per_thread)) as i64) {
                    for j in 0i64..(band.shape()[1] as i64) {
                        let mut normalization_factor: f64 = 0.0;
                        let mut sum: f64 = 0.0;
            
                        // For each pixel in the neighborhood of the current pixel
                        for m in ((i as i64)-(radius as i64))..=((i as i64)+(radius as i64)) {
                            for n in ((j as i64)-(radius as i64))..=((j as i64)+(radius as i64)) {
                                // If this pixel is out of bounds, skip it
                                if m >= 0 && m < (height as i64) && n >= 0 && n < (width as i64) {
                                    // Add to Normalization Factor and Sum
                                    let spatial_and_brightness: f64 = 
                                    spatial_gaussian_array_ref[[(i - m + (height as i64) - 1) as usize, 
                                                            (j - n + (width as i64) - 1) as usize]] 
                                        * brightness_gaussian_array_ref[((arr_ref[[m as usize,n as usize]] 
                                                             - arr_ref[[i as usize, j as usize]]) + 255) as usize];
                                    normalization_factor += spatial_and_brightness;
                                    sum += (arr_ref[[m as usize, n as usize]] as f64) * spatial_and_brightness;
                                }
                            }
                        }
                        // Calculate result and save it in output 
                        let final_num = (1.0 / normalization_factor) * sum;
                        band[[((i as usize)-(k*rows_per_thread)) as usize, j as usize]] = final_num.round() as i32;
                    }
                }
            });
        }
    }).unwrap();

    // Return the result
    crate::arr2_to_vec2_i32(result)
}