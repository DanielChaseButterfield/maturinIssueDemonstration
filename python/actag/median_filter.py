import numpy as np
import multiprocessing as mp

def median_filter(img: np.ndarray, kernel_radius: int) -> np.ndarray:
    '''
    Median filter for 2D images.

    :param img: 2D image.
    :type img: np.ndarray
    :param kernel_radius: Radius of the kernel.
    :type kernel_radius: int
    :return: Filtered image.
    :rtype: np.ndarray
    '''
    # Ensure that the input parameters are valid
    if not isinstance(img, np.ndarray):
        if isinstance(img, list):
            if isinstance(img[0], list):
                img = np.asarray(img) # Convert to an array if a list of lists was passed in
            else:
                raise ValueError("Median Filter: Input image must be a numpy array.")
    if len(img.shape) != 2:
        raise ValueError("Median Filter: Input image must be a 2D image.")
    if not isinstance(kernel_radius, int):
        raise ValueError("Median Filter: Kernel radius must be an integer.")
    
    # Pad the image by extended the edge values
    padded_img = np.pad(img, kernel_radius, mode="edge")
    # Create an empty image for the filtered values
    filtered_img = np.zeros_like(img)
    # Iterate through the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Set the filtered output value to the median value of the kernel
            filtered_img[i, j] = np.median(padded_img[i : i + 2 * kernel_radius + 1, j : j + 2 * kernel_radius + 1])
    # Return the filtered image
    return filtered_img


def median_filter_multiprocessed(img: np.ndarray, kernel_radius: int, num_processes: int=None) -> np.ndarray:
    '''
    Same as :func:`median_filter`, but leverages multiprocessing to speed up computation time.

    :param img: 2D image.
    :type img: np.ndarray
    :param kernel_radius: Radius of the kernel.
    :type kernel_radius: int
    :param num_processes: The number of processes to use. The default is ``None``, which will attempt to use the maximum number of CPUs available, as reported by the OS.
    :type num_processes: int, default_value=None

    :return: Filtered image.
    :rtype: np.ndarray
    '''
    # Ensure that the input parameters are valid
    if not isinstance(img, np.ndarray):
        if isinstance(img, list):
            if isinstance(img[0], list):
                img = np.asarray(img) # Convert to an array if a list of lists was passed in
            else:
                raise ValueError("Median Filter: Input image must be a numpy array.")
    if len(img.shape) != 2:
        raise ValueError("Median Filter: Input image must be a 2D image.")
    if not isinstance(kernel_radius, int):
        raise ValueError("Median Filter: Kernel radius must be an integer.")
    if num_processes is not None and not isinstance(num_processes, int):
        raise ValueError("Median Filter: Number of processes must be an integer.")
    # Set the number of processes to the maximum number of CPUs available if no value was passed in
    if num_processes is None:
        num_processes = min(mp.cpu_count(), img.shape[0])
    # Ensure the number of processes is not greater than the number of rows in the image
    elif num_processes > img.shape[0]:
        print(f"Warning: Number of processes ({num_processes}) is greater than the number of rows in the image ({img.shape[0]}) and has been reduced to match the number of rows in the image.")
        num_processes = img.shape[0]
    # Identify the indices to divide the image up to have one unique section per process
    section_indices = divide_array(img.shape[0], num_processes)
    # Pad the image by extended the edge values
    padded_img = np.pad(img, kernel_radius, mode="edge")
    sub_images = []
    for i in range(num_processes):
        sub_images.append(padded_img[section_indices[i][0] : section_indices[i][1] + 2 * kernel_radius, :])

    # Create the pool of processes and run the filter on each sub-image
    pool = mp.Pool(processes=num_processes)
    sub_filtered_images = pool.starmap(filter_sub_image, [(sub_img, kernel_radius) for sub_img in sub_images])
    # Combine the sub-images into a single image
    filtered_img = np.vstack(sub_filtered_images)

    # Close the multiprocessing pool
    pool.close()

    # Return the filtered image
    return filtered_img

def divide_array(arr_len: int, num_sections: int) -> list:
    '''
    Divides an array into a specified number of sections and 
    returns a list containing the start and end indices for 
    each section.

    :param arr_len: Length of the array.
    :type arr_len: int
    :param num_sections: Number of sections to divide the array into.
    :type num_sections: int
    :return: List containing the start and end indices for each section.
    :rtype: list
    '''
    # Ensure that the input parameters are valid
    if arr_len < num_sections:
        raise ValueError("List length must be greater than or equal to number of portions")
    quotient, remainder = divmod(arr_len, num_sections)
    result = []
    start = 0
    # Iterate through the number of sections and determine the start and end indices for each section
    for i in range(num_sections):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        result.append((start, end))
        start = end
    # Return a list containing the start and end indices for each section
    return result

def filter_sub_image(sub_img: np.ndarray, kernel_radius: int) -> np.ndarray:
    '''
    Filters a sub-image using a median filter.
    
    :param sub_img: Sub-image to filter.
    :type sub_img: np.ndarray
    :param kernel_radius: Radius of the kernel.
    :type kernel_radius: int
    :return: Filtered sub-image.
    :rtype: np.ndarray
    '''
    # Extract the non-padded portion of the sub-image
    non_padded_sub_img = sub_img[kernel_radius:-kernel_radius, kernel_radius:-kernel_radius]
    # Create an empty image for the filtered values
    sub_filtered_img = np.zeros_like(non_padded_sub_img)
    # Iterate through the image
    for i in range(sub_filtered_img.shape[0]):
        for j in range(sub_filtered_img.shape[1]):
            sub_filtered_img[i, j] = np.median(sub_img[i : i + 2 * kernel_radius + 1, j : j + 2 * kernel_radius + 1])
    # Return the filtered image
    return sub_filtered_img


if __name__ == "__main__":
    my_img = np.random.randint(0, 255, size=(500, 500))
    import time
    t1 = time.time()
    out1 = median_filter(my_img, 2)
    t2 = time.time()
    out2 = median_filter_multiprocessed(my_img, 2)
    t3 = time.time()
    print(f'Elapsed time for single process: {t2 - t1} seconds')
    print(f'Elapsed time for multi process: {t3 - t2} seconds')
    print(f'All values equal for single and multi process?: {np.all(out1 == out2)}')