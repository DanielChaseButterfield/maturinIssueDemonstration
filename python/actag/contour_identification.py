import numpy as np
import sys

#path_to_actag_detection_src = __file__[:__file__.find("/actag") + 1] + "actag/actag_detection/src/"
#sys.path.append(path_to_actag_detection_src)

from . import sonar_parameters as sprm
from . import tag_parameters as tprm

def get_contours(img: np.ndarray, 
                 sonar_params: sprm.SonarParams,
                 tag_params: tprm.AcTagParams,
                 tag_size_tolerance: float=0.2,
                 tag_area_tolerance: float=0.2,
                 min_tag_area_ratio: float=0.1,
                 reject_black_shapes: bool=True, 
                 reject_white_shapes: bool=False, 
                 reject_by_tag_size: bool=True,
                 reject_by_area: bool=True,
                 ) -> list:
    '''
    Contours are identified using a border following algorithm described 
    in "`Border following: new definition gives improved borders 
    <https://researchrepository.murdoch.edu.au/id/eprint/18862/1/border_following.pdf>`_".
      
    In this implementation we iterate through the 'layers', or levels in 
    the parenthood heirarchy. So the first layer identifies all of the 
    shapes whose parent contour is the frame of the image. Then the 
    second layer identifies all of the shapes whose parent contour is 
    found in the first layer, and so on. The algorithm terminates when it
    fails to find any contours in the layer it is working on, or in other 
    words, when the image it is working on is a zero array.
      
    :param img: the image of which the contours will be found.
    :type img: np.ndarray
    :param sonar_params: The parameters of the sonar that took the image.
    :type sonar_params: SonarParams
    :param tag_params: The parameters of the tag that is being detected.
    :type tag_params: AcTagParams
    :param tag_size_tolerance: The tolerance for rejecting contours that are too long in the range or azimuth axes
    :type tag_size_tolerance: float
    :param tag_area_tolerance: The tolerance for rejecting contours that are too large or too small in pixel area
    :type tag_area_tolerance: float
    :param min_tag_area_ratio: The ratio of the smallest size of a tag when rotated to the full size of the tag
    :type min_tag_area_ratio: float
    :param reject_black_shape: Flag to reject black contours (i.e. outer area is white, inner is black)
    :type reject_black_shape: bool
    :param reject_white_shapes: Flag to reject white contours (i.e. outer area is black, inner is white)
    :type reject_white_shapes: bool
    :param reject_by_tag_size: Flag to reject contours that are too long in range or azimuth axes
    :type reject_by_tag_size: bool
    :param reject_by_area: Flag to reject contours that are too large or too small in pixel area
    :type reject_by_area: bool
    :return: A list containing all of the image contours.
    :rtype: list
    '''
    
    # Ensure that the inputs are valid
    if not isinstance(img, np.ndarray):
        if isinstance(img, list):
            if isinstance(img[0], list):
                img = np.asarray(img) # Convert to an array if a list of lists was passed in
            else:
                raise ValueError("Contour Identification: Input image must be a numpy array.")
    if len(img.shape) != 2:
        raise ValueError("Contour Identification: Input image must be a 2D image.")
    
    # Contours are eight-connected, define neighbors for this case
    eight_connected_neighbors = np.array([[0, 0, 1],    # Direction 0, row + 0, col + 1
                                          [1, -1, 1],   # Direction 1, row - 1, col + 1
                                          [2, -1, 0],   # Direction 2, row - 1, col + 0
                                          [3, -1, -1],  # Direction 3, row - 1, col - 1
                                          [4, 0, -1],   # Direction 4, row + 0, col - 1
                                          [5, 1, -1],   # Direction 5, row + 1, col - 1
                                          [6, 1, 0],    # Direction 6, row + 1, col + 0
                                          [7, 1, 1]])   # Direction 7, row + 1, col + 1
    # Flip the array to make it easy to work around the neighbors in a clockwise order
    eight_connected_neighbors = np.flipud(eight_connected_neighbors)
    
    # Get the image dimensions
    img_rows, img_cols = img.shape
    
    # Calculate useful values for contour rejection based on area or tag size
    if reject_by_area or reject_by_tag_size:
        range_pixel_resolution = (sonar_params.max_range - sonar_params.min_range) / img_rows # Meters per pixel
        max_tag_range_pixels = tag_params.tag_diag / range_pixel_resolution
        ranges = np.linspace(sonar_params.max_range, sonar_params.min_range, img_rows) # Max to min, to line up with image
        azimuth_pixel_resolution = ranges * sonar_params.horizontal_aperture / img_cols # Meters per pixel
        
    # Set up an array to use for processing the image
    processing_img = img.copy().astype(np.int16)
    
    # Variables for tracking the contours
    contour_list = [] # list of lists, where each contour is a list containing [layer_num, row_idx_1, col_idx_1, row_idx_2, col_idx_2, ..., row_idx_n, col_idx_n]
    layer_num = 0
    
    # Iterate throught the image, layer by layer, identifying any contours
    while not np.all(processing_img == 0):
        # With the very first layer being white shapes surrounded by black (we assume a black frame around the
        # input image, even though in the code that border doesn't need to be present), we can then say that 
        # every odd layer number indicates when we're identifying white shapes, and every even layer number 
        # indicates when we're identifying black shapes.
        layer_num += 1
        # Skip any layers the user has specified should be skipped
        if (layer_num % 2 == 0 and reject_black_shapes) or (layer_num % 2 == 1 and reject_white_shapes):
            # Merge outer 'frame' into current layer and invert the colors to move to the next layer
            processing_img = merge_and_invert_colors(processing_img)
            continue
        # Array to indicate if a given pixel is 'searchable'. Pixels part of, or inside of, a contour are 
        # marked so that they are not searchable in the current layer.
        searchable_img = np.zeros_like(processing_img)
        # Iterate through the current image, identifying all contours
        for i, row in enumerate(processing_img):
            for j, pixel_val in enumerate(row):
                
                # Trigger contour following when we find a searchable pixel that is white
                if pixel_val == 1 and searchable_img[i, j] == 0:
                    searchable_img, contour_chain, pixels_in_contour, extrema = trace_contour(processing_img, 
                                                                                     searchable_img, i, j, 
                                                                                     eight_connected_neighbors)
                    
                    # Determine if the contour is added to our list or not
                    rmin, rmax, cmin, cmax = extrema
                    if reject_by_tag_size or reject_by_area:
                        # Calculate useful values for contour rejection
                        range_pix = rmax - rmin
                        azimuth_pix = cmax - cmin
                        r_centroid = rmin + ((rmax - rmin) // 2)
                        max_tag_azimuth_pixels = tag_params.tag_diag / azimuth_pixel_resolution[r_centroid]
                        # Reject contours based on known tag size and user-specified tolerance
                        if reject_by_tag_size:
                            if range_pix > max_tag_range_pixels * (1 + tag_size_tolerance):
                                continue
                            if azimuth_pix > max_tag_azimuth_pixels * (1 + tag_size_tolerance):
                                continue
                        # Reject contours based on known area limitations and user-specified ratios and tolerances
                        if reject_by_area:
                            max_num_pixels = max_tag_range_pixels * max_tag_azimuth_pixels
                            min_num_pixels = max_num_pixels * min_tag_area_ratio

                            if pixels_in_contour > max_num_pixels * (1 + tag_area_tolerance):
                                continue
                            elif pixels_in_contour < min_num_pixels * (1 - tag_area_tolerance):
                                continue
                    # Add the contour to the list
                    contour_list.append([layer_num] + contour_chain)

                # If not on or within a contour, set the pixel to -1 in preparation of next layer
                elif searchable_img[i, j] == 0:
                    processing_img[i, j] = -1
        # Invert the image, changing what was 0 to 1 and what was 1 to 0
        processing_img[processing_img == 1] = -1
        processing_img += 1
    # Return the contour list
    return contour_list
    
def merge_and_invert_colors(binary_img: np.ndarray) -> np.ndarray:
    '''
    Merge the outermost two layers and invert the colors of the image. 
    This prepare the next layer of the image to be searched for contours.

    :param binary_img: 2D image.
    :type binary_img: np.ndarray:
    :return: The merged and inverted image.
    :rtype: np.ndarray
    '''

    new_img = binary_img.copy()
    # Set anything connected to the first pixel to -1
    new_img = flood_fill(new_img, 0, 0, -1)
    # Set all white pixels to -1
    new_img[new_img == 1] = -1
    # Increase the value of all pixels in the image by one, so all black pixels bexome white
    # and all white pixels were previously set to -1 and are now black
    new_img += 1
    # Return the inverted image
    return new_img

def flood_fill(img: np.ndarray, row_seed: int, col_seed: int, fill_val: int) -> np.ndarray:
    '''
    Fill anything connected to the seeded pixel with the designated fill value.

    :param img: 2D image.
    :type img: np.ndarray
    :param row_seed: The row index of the pixel used to start the fill.
    :type row_seed: int
    :param col_seed: The column index of the pixel used to start the fill.
    :type col_seed: int
    :param fill_val: The value with which to fill the seeded pixel and its connected neighbors.
    :type fill_val: int
    :return: The image that has been filled.
    :rtype: np.ndarray
    '''

    new_img = img.copy()
    # Fill using four-connected neighbors
    neighbors = np.array([[0, 1],   # Pixel to the right (col + 1)
                          [-1, 0],  # Pixel above (row - 1)
                          [0, -1],  # Pixel to the left (col - 1)
                          [1, 0]])  # Pixel below (row + 1)
    # Get bounds 
    [rows, cols] = new_img.shape
    # Determine value at start pixel
    val_of_seed_pixel = new_img[row_seed, col_seed]
    # If the start pixel is already the desired value, return the image
    if val_of_seed_pixel == fill_val:
        return new_img
    # Create a queue to hold the pixels to be filled
    new_img[row_seed, col_seed] = fill_val
    myqueue = [[row_seed, col_seed]]
    # Iterate through the queue
    while myqueue:
        r, c = myqueue.pop()
        # Check the neighbors of the current pixel
        for add_r, add_c in neighbors:
            nr, nc = r+add_r, c+add_c
            # If the neighbor is in bounds and is the old value, fill it and add it to the queue
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if new_img[nr, nc] == val_of_seed_pixel:
                new_img[nr, nc] = fill_val
                myqueue.append([nr, nc])
    # Return the image
    return new_img

def trace_contour(contour_img: np.ndarray, 
                  searchable_img: np.ndarray, 
                  row_seed: int, 
                  col_seed: int, 
                  neighbors: np.ndarray
                  ) -> tuple[np.ndarray, list, int, list]:
    '''
    From a starting pixel, trace and return the entirety of the contour.

    :param contour_img: Binarized 2D image for this layer.
    :type contour_img: np.ndarray
    :param searchable_img: Array to indicate if a given pixel is "searchable". Pixels part of, or inside of, a contour are marked so that they are not searchable in the current layer.
    :type searchable_img: np.ndarray
    :param row_seed: The row index of the starting pixel of the contour.
    :type row_seed: int
    :param col_seed: The column index of the starting pixel of the contour.
    :type col_seed: int
    :param neighbors: Array of the eight-connected neighbors of a pixel.
    :type neighbors: np.ndarray
    :return: A tuple containing the searchable image, the contour chain, the number of pixels in the contour, and the min/max pixel values of the contour.
    :rtype: tuple[np.ndarray, list, int, list]
    '''
    # Set up variables used in tracing contour
    fully_traced = False
    last_direction = 4
    cur_r, cur_c = row_seed, col_seed
    contour_pixel_chain = [row_seed, col_seed]
    # Work around the contour in a clockwise direction
    while not fully_traced:
        ray_path = np.roll(neighbors, last_direction, axis=0)
        for direction, add_r, add_c in ray_path:
            # Contour has been traced if we return to the starting pixel and direction
            if cur_r == row_seed and cur_c == col_seed and direction == 4:
                fully_traced = True
                # If the contour contains more than a single pixel, it will have a duplicate in the first
                # and last positions, so we can remove the last pixel, as it is redundant
                if len(contour_pixel_chain) > 2:
                    del contour_pixel_chain[-2:]
                # Then break from this loop, as we have traced the full contour
                break
            # Skip anything that would trigger moving outside of the image
            if cur_r + add_r < 0 or cur_r + add_r >= contour_img.shape[0] or cur_c + add_c < 0 or cur_c + add_c >= contour_img.shape[1]:
                continue
            # The first pixel in the ray path that is white is the next pixel in the contour
            if contour_img[cur_r + add_r, cur_c + add_c] == 1:
                # Update current indices and append to the chain path
                cur_r += add_r
                cur_c += add_c
                contour_pixel_chain.append(cur_r)
                contour_pixel_chain.append(cur_c)
                # Update the searchable array
                searchable_img[cur_r, cur_c] = 1
                # Update the last direction we came from 
                # (we want to know the dirction opposite of the one we just took)
                last_direction = (direction + 4) % 8
                break
    # Initialize the count of how many pixels are in the contour
    total_pixels_in_contour = len(contour_pixel_chain) // 2
    # Identify the bounding box of the contour
    row_col_array = np.array(contour_pixel_chain).reshape(-1, 2)
    r_min, c_min = np.min(row_col_array, axis=0)
    r_max, c_max = np.max(row_col_array, axis=0)
    # After tracing the full contour, if the contour contains more than three pixels then it may
    # contain zeros inside of it, which we want to mark as unsearchable.
    if len(contour_pixel_chain) > 2 * 3:
        # Extract the area around the contour
        searchable_around_contour = searchable_img[r_min:r_max+1, c_min:c_max+1]
        # Pad with a frame of zeros to ensure the flood fill will work as intended
        searchable_around_contour = np.pad(searchable_around_contour, 1, 'constant', constant_values=0)
        # Fill outer area of 0's with 1's, invert, and bitwise or with original to get filled area
        filled_inside_contour = flood_fill(searchable_around_contour, 0, 0, 1)
        filled_inside_contour[filled_inside_contour == 1] = -1
        filled_inside_contour += 1

        total_pixels_in_contour += np.sum(filled_inside_contour)
        searchable_around_contour = searchable_around_contour | filled_inside_contour
        # Replace the appropriate section original searchable array with the updated version
        searchable_img[r_min:r_max+1, c_min:c_max+1] = searchable_around_contour[1:-1, 1:-1]
    # Return the searchable image, contour and the total number of pixels in the contour, and max/min vals
    return searchable_img, contour_pixel_chain, total_pixels_in_contour, [r_min, r_max, c_min, c_max]

def convert_contour_list_to_img(contours_list: list, img_shape: tuple[int, int]) -> np.ndarray:
    '''
    Visualize a contour list onto an image of the specified shape.

    :param contours_list: A list of contours to visualize.
    :type contours_list: list
    :param img_shape: The shape of the image to visualize the contours on.
    :type img_shape: tuple[int, int]
    :return: An image with the contours drawn.
    :rtype: np.ndarray
    '''
    # Initialize the image
    img = 0.5 * np.ones(img_shape)
    # Iterate through the contour list
    for i in range(len(contours_list)):
        # Extract the layer number and determine if the pixel value for this contour should be 0 or 1
        layer_num = contours_list[i][0]
        if layer_num % 2 == 0:
            img_val = 0
        else: 
            img_val = 1
        # Extract the contour and set the pixels to the appropriate value
        contour = contours_list[i][1:]
        rows = contour[::2]
        cols = contour[1::2]
        for r, c, in zip(rows, cols):
            img[r, c] = img_val
    # Ensure that at least one pixel is black and at least one pixel is white, to maintain a consistent
    # visual when plotting
    if not np.any(img == 0):
        img[0, 0] = 0
    if not np.any(img == 1):
        img[-1, -1] = 1
    # Return the image
    return img
    
    