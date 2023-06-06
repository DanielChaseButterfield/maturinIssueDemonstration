import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import copy

#path_to_actag_detection_src = __file__[:__file__.find("/actag") + 1] + "actag/actag_detection/src/"
#sys.path.append(path_to_actag_detection_src)

from .sonar_parameters import SonarParams
from .tag_parameters import AcTagParams

def decode_tags(img: np.ndarray, 
                quads: list[np.ndarray], 
                sonar_params: SonarParams, 
                tag_params: AcTagParams,
                num_bit_corrections: int,
                ) -> list[list]:
    '''
    For each quad in the image, try to decode it as an AcTag. Return a list of all of the decoded tags found in the image.

    :param img: The sonar image.
    :type img: np.ndarray
    :param quads: A list of quadrilaterals found in the image.
    :type quads: list[np.ndarray]
    :param sonar_params: The parameters of the sonar that took the image.
    :type sonar_params: SonarParams
    :param tag_params: The parameters of the AcTag and its family.
    :type tag_params: AcTagParams
    :param num_bit_corrections: The number of bit corrections to perform when decoding the tag.
    :type num_bit_corrections: int
    :return: A list of all of the decoded tags found in the image. Each tag is a list, with the form ``[tag_id, [corner points in image in correct order], [range and azimuth of corner points in order]]``.
    :rtype: list[list]
    '''

    decoded_tags = []
    
    # Get a list of all of the range and azimuth values for the image
    ranges = np.linspace(sonar_params.max_range, sonar_params.min_range, img.shape[0])
    azimuths = np.linspace(sonar_params.max_azimuth, sonar_params.min_azimuth, img.shape[1])
    
    for quad in quads:
        # Find where the data bits for the tag should be in the sonar image
        data_bit_locations = get_data_bit_locations(quad, tag_params)

        # Skip to the next quad if any of the data bit locations are outside (or on the border) of the image
        if np.any(data_bit_locations < 0) or np.any(data_bit_locations >= img.shape):
            continue

        # Attempt to decode a tag at the location of the current quadrilateral
        tags_found = check_quad_for_tag(img, quad, data_bit_locations, tag_params, num_bit_corrections)

        if len(tags_found) > 0:
            for tag in tags_found:
                # Extract tag info
                tag_id = tag[0]
                ordered_corners = tag[1]

                # Get the range and azimuth value for each corner
                ordered_range_and_azimuth = np.zeros_like(ordered_corners, dtype=float)
                for i, corner in enumerate(ordered_corners):
                    ordered_range_and_azimuth[i] = [ranges[corner[0]], azimuths[corner[1]]]
                # Store the tag info
                decoded_tags.append([tag_id, ordered_corners, ordered_range_and_azimuth])
    # Return the list of any tags that were found in the image
    return decoded_tags

def get_data_bit_locations(quad: np.ndarray, tag_params: AcTagParams) -> np.ndarray:
    '''
    Get the tag data bit locations within the sonar image, by calculating the homography
    that maps the tag into the sonar image.
    
    :param quad: The quadrilateral to use as the central white square of the AcTag.
    :type quad: np.ndarray
    :param tag_params: The parameters of the AcTag and its family.
    :type tag_params: AcTagParams
    :return: The locations of the data bits in the sonar image.
    :rtype: np.ndarray
    '''
    # Given the tag parameters, determine the coordinates of the tag corners
    tag_pixels_across = (tag_params.tag_fam_data_bits + 4) // 4
    tag_corner_coordinates = np.array([[2, 2, tag_pixels_across - 2, tag_pixels_across - 2],
                                       [2, tag_pixels_across - 2, tag_pixels_across - 2, 2],
                                       [1, 1, 1, 1]])

    # Ensure the tag corner coordinates from the image are in the correct format
    sonar_image_coordinates = np.vstack((quad.T, np.ones((1, 4))))

    # Solve for the homography where sonar_coords = H @ tag_coords, so H = sonar_coords @ pinv(tag_coords)
    H = sonar_image_coordinates @ np.linalg.pinv(tag_corner_coordinates)

    # Round Homography to 8 decimal places to guarantee same results as Rust
    H = np.round(H, 10)

    # Create the array of data bit locations in the tag image
    data_bit_x_coords = np.hstack((0.5 * np.ones(tag_pixels_across),
                                   np.arange(1.5, tag_pixels_across-1, 1), 
                                   (tag_pixels_across - 0.5) * np.ones(tag_pixels_across),
                                   np.flip(np.arange(1.5, tag_pixels_across-1, 1))))
    data_bit_y_coords = np.hstack((np.arange(0.5, tag_pixels_across, 1),
                                   (tag_pixels_across - 0.5) * np.ones(tag_pixels_across-2),
                                   np.flip(np.arange(0.5, tag_pixels_across, 1)),
                                   (0.5) * np.ones(tag_pixels_across-2)))
    data_bit_tag_coords = np.vstack((data_bit_x_coords, data_bit_y_coords, np.ones(tag_params.tag_fam_data_bits)))

    # Map the data bit locations into the sonar image
    data_bit_sonar_img_vals = H @ data_bit_tag_coords

    # Round to nearest integer values and return
    data_bit_sonar_img_coords = np.round(data_bit_sonar_img_vals).astype(int)[:2, :].T

    # Return the data bit locations in the sonar image
    return data_bit_sonar_img_coords

# TODO: Clean up this function
def check_quad_for_tag(img: np.ndarray, 
                       quad_corners: np.ndarray,
                       data_bit_locations: np.ndarray, 
                       tag_params: AcTagParams, 
                       num_bit_corrections: int
                       ) -> list[list]:
    '''
    Determine bit values at each of the data bit locations, and if there is a match, return the corresponding tag.
    
    :param img: The sonar image to search for tags in.
    :type img: np.ndarray
    :param quad_corners: The corners of the quadrilateral.
    :type quad_corners: np.ndarray
    :param data_bit_locations: The locations of the data bits for the potential tag in the sonar image.
    :type data_bit_locations: np.ndarray
    :param tag_params: The parameters of the AcTag and its family.
    :type tag_params: AcTagParams
    :param num_bit_corrections: The number of bit corrections to allow when checking for a tag.
    :type num_bit_corrections: int
    :return: A list of any tags that were found in this quad. Each tag is a list, with the form ``[tag_id, [corner points in image in correct order], [range and azimuth of corner points in order]]``.
    :rtype: list[list]
    '''
    # Make clones of input values so we don't mutate them
    quad_corners = copy.deepcopy(quad_corners)
    data_bit_locations = copy.deepcopy(data_bit_locations)
    
    # Get the image shape
    rows, cols = img.shape

    # Find distances between each pair of data bits
    distances = []
    for i in range(0, len(data_bit_locations)):
        if i < (len(data_bit_locations) - 1):
            r_dist = data_bit_locations[i][0] - data_bit_locations[i+1][0]
            c_dist = data_bit_locations[i][1] - data_bit_locations[i+1][1] 
        else:
            r_dist = data_bit_locations[i][0] - data_bit_locations[0][0]
            c_dist = data_bit_locations[i][1] - data_bit_locations[0][1]
        distances.append(np.sqrt(r_dist**2 + c_dist**2))
    
    # Find radii of neighboorhood for each data bit
    radii = []
    for i in range(0, len(distances)):
        if i == 0:
            radii.append(int(np.floor((distances[len(distances) - 1] + distances[i]) / 8)))
        else:
            radii.append(int(np.floor((distances[i-1] + distances[i]) / 8)))
    
    # Calculate the average mean value of each pixel considered part of a data bit
    total_value = 0
    total_pixel_count = 0
    for i in range(0, len(data_bit_locations)):
        # For each pixel in the data bit neighborhood
        for r in range(data_bit_locations[i][0] - radii[i], data_bit_locations[i][0] + radii[i] + 1):
            for c in range(data_bit_locations[i][1] - radii[i], data_bit_locations[i][1] + radii[i] + 1):
                # If the pixel is in the image
                if r >= 0 and r < rows and c >= 0 and c < cols:
                    # Add the pixel value to the mean value
                    total_value += img[r, c]
                    total_pixel_count += 1
    average_mean = total_value / total_pixel_count

    # Calculate the average mean value of each data bit and determine if it is a 1 or a 0
    decoded_data_bits = []
    for i in range(0, len(data_bit_locations)):
        total_value_bit = 0
        total_pixel_count_bit = 0
        
        # For each pixel in the data bit neighborhood
        for r in range(data_bit_locations[i][0] - radii[i], data_bit_locations[i][0] + radii[i] + 1):
            for c in range(data_bit_locations[i][1] - radii[i], data_bit_locations[i][1] + radii[i] + 1):
                # If the pixel is in the image
                if r >= 0 and r < rows and c >= 0 and c < cols:
                    # Add the pixel value to the mean value
                    total_value_bit += img[r, c]
                    total_pixel_count_bit += 1
        
        # Get Average data bit mean
        average_mean_bit = total_value_bit / total_pixel_count_bit

        # Assign it a 1 or a 0 based on the average mean value
        if average_mean_bit > average_mean:
            decoded_data_bits.append(1)
        else:
            decoded_data_bits.append(0)

    # Compare decoded data bits to the tags in the family
    # TODO: This is written in such a way that it will handle if multiple tags are detected around the single quad passed in, which should be impossible due to the hamming distances and limitations there, so this can be simplified
    decoded_tags = []
    numInFam = len(tag_params.tags_in_family)/8
    numInFamRev = len(tag_params.tags_in_family)/2
    for i in range(0, len(tag_params.tags_in_family)):

        hamming_dist = 0
        for j in range(0, len(tag_params.tags_in_family[i])):
            hamming_dist += tag_params.tags_in_family[i][j] ^ decoded_data_bits[j]
        if(hamming_dist <= num_bit_corrections):
            # A decoded tag contains the following information -
            # [tag_id, [corner_points in proper order]]
            if i >= numInFamRev:
                actual_points = np.roll(quad_corners, -int(i / numInFam), axis=0)
                temp = actual_points[3].copy()
                actual_points[3] = actual_points[1]
                actual_points[1] = temp
            else:
                actual_points = np.roll(quad_corners, int(i / numInFam), axis=0)

            decoded_tags.append([int(i % numInFam), actual_points])

    # # Make the output image
    # for i in range(0, len(data_bit_locations)):
    #     filtered_img[data_bit_locations[i][0], data_bit_locations[i][1]] = decoded_data_bits[i] * 255

    return decoded_tags

def add_sample_locations_to_axis(ax: matplotlib.pyplot.Axes, 
                                 quads: list[np.ndarray], 
                                 tag_params: AcTagParams,
                                 img_dims: tuple[int, int]
                                 ) -> None:
    '''
    Adds the sampled data bit locations for each quad to the axis as green dots.

    :param ax: The axis to add the data bit locations to.
    :type ax: matplotlib.pyplot.Axes
    :param quads: The quads to add the data bit locations for.
    :type quads: list[np.ndarray]
    :param tag_params: The parameters of the AcTag and its family.
    :type tag_params: AcTagParams
    :param img_dims: The dimensions of the image the quads are from.
    :type img_dims: tuple[int, int]
    :return: None
    '''

    for quad in quads:
        # Find where the data bits for the tag should be in the sonar image
        data_bit_locations = get_data_bit_locations(quad, tag_params)
        if np.any(data_bit_locations < 0) or np.any(data_bit_locations >= img_dims):
            continue
        ax.scatter(data_bit_locations[:, 1], data_bit_locations[:, 0], s=1, c='g')
        
def add_detected_tags_to_axis(ax: matplotlib.pyplot.Axes,
                              detected_tags: list[list],
                              img_dims: tuple[int, int]
                              ) -> None:
    '''
    Visualizes detected tags onto a matplotlib.pyplot.Axes.

    :param ax: The axis to add the detected tags to.
    :type ax: matplotlib.pyplot.Axes
    :param detected_tags: The detected tags to visualize.
    :type detected_tags: list[list]
    :param img_dims: The dimensions of the image the tags are from.
    :type img_dims: tuple[int, int]
    :return: None
    '''

    # For each detected tag
    for tag in detected_tags:
        # Draw lines between each pair of points
        for i in range(0, 4):
            pt1 = tag[1][i]
            pt2 = tag[1][(i+1)%4]
            x_vals = [pt1[0], pt2[0]]
            y_vals = [pt1[1], pt2[1]]
            ax.add_line(Line2D(y_vals, x_vals, color="lime"))

            # TODO: Add text with the tag id and the tag corners