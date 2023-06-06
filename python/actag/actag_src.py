import numpy as np
import matplotlib.pyplot as plt
import sys

#path_to_actag_detection_src = __file__[:__file__.find("/actag.py")+1]
#sys.path.append(path_to_actag_detection_src)

from .sonar_parameters import SonarParams
from .tag_parameters import AcTagParams
from .median_filter import median_filter_multiprocessed
from .adaptive_threshold import adaptive_threshold_multiprocessed
from .contour_identification import get_contours, convert_contour_list_to_img
from .quad_fitting import fit_quadrilaterals_to_contours, add_quads_to_axis
from .tag_decoding import decode_tags, add_sample_locations_to_axis, add_detected_tags_to_axis

import actag as actag_rust

class AcTag:
    '''
    Class for detecting AcTags in sonar images.
    
    :param min_range: The minimum range setting on the sonar used to capture this image.
    :type min_range: float
    :param max_range: The maximum range setting on the sonar used to capture this image.
    :type max_range: float
    :param horizontal_aperature: The horizontal aperature setting (in radians) on the sonar used to capture this image.
    :type horizontal_aperature: float
    :param vertical_aperature: The vertical aperature setting (in radians) on the sonar used to capture this image.
    :type vertical_aperature: float
    :param tag_family: The name of a AcTag family found in the ``actag_families/generated_families`` folder, or the path to a valid AcTag family file. See the `actag_families <https://bitbucket.org/frostlab/actag/src/master/actag_families/>`_ module for more information.
    :type tag_family: str
    :param tag_size: The length of the inner square of white data bits in meters.
    :type tag_size: float
    :param median_filter_kernel_radius: The neighborhood radius for the median filter.
    :type median_filter_kernel_radius: int
    :param adaptive_threshold_kernel_radius: The neighborhood radius for the adaptive threshold.
    :type adaptive_threshold_kernel_radius: int
    :param adaptive_threshold_offset: The offset value for the adaptive threshold.
    :type adaptive_threshold_offset: float
    :param contours_tag_size_tolerance: The error tolerance for contours bigger than the tag_size. Contours with an x or y length longer than ``tag_size * (1 + contours_tag_size_tolerance)`` will be rejected.
    :type contours_tag_size_tolerance: float
    :param contours_tag_area_tolerance: The error tolerance for contours outside the range of expected tag areas. Contours with an area outside of the tolerances will be rejected.
    :type contours_tag_area_tolerance: float
    :param contours_min_tag_area_ratio: The minimum expected tag area, as ``max_expected_area * contours_min_tag_area_ratio``.
    :type contours_min_tag_area_ratio: float
    :param contours_reject_black_shapes: Determines whether the contour rejection algorithm should reject edges that contours that go from white to black. When true, this will reject contours that can't be the inner quadrilateral of an AcTag.
    :type contours_reject_black_shapes: bool
    :param contours_reject_white_shapes: Determines whether the contour rejection algorithm should reject edges that contours that go from black to white. When false, this won't reject contours that can be the inner quadrilateral of an AcTag.
    :type contours_reject_white_shapes: bool
    :param contours_reject_by_tag_size: If true, will reject contours outside of the tag_size error tolerances.
    :type contours_reject_by_tag_size: bool
    :param contours_reject_by_area: If true, will reject contours outside of the tag area error tolerances.
    :type contours_reject_by_area: bool
    :param quads_random_seed: If specified, the discrete uniform random number generator will be seeded with this value. If not specified, the random number generator will have a random seed, leading to new values each time it is run.
    :type quads_random_seed: int or None
    :param quads_points_per_line_ratio: The ratio of the number of points within a contour to use for each line fit.
    :type quads_points_per_line_ratio: float
    :param quads_dist_for_inlier: The distance threshold for a point to be considered an inlier.
    :type quads_dist_for_inlier: float
    :param quads_desired_inlier_ratio: The desired ratio of inliers within the total number of points.
    :type quads_desired_inlier_ratio: float
    :param quads_required_inlier_ratio: The required ratio of inliers within the total number of points to successfully fit a quadrilateral.
    :type quads_required_inlier_ratio: float
    :param quads_parallel_threshold: Value used for determining whether or not two lines are parallel, when checking if the quad is a parallelogram.
    :type quads_parallel_threshold: float
    :param quads_use_same_random_vals: Set to true to use repeatable random values, starting with the ``starting_roll_val``. Used to syncronize random values between Python & Rust versions of the code (for testing purposes).
    :type quads_use_same_random_vals: bool
    :param quads_starting_roll_val: The starting roll value used for when ``quads_use_same_random_vals`` is set to True.
    :type quads_starting_roll_val: int
    :param decoding_num_bit_corrections: The number of bit corrections allowed during the coding process. THis value must be greater than 0 and less than half of the minimum hamming distance of the AcTag family.
    :type decoding_num_bit_corrections: int
    :param use_optimized_rust_code: If true, will use the optimized Rust code, which runs significantly faster. If false, will use the Python code.
    :type use_optimized_rust_code: bool

    :raises ValueError: Raised if ``decoding_num_bit_corrections`` is less than 0 or greater than/equal to half of the hamming distance of the Actag family.
    '''

    def __init__(self, 
                 min_range: float,
                 max_range: float,
                 horizontal_aperture: float,
                 tag_family: str,
                 tag_size: float,
                 median_filter_kernel_radius: int,
                 adaptive_threshold_kernel_radius: int,
                 adaptive_threshold_offset: float,
                 contours_tag_size_tolerance: float=0.2,
                 contours_tag_area_tolerance: float=0.2,
                 contours_min_tag_area_ratio: float=0.1,
                 contours_reject_black_shapes: bool=True, 
                 contours_reject_white_shapes: bool=False, 
                 contours_reject_by_tag_size: bool=True,
                 contours_reject_by_area: bool=True,
                 quads_random_seed: int=None,
                 quads_points_per_line_ratio: float=0.1,
                 quads_dist_for_inlier: float=2.5,
                 quads_desired_inlier_ratio: float=0.85,
                 quads_required_inlier_ratio: float=0.8,
                 quads_parallel_threshold: float = 0.9,
                 quads_use_same_random_vals: bool=False,
                 quads_starting_roll_val: int=123456,
                 decoding_num_bit_corrections: int=4,
                 use_optimized_rust_code: bool=True
                 ) -> None:
        self.sonar_parameters = SonarParams(min_range, max_range, horizontal_aperture)
        self.actag_parameters = AcTagParams(tag_family, tag_size)
        self.median_filter_kernel_radius = median_filter_kernel_radius
        self.adaptive_threshold_kernel_radius = adaptive_threshold_kernel_radius
        self.adaptive_threshold_offset = adaptive_threshold_offset
        self.contours_tag_size_tolerance = contours_tag_size_tolerance
        self.contours_tag_area_tolerance = contours_tag_area_tolerance
        self.contours_min_tag_area_ratio = contours_min_tag_area_ratio
        self.contours_reject_black_shapes = contours_reject_black_shapes
        self.contours_reject_white_shapes = contours_reject_white_shapes
        self.contours_reject_by_tag_size = contours_reject_by_tag_size
        self.contours_reject_by_area = contours_reject_by_area
        self.quads_random_seed = quads_random_seed
        self.quads_points_per_line_ratio = quads_points_per_line_ratio
        self.quads_dist_for_inlier = quads_dist_for_inlier
        self.quads_desired_inlier_ratio = quads_desired_inlier_ratio
        self.quads_required_inlier_ratio = quads_required_inlier_ratio
        self.quads_parallel_threshold = quads_parallel_threshold
        self.quads_use_same_random_vals = quads_use_same_random_vals
        self.quads_starting_roll_val = quads_starting_roll_val
        self.decoding_num_bit_corrections = decoding_num_bit_corrections
        if self.decoding_num_bit_corrections < 0:
            raise ValueError('decoding_num_bit_corrections must be >= 0')
        if self.decoding_num_bit_corrections >= self.actag_parameters.tag_fam_hamming_dist / 2:
            raise ValueError('decoding_num_bit_corrections must be between 0 and '
                             f'{np.round(self.actag_parameters.tag_fam_hamming_dist / 2 - 1).astype(int)} '
                             f'with tag family {self.actag_parameters.tag_family}')
        self.use_optimized_rust_code = use_optimized_rust_code


    def run_detection_python(self, sonar_img: np.ndarray) -> list[list[int, np.ndarray, np.ndarray]]:
        '''
        Runs the full detection algorithm on the given sonar image with Python functions. 
        Slower than :func:`run_detection_rust`.
        
        :param sonar_img: The sonar image to run the detection algorithm on.
        :type sonar_img: np.ndarray
        :return: The list of detected tags.
        :rtype: list[list[int, np.ndarray, np.ndarray]]

        :meta private:
        '''

        filtered_img = median_filter_multiprocessed(sonar_img, 
                                            self.median_filter_kernel_radius)
        binarized_img = adaptive_threshold_multiprocessed(filtered_img, 
                                            self.adaptive_threshold_kernel_radius, 
                                            self.adaptive_threshold_offset)
        contour_list = get_contours(binarized_img, 
                                    self.sonar_parameters,
                                    self.actag_parameters,
                                    self.contours_tag_size_tolerance,
                                    self.contours_tag_area_tolerance,
                                    self.contours_min_tag_area_ratio,
                                    self.contours_reject_black_shapes, 
                                    self.contours_reject_white_shapes, 
                                    self.contours_reject_by_tag_size, 
                                    self.contours_reject_by_area)
        quad_list = fit_quadrilaterals_to_contours(contour_list, 
                                    sonar_img.shape, 
                                    self.quads_random_seed,
                                    self.quads_points_per_line_ratio,
                                    self.quads_dist_for_inlier,
                                    self.quads_desired_inlier_ratio,
                                    self.quads_required_inlier_ratio,
                                    self.quads_parallel_threshold,
                                    self.quads_use_same_random_vals,
                                    self.quads_starting_roll_val)
        detected_tags = decode_tags(filtered_img, 
                                    quad_list, 
                                    self.sonar_parameters, 
                                    self.actag_parameters,
                                    self.decoding_num_bit_corrections)
        return detected_tags

    def run_detection_rust(self, sonar_img: np.ndarray) -> list[list[int, np.ndarray, np.ndarray]]:
        '''
        Runs the full detection algorithm on the given sonar image with the Rust functions.
        
        :param sonar_img: The sonar image to run the detection algorithm on.
        :type sonar_img: np.ndarray
        :return: The list of detected tags.
        :rtype: list[list[int, np.ndarray, np.ndarray]]

        :meta private:
        '''
        filtered_img = actag_rust.median_filter_multithread(sonar_img, 
                                            self.median_filter_kernel_radius, None)
        binarized_img = actag_rust.adaptive_threshold_multithread(filtered_img, 
                                            self.adaptive_threshold_kernel_radius, 
                                            self.adaptive_threshold_offset, None)
        contour_list = actag_rust.get_contours(binarized_img, 
                                    self.sonar_parameters.min_range,
                                    self.sonar_parameters.max_range,
                                    self.sonar_parameters.horizontal_aperture,
                                    self.actag_parameters.tag_size,
                                    self.contours_tag_size_tolerance,
                                    self.contours_tag_area_tolerance,
                                    self.contours_min_tag_area_ratio,
                                    self.contours_reject_black_shapes, 
                                    self.contours_reject_white_shapes, 
                                    self.contours_reject_by_tag_size, 
                                    self.contours_reject_by_area)
        quad_list = actag_rust.fit_quadrilaterals_to_contours(contour_list, 
                                    sonar_img.shape, 
                                    self.quads_random_seed,
                                    self.quads_points_per_line_ratio,
                                    self.quads_dist_for_inlier,
                                    self.quads_desired_inlier_ratio,
                                    self.quads_required_inlier_ratio,
                                    self.quads_parallel_threshold,
                                    self.quads_use_same_random_vals,
                                    self.quads_starting_roll_val)
        detected_tags = actag_rust.decode_tags(filtered_img, 
                                    quad_list, 
                                    self.sonar_parameters.min_range,
                                    self.sonar_parameters.max_range,
                                    self.sonar_parameters.horizontal_aperture, 
                                    self.actag_parameters.tag_family,
                                    self.decoding_num_bit_corrections)
        
        # Format the result like Python's version
        detected_tags_rearranged = []
        for tag in detected_tags:
            tag_rearranged = []
            
            # Add the tag id
            tag_rearranged.append(int(tag[0]))

            # Add the tag corners
            tag_corners = np.zeros((4, 2), dtype=int)
            for i in range(1, 9, 2):
                tag_corners[(i-1)//2,0] = int(tag[i])
                tag_corners[(i-1)//2,1] = int(tag[i+1])
            tag_rearranged.append(tag_corners)

            # Add the tag range/azi values
            tag_azi_range_values = np.zeros((4, 2), dtype=float)
            for i in range(9, 17, 2):
                tag_azi_range_values[(i-9)//2,0] = tag[i]
                tag_azi_range_values[(i-9)//2,1] = tag[i+1]
            tag_rearranged.append(tag_azi_range_values)

            # Add the tag to the list of detected tags
            detected_tags_rearranged.append(tag_rearranged)

        return detected_tags_rearranged
        
    def run_detection(self, sonar_img: np.ndarray) -> list[list[int, np.ndarray, np.ndarray]]:
        '''
        Runs the full detection algorithm on the given sonar image. 

        Steps:
            1. Filter the image with a Median filter.
            2. Binarize the image with an adaptive threshold.
            3. Find contours in the image.
            4. Fit quadrilaterals to the contours.
            5. Decode the tags.
            
        :param sonar_img: The sonar image. The first row, last row, first column, and last column of the image should correspond with the maximum range, minimum range, maximum azimuth, and minimum azimuth of the sonar, respectively. 
        :type sonar_img: np.ndarray
        :return: The list of detected tags. Each detected tag is a list of the form ```[tag_id, corner_indices, range_and_azimuth_values]```. ```tag_id``` is an integer value indicating the tag ID in the specified tag family. ```corner_indices``` is a 2D numpy array containing ```[row, column]``` pairs for each of the four corners of the tag. The top-left tag corner is first, followed by the top-right, and around in a clockwise fashion. Note that "top-left" refers to the corner in the top-left of the AcTag's image file, not the corner which happens to appear in the top-left of the current sonar image. ```range_and_azimuth_values``` is a 2D numpy array with ```[range, azimuth]``` pairs (in meters and radians, respectively) corresponding to each of the four corners, in the same order as ```corner_indices```.
        :rtype: list[list[int, np.ndarray, np.ndarray]]
        '''
        if(self.use_optimized_rust_code):
            return self.run_detection_rust(sonar_img)
        else:
            return self.run_detection_python(sonar_img)
    
    def visualize_detection(self, sonar_img: np.ndarray) -> list[list[int, np.ndarray, np.ndarray]]:
        '''
        Same as :func:`run_detection`, but also visualizes each step of the process using matplotlib.

        :param sonar_img: The sonar image. The first row, last row, first column, and last column of the image should correspond with the maximum range, minimum range, maximum azimuth, and minimum azimuth of the sonar, respectively.
        :type sonar_img: np.ndarray
        :return: The list of detected tags. Each detected tag is a list of the form ```[tag_id, corner_indices, range_and_azimuth_values]```. ```tag_id``` is an integer value indicating the tag ID in the specified tag family. ```corner_indices``` is a 2D numpy array containing ```[row, column]``` pairs for each of the four corners of the tag. The top-left tag corner is first, followed by the top-right, and around in a clockwise fashion. Note that "top-left" refers to the corner in the top-left of the AcTag's image file, not the corner which happens to appear in the top-left of the current sonar image. ```range_and_azimuth_values``` is a 2D numpy array with ```[range, azimuth]``` pairs (in meters and radians, respectively) corresponding to each of the four corners, in the same order as ```corner_indices```.
        :rtype: list[list[int, np.ndarray, np.ndarray]]
        '''
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 6))
        ax1.imshow(sonar_img, cmap='gray')
        ax1.set_title("Input Image")
        filtered_img = median_filter_multiprocessed(sonar_img, 
                                            self.median_filter_kernel_radius)
        ax2.imshow(filtered_img, cmap='gray')
        ax2.set_title("Filtered Image")
        binarized_img = adaptive_threshold_multiprocessed(filtered_img, 
                                            self.adaptive_threshold_kernel_radius, 
                                            self.adaptive_threshold_offset)
        ax3.imshow(binarized_img, cmap='gray')
        ax3.set_title("Binarized Image")
        contour_list = get_contours(binarized_img, 
                                    self.sonar_parameters,
                                    self.actag_parameters,
                                    self.contours_tag_size_tolerance,
                                    self.contours_tag_area_tolerance,
                                    self.contours_min_tag_area_ratio,
                                    self.contours_reject_black_shapes, 
                                    self.contours_reject_white_shapes, 
                                    self.contours_reject_by_tag_size, 
                                    self.contours_reject_by_area)
        ax4.imshow(convert_contour_list_to_img(contour_list, sonar_img.shape), cmap='gray')
        ax4.set_title("Remaining Contours")
        quad_list = fit_quadrilaterals_to_contours(contour_list, 
                                    sonar_img.shape, 
                                    self.quads_random_seed,
                                    self.quads_points_per_line_ratio,
                                    self.quads_dist_for_inlier,
                                    self.quads_desired_inlier_ratio,
                                    self.quads_required_inlier_ratio,
                                    self.quads_parallel_threshold,
                                    self.quads_use_same_random_vals,
                                    self.quads_starting_roll_val)
        ax5.imshow(filtered_img, cmap='gray')
        ax5.set_title("Detected Quads")
        add_quads_to_axis(ax5, quad_list)
        add_sample_locations_to_axis(ax5, quad_list, self.actag_parameters, sonar_img.shape)
        detected_tags = decode_tags(filtered_img, 
                                    quad_list, 
                                    self.sonar_parameters, 
                                    self.actag_parameters,
                                    self.decoding_num_bit_corrections)
        ax6.imshow(sonar_img, cmap='gray')
        ax6.set_title("Tag Detections")
        add_detected_tags_to_axis(ax6, detected_tags, sonar_img.shape)
        plt.show()

        return detected_tags
        