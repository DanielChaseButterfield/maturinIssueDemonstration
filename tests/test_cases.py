import unittest
import sys
import cv2
import numpy as np
import copy

path_to_actag_detection = __file__[:__file__.find("/tests")+1]
sys.path.append(path_to_actag_detection)

# Import Python library
from src.actag import AcTag
from src.tag_parameters import AcTagParams
from src.sonar_parameters import SonarParams
from src.median_filter import median_filter, median_filter_multiprocessed
from src.adaptive_threshold import adaptive_threshold, adaptive_threshold_multiprocessed
from src.contour_identification import get_contours
from src.quad_fitting import fit_quadrilaterals_to_contours, get_random_point_and_fit_line, least_squares_line_fit, get_intersection_of_lines
from src.tag_decoding import decode_tags, check_quad_for_tag,  get_data_bit_locations

# Import Rust library
import actag_optimized_functions as actag_rust

class TestAcTagAlgorithm(unittest.TestCase):

    ''' ========== TEST IMAGES ========== '''

    testImg1 = [[31, 255], [2, 255]]
    testImg2 = [[1, 2, 3, 4], [2, 3, 4, 5]]
    testImg3 = [[1, 2, 24, 3, 4], [2, 3, 111, 4, 5], [81, 80, 121, 66, 66]]
    realImg = cv2.imread(path_to_actag_detection + "example_scripts/exampleSonarImage.png", cv2.IMREAD_GRAYSCALE)

    ''' ========== HELPER FUNCTIONS ========== '''

    # This helper function can be used to ensure that arrays
    # are identical
    def helper_assert_arrays_equal(self, arr1, arr2):
        # Make sure they have the same number of rows and columns
        self.assertEqual(len(arr1), len(arr2))
        self.assertEqual(len(arr1[0]), len(arr2[0]))

        # Make sure each element in the arrays match
        for i in range(0, len(arr1)):
            for j in range(0, len(arr1[0])):
                self.assertEqual(arr1[i][j], arr2[i][j])

    # This helper function can be used to ensure that two lists
    # of list are identical
    def helper_assert_list_of_lists_equal(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for i in range(len(list1)):
            # Make sure each contour is identical
            sublist1 = list1[i]
            sublist2 = list2[i]
            
            self.assertEqual(len(sublist1), len(sublist2))
            for j in range(len(sublist1)):
                self.assertEqual(sublist1[j], sublist2[j])
    
    # This helper function ensure that two sets of quadrilaterals
    # are equal
    def helper_assert_quads_equal(self, quads1, quads2):
        self.assertEqual(len(quads1), len(quads2))
        for i in range(len(quads1)):
            # Make sure each quad is identical
            quad1 = quads1[i]
            quad2 = quads2[i]
            for j in range(0, 4):
                self.assertEqual(quad1[j][0], quad2[j][0])
                self.assertEqual(quad1[j][1], quad2[j][1])

    # This helper function asserts that the median filter output
    # of all four actag implementations are identical
    def helper_assert_four_median_filters_identical(self, img):
        # Run the python base median filter function
        python_results = median_filter(img, 4).tolist()

        # Make sure it matches the other three implementations
        self.assertSequenceEqual(python_results, median_filter_multiprocessed(img, 4, None).tolist())
        self.helper_assert_arrays_equal(python_results, actag_rust.median_filter(img, 4))
        self.helper_assert_arrays_equal(python_results, actag_rust.median_filter_multithread(img, 4, None))

    # This helper function makes sure that all four of the adaptive
    # threshold implementations perform the same
    def helper_assert_four_adaptive_thresholds_identical(self, img):
        # Run the python base median filter function
        python_results = adaptive_threshold(img, 8, 1).tolist()

        # Make sure it matches the other three implementations
        self.assertSequenceEqual(python_results, adaptive_threshold_multiprocessed(img, 8, 1, None).tolist())
        self.helper_assert_arrays_equal(python_results, actag_rust.adaptive_threshold(img, 8, 1))
        self.helper_assert_arrays_equal(python_results, actag_rust.adaptive_threshold_multithread(img, 8, 1, None))

    # This helper functions makes sure that the contours extracted 
    # by two different implementations are identical for a provided image.
    def helper_assert_two_contour_functions_identical(self, img, reject):
        # Run the python base get_contours() function
        sonar_params = SonarParams(0.1, 5, 1.0472)
        tag_params = AcTagParams("AcTag24h10", 0.130628571428644)
        python_results = get_contours(img, sonar_params, tag_params, 0.2, 0.2, 0.1, True, False, reject, reject)

        # Get the Rust results
        rust_results = actag_rust.get_contours(img, 0.1, 5.0, 1.0472, 0.130628571428644, 0.2, 0.2, 0.1, True, False, reject, reject)
        
        # Make sure that they're identical
        self.helper_assert_list_of_lists_equal(python_results, rust_results)

    # This helper function is used to make sure that parsing a bad
    # family name returns the same error for both Python and Rust
    def helper_family_name_assert_raise(self, family_name, error):
        with self.assertRaises(error) as cm_python:
            tag_params = AcTagParams(family_name, 0.5)
        with self.assertRaises(error) as cm_rust:
            rust_tags = actag_rust.parse_apriltag_family(family_name)

        self.assertMultiLineEqual(str(cm_python.exception), str(cm_rust.exception))

    # This helper functions assures that the decoded tags match
    def helper_assert_decoded_tags_equal(self, python_results, rust_results):
        # Make sure the same num of tags were detected
        self.assertEqual(len(python_results), len(rust_results))

        # Make sure each tag matches
        for i in range(len(python_results)):
            python_tag = python_results[i]
            rust_tag = rust_results[i]
            
            # Make sure the id matches
            self.assertEqual(python_tag[0], rust_tag[0])

            # Make sure the corner locations match
            python_corners = python_tag[1]
            for j in range(1, 9):
                self.assertEqual(python_corners[int((j-1)/2)][(j-1)%2], rust_tag[j])
            
            # Make sure the range and azimuth locations match
            python_range_azi = python_tag[2]
            for j in range(9, 17):
                self.assertAlmostEqual(python_range_azi[int((j-9)/2)][(j-9)%2], rust_tag[j], places=15)
            

    # This helper functions assures that the final results of calling the AcTag
    # algorithm are the same
    def helper_assert_final_results_equal(self, python_results, rust_results):
        # Make sure the same num of tags were detected
        self.assertEqual(len(python_results), len(rust_results))

        # Make sure each tag matches
        for i in range(len(python_results)):
            python_tag = python_results[i]
            rust_tag = rust_results[i]
            
            # Make sure the id matches
            self.assertEqual(python_tag[0], rust_tag[0])
    
            # Make sure the corner locations match
            python_corners = python_tag[1]
            rust_corners = rust_tag[1]
            for j in range(1, 9):
                self.assertEqual(python_corners[int((j-1)/2)][(j-1)%2], rust_corners[int((j-1)/2)][(j-1)%2])
            
            # Make sure the range and azimuth locations match
            python_range_azi = python_tag[2]
            rust_range_azi = rust_tag[2]
            for j in range(9, 17):
                self.assertAlmostEqual(python_range_azi[int((j-9)/2)][(j-9)%2], rust_range_azi[int((j-9)/2)][(j-9)%2], places=15)

    ''' ========== TEST FUNCTIONS ========== '''

    # This test function ensures that the median filter implementations
    # match, including the multithreaded functions
    def test_median_filters(self):
        self.helper_assert_four_median_filters_identical(self.testImg1)
        self.helper_assert_four_median_filters_identical(self.testImg2)
        self.helper_assert_four_median_filters_identical(self.testImg3)
        self.helper_assert_four_median_filters_identical(self.realImg)

    # This test function ensures that the adaptive threshold implementaitons
    # match for both the Python and Rust sides
    def test_adaptive_thresholds(self):
        self.helper_assert_four_adaptive_thresholds_identical(self.testImg1)
        self.helper_assert_four_adaptive_thresholds_identical(self.testImg2)
        self.helper_assert_four_adaptive_thresholds_identical(self.testImg3)
        self.helper_assert_four_adaptive_thresholds_identical(self.realImg)

    # This test function ensures that the contour identification is identical
    # for Rust and Python
    def test_get_contours(self):
        # Get the thresholded image
        realImgFilt = actag_rust.median_filter_multithread(self.realImg, 4, None)
        realImgThreshold = actag_rust.adaptive_threshold_multithread(realImgFilt, 8, 1)

        # Assert equal functionality
        self.helper_assert_two_contour_functions_identical(realImgThreshold, True)
        self.helper_assert_two_contour_functions_identical(realImgThreshold, False)

    # This test function ensures that the intersection of line functionality
    # is identical for both Python and Rust
    def test_get_intersection_of_lines(self):
        # Open the file with the test data
        with open(path_to_actag_detection + "tests/test_data/get_intersection_of_lines_inputs.txt", "r") as f:

            # Run a test for each line in the file
            linePoints = f.readline()
            while linePoints != "":
                # Parse the data
                lines = []

                index = linePoints.find(" ")
                while index != -1:
                    first_num = float(linePoints[:index])
                    linePoints = linePoints[index + 1:]
                    index = linePoints.find(" ")
                    second_num = float(linePoints[:index])
                    linePoints = linePoints[index + 1:]
                    index = linePoints.find(" ")
                    third_num = float(linePoints[:index])
                    linePoints = linePoints[index + 1:]
                    lines.append([first_num, second_num, third_num])
                    index = linePoints.find(" ")

                # Convert the 1.0 and 0.0 to 'row' and 'column', respectively,
                # for the python code
                linesPython = copy.deepcopy(lines)
                for line in linesPython:
                    if line[0] == 0.0:
                        line[0] = 'column'
                    else:
                        line[0] = 'row'

                # Iterate through each combination of lines
                for i in range(len(lines) - 1):
                    for j in range(i + 1, len(lines)):
                        try:
                            # Get the Python results
                            python_results = get_intersection_of_lines(linesPython[i], linesPython[j])

                            # Get the Rust results
                            rust_results = actag_rust.get_intersection_of_lines(
                                (lines[i][1], lines[i][2]), (lines[j][1], lines[j][2]),
                                lines[i][0] == 1.0, lines[j][0] == 1.0)

                            # Make sure that they're identical
                            self.assertEqual(rust_results[0], python_results[0])
                            self.assertEqual(rust_results[1], python_results[1])

                        except ValueError:  # If one throws and error, make sure they both do
                            with self.assertRaises(ValueError) as cm_python:
                                python_results = get_intersection_of_lines(linesPython[i], linesPython[j])
                            with self.assertRaises(ValueError) as cm_rust:
                                rust_results = actag_rust.get_intersection_of_lines(
                                    (lines[i][1], lines[i][2]), (lines[j][1], lines[j][2]),
                                    lines[i][0] == 1.0, lines[j][0] == 1.0)

                            self.assertMultiLineEqual(str(cm_python.exception), str(cm_rust.exception))


                # See if there is more in the file
                linePoints = f.readline()

    # This test function ensures that the least squares line fitting
    # functions are identical
    def test_least_squares_line_fit(self):
        # Open the file with the test data
        with open(path_to_actag_detection + "tests/test_data/least_squares_line_fit_inputs.txt", "r") as f:

            # Run a test for each line in the file
            linePoints = f.readline()
            while linePoints != "":
                # Parse the data
                line_points = []

                index = linePoints.find(" ")
                while index != -1:
                    first_num = int(linePoints[:index])
                    linePoints = linePoints[index + 1:]
                    index = linePoints.find(" ")
                    second_num = int(linePoints[:index])
                    linePoints = linePoints[index + 1:]
                    line_points.append((first_num, second_num))
                    index = linePoints.find(" ")

                # Get the Python and Rust results
                python_results = least_squares_line_fit(np.array(line_points))
                rust_results = actag_rust.least_squares_line_fit_python_wrap(line_points)

                # Convert the python 'row' and 'column' to 1.0 and 0.0, respectively
                if python_results[0] == 'column':
                    python_results = [0.0, python_results[1], python_results[2]]
                else:
                    python_results = [1.0, python_results[1], python_results[2]]

                # Make sure they ALMOST match, to 12 decimal places
                self.assertEqual(python_results[0], rust_results[0])
                self.assertAlmostEqual(python_results[1], rust_results[1], places=12)
                self.assertAlmostEqual(python_results[2], rust_results[2], places=12)
                
                # See if there is more in the file
                linePoints = f.readline()

    # This test function ensures that both get_random_point_and_fit_line
    # functions perform identically
    def test_get_random_point_and_fit_line(self):
        # Make the random number generator
        rand = np.random.default_rng(123456)

        # Open the file with the test data
        with open(path_to_actag_detection + "tests/test_data/get_random_point_and_fit_line_inputs.txt", "r") as f:

            # Run a test for each pair of lines in the file
            tempPointsLine = f.readline()
            while tempPointsLine != "":
                currRollValLine = f.readline()

                # Parse the data
                curr_roll_val = int(currRollValLine)
                temp_nums = []

                index = tempPointsLine.find(" ")
                while index != -1:
                    temp_nums.append(int(float(tempPointsLine[:index])))
                    tempPointsLine = tempPointsLine[index + 1:]
                    index = tempPointsLine.find(" ")
                temp_nums.append(int(float(tempPointsLine)))

                shape_points = []
                for i in range(0, len(temp_nums) - 1, 2):
                    shape_points.append((temp_nums[i], temp_nums[i + 1]))

                # Get the Python and Rust results
                python_results = get_random_point_and_fit_line(np.array(shape_points), 4, True, rand, curr_roll_val)
                rust_results = actag_rust.get_random_point_and_fit_line_python_wrap(shape_points, 4, True, curr_roll_val)

                # Convert the python 'row' and 'column' to 1.0 and 0.0, respectively
                if python_results[0] == 'column':
                    python_results[0] = 0.0
                else:
                    python_results[0] = 1.0

                # Put the leftover points for Rust into a list
                leftover_points = []
                for i in range(3, len(rust_results) - 1, 2):
                    leftover_points.append([int(rust_results[i]), int(rust_results[i + 1])])
                rust_results[3] = leftover_points
                rust_results = rust_results[:4]

                # Convert the leftover point array in Python to a list
                python_results[3] = python_results[3].tolist()

                # Make sure they ALMOST match, to 7 decimal places
                self.assertEqual(python_results[0], rust_results[0])
                self.assertAlmostEqual(python_results[1], rust_results[1], places=7)
                self.assertAlmostEqual(python_results[2], rust_results[2], places=7)
                self.assertSequenceEqual(python_results[3], rust_results[3])
                
                # See if there is more in the file
                tempPointsLine = f.readline()

    # This test function ensures that the quadrilateral fitting is identical
    # TODO: Needs to be more extensive, as it is currently only testing one image
    def test_fit_quadrilaterals_to_contours(self):
        # Get contours from the real sonar image
        realImgFilt = actag_rust.median_filter_multithread(self.realImg, 4, None)
        realImgThreshold = actag_rust.adaptive_threshold_multithread(realImgFilt, 8, 1)
        realContours = actag_rust.get_contours(realImgThreshold, 0.1, 5.0, 1.0472, 0.130628571428644, 0.2, 0.2, 0.1, True, False, True, True)

        # Get the quadrilaterals from the python and rust functions
        python_results = fit_quadrilaterals_to_contours(realContours, (len(realImgThreshold), len(realImgThreshold[0])), 
                                           None, 0.1, 2.5, 0.85, 0.8, 0.9, True, 123456)
        rust_results = actag_rust.fit_quadrilaterals_to_contours(realContours, (len(realImgThreshold), len(realImgThreshold[0])), 
                                           None, 0.1, 2.5, 0.85, 0.8, 0.9, True, 123456)

        # Make sure they match
        self.helper_assert_quads_equal(python_results, rust_results)
        
    # This test function ensures the AcTag family parsing matches,
    # including the error messages
    def test_get_tags_in_family(self):
        # ================ TEST 1 ================
        # Ensure same output for parsing the AcTag24h10 family
        family_name = "AcTag24h10"

        tag_params = AcTagParams(family_name, 0.5)
        python_tags = tag_params.tags_in_family
        rust_tags = actag_rust.parse_apriltag_family(family_name)

        self.assertEqual(python_tags, rust_tags)   

        # ================ TEST 2 ================     
        # Ensure same output for when family_name is invalid
        self.helper_family_name_assert_raise("AcTagNotReal", ValueError)
        self.helper_family_name_assert_raise("AcTag25h10", ValueError)
        self.helper_family_name_assert_raise("AcTag24h100", FileNotFoundError)
        self.helper_family_name_assert_raise("FakeTag24h10", ValueError)
        self.helper_family_name_assert_raise("AcTagh10", ValueError)
        self.helper_family_name_assert_raise("AcTag24h", ValueError)

    # Ensure that the data bit location functionality is identical between
    # Python and Rust
    def test_get_data_bit_locations(self):
        # Get the filtered image
        realImgFilt = actag_rust.median_filter_multithread(self.realImg, 4, None)

        # Create the AcTag Params object
        tag_params = AcTagParams("AcTag24h10", 0.130628571428644)

        # Open the file with the test data
        with open(path_to_actag_detection + "tests/test_data/get_data_bit_locations_inputs.txt", "r") as f:
            # Run a test for each line in the file
            quadLine = f.readline()
            while quadLine != "":

                # Parse the quad
                temp_nums = []
                index = quadLine.find(" ")
                while index != -1:
                    temp_nums.append(int(quadLine[:index]))
                    quadLine = quadLine[index + 1:]
                    index = quadLine.find(" ")
                quad = []
                for i in range(0, len(temp_nums) - 1, 2):
                    quad.append((temp_nums[i], temp_nums[i + 1]))

                # Get the Python and Rust results
                python_results = get_data_bit_locations(np.array(quad), tag_params)
                rust_results = actag_rust.get_data_bit_locations(quad, 24)

                # Make sure they match
                self.assertEqual(len(python_results), len(rust_results))
                for i in range(len(python_results)):
                    # Make sure each point matches
                    python_point = python_results[i]
                    rust_point = rust_results[i]

                    self.assertEqual(python_point[0], rust_point[0])
                    self.assertEqual(python_point[1], rust_point[1])

                # See if there is more in the file
                quadLine = f.readline()

    # Ensure that the check_quad_for_tag function is identical between
    # Python and Rust
    def test_check_quad_for_tag(self):
        # Get the filtered image
        realImgFilt = actag_rust.median_filter_multithread(self.realImg, 4, None)

        # Make the tag params
        family_name = "AcTag24h10"
        tag_params = AcTagParams(family_name, 0.130628571428644)
        num_of_bit_corrections = 4

        # Open the file with the test data
        with open(path_to_actag_detection + "tests/test_data/check_quad_for_tag_inputs.txt", "r") as f:

            # Run a test for each pair of lines in the file
            quadLine = f.readline()
            while quadLine != "":
                dataPointsLine = f.readline()

                # Parse the quad
                temp_nums = []
                index = quadLine.find(" ")
                while index != -1:
                    temp_nums.append(int(quadLine[:index]))
                    quadLine = quadLine[index + 1:]
                    index = quadLine.find(" ")
                quad = []
                for i in range(0, len(temp_nums) - 1, 2):
                    quad.append((temp_nums[i], temp_nums[i + 1]))

                # Parse the data points
                temp_nums.clear()
                index = dataPointsLine.find(" ")
                while index != -1:
                    temp_nums.append(int(dataPointsLine[:index]))
                    dataPointsLine = dataPointsLine[index + 1:]
                    index = dataPointsLine.find(" ")
                data_bits = []
                for i in range(0, len(temp_nums) - 1, 2):
                    data_bits.append((temp_nums[i], temp_nums[i + 1]))

                # Get the Python and Rust results
                python_results = check_quad_for_tag(np.array(realImgFilt), quad, data_bits, 
                                                    tag_params, num_of_bit_corrections)
                rust_results = actag_rust.check_quad_for_tag_python_wrap(realImgFilt, quad, data_bits, 
                                                   tag_params.tags_in_family, num_of_bit_corrections)
                
                # Make sure they match
                self.assertEqual(len(python_results), len(rust_results))
                for i in range(len(python_results)):
                    python_tag = python_results[i]
                    rust_tag = rust_results[i]

                    # Make sure the id matches
                    self.assertEqual(python_tag[0], rust_tag[0])

                    # Make sure the corner locations match
                    python_corners = python_tag[1]
                    for j in range(1, 9):
                        self.assertEqual(python_corners[int((j-1)/2)][(j-1)%2], rust_tag[j])

                # See if there is more in the file
                quadLine = f.readline()

    # Ensure that tag decoding functionality matches between Rust and Python
    def test_decode_tags(self):
        # Get the thresholded image
        realImgFilt = actag_rust.median_filter_multithread(self.realImg, 4, None)

        # Get the contours
        realImgThreshold = actag_rust.adaptive_threshold_multithread(realImgFilt, 8, 1)
        realContours = actag_rust.get_contours(realImgThreshold, 0.1, 5.0, 1.0472, 0.130628571428644, 0.2, 0.2, 0.1, True, False, True, True)
        
        # Define Sonar and AcTag parameters
        min_range = 0.1
        max_range = 5.0
        horizontal_aperature = 1.0472
        sonar_params = SonarParams(min_range, max_range, horizontal_aperature)
        tag_family_name = "AcTag24h10"
        tag_params = AcTagParams(tag_family_name, 0.130628571428644)
        bits_to_correct = 4

        # Run the quad detection algorithm with multiple differnt seeds. This
        # will provide many different quadrilaterals to test with
        for seed in range(0, 30):
            # Get the quadrilaterals
            realQuads = actag_rust.fit_quadrilaterals_to_contours(realContours, (len(realImgThreshold), len(realImgThreshold[0])), 
                                           None, 0.1, 2.5, 0.85, 0.8, 0.9, True, 123456)

            # Get the Python & Rust results
            rust_results = actag_rust.decode_tags(realImgFilt, realQuads, min_range, max_range, horizontal_aperature, tag_family_name, bits_to_correct)
            python_results = decode_tags(np.array(realImgFilt), np.array(realQuads), sonar_params, tag_params, bits_to_correct)

            # Make sure the results match
            self.helper_assert_decoded_tags_equal(python_results, rust_results)

    # This will make sure that both sides of the algorithm match exactly,
    # all the way at the top level.
    def test_run_detection(self):
        # Load the test image
        img_path = "../example_scripts/exampleSonarImage.png"
        my_sonar_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Initialize the AcTag Detector for Python & Rust
        python_detector = AcTag(min_range=0.1, max_range=1.5, horizontal_aperture=1.0472, tag_family="AcTag24h10", 
                        tag_size=0.130628571428644, median_filter_kernel_radius=4, adaptive_threshold_kernel_radius=8, 
                        adaptive_threshold_offset=1, quads_use_same_random_vals=True, use_optimized_rust_code=False)
        rust_detector = AcTag(min_range=0.1, max_range=1.5, horizontal_aperture=1.0472, tag_family="AcTag24h10", 
                        tag_size=0.130628571428644, median_filter_kernel_radius=4, adaptive_threshold_kernel_radius=8, 
                        adaptive_threshold_offset=1, quads_use_same_random_vals=True, use_optimized_rust_code=True)
        
        # Get the results
        python_results = python_detector.run_detection(my_sonar_image)
        rust_results = rust_detector.run_detection(my_sonar_image)

        # Make sure they match
        self.helper_assert_final_results_equal(python_results, rust_results)

        # Load the video file
        cap = cv2.VideoCapture('../example_scripts/exampleSonarVideo.mp4')

        # Check if camera opened successfully
        if (cap.isOpened() == False): 
            raise Exception("Error opening video stream or file")
        
        # Test with the first 30 frames of the video
        for i in range(0, 30):
            # Get the next frame
            ret, frame = cap.read()

            # Convert it to greyscale
            my_sonar_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get the results
            python_results = python_detector.run_detection(my_sonar_image)
            rust_results = rust_detector.run_detection(my_sonar_image)

            # Make sure they match
            self.helper_assert_final_results_equal(python_results, rust_results)
        
        # When everything done, release the video capture object
        cap.release()

if __name__ == '__main__':
    unittest.main()