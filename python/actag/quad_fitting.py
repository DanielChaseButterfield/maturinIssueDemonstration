import numpy as np
import matplotlib.pyplot

def fit_quadrilaterals_to_contours(contour_list: list[list], 
                                   img_shape: tuple[int, int],
                                   random_seed: None, # or int
                                   points_per_line_ratio: float=0.1,
                                   dist_for_inlier: float=2.5,
                                   desired_inlier_ratio: float=0.85,
                                   required_inlier_ratio: float=0.8,
                                   parallel_threshold: float=0.9,
                                   use_same_random_vals: bool=False,
                                   starting_roll_val: int=123456) -> list[np.ndarray]:
    
    '''
    Attempts to fit a quadrilateral to each contour in the list, 
    using a custom RANSAC algorithm.
    
    :param contour_list: A list of contours, with each contour having the form ``[layer_num, row1, col1, row2, col2, ...]``.
    :type contour_list: list[list]
    :param img_shape: The shape of the image, in the form ``(num_rows, num_cols)``.
    :type img_shape: tuple[int, int]
    :param random_seed: If specified, the discrete uniform random number generator will be seeded with this value. If not specified, the random number generator will have a random seed, leading to new values each time it is run. 
    :type random_seed: None (or int)
    :param points_per_line_ratio: The ratio of the number of points within a contour to use for each line fit.
    :type points_per_line_ratio: float    
    :param dist_for_inlier: The distance threshold for a point to be considered an inlier.
    :type dist_for_inlier: float
    :param desired_inlier_ratio: The desired ratio of inliers within the total number of points.
    :type desired_inlier_ratio: float
    :param required_inlier_ratio: The required ratio of inliers within the total number of points.
    :type required_inlier_ratio: float
    :param parallel_threshold: The value used for determining whether or not two lines are parallel, when checking if the quad is a parallelogram.
    :type parallel_threshold: float
    :param use_same_random_vals: Set to true to use repeatable random values, starting with the ``starting_roll_val``. Used to syncronize random values between Python & Rust versions of this function (for testing purposes).
    :type use_same_random_vals: bool
    :param starting_roll_val: The starting roll value for when ``use_same_random_vals`` is set to true.
    :type starting_roll_val: int
    :return: A list of quadrilaterals. Each quadrilateral is an np.ndarray that contains four corners, each of which is a tuple of the form [row, column].
    :rtype: list[np.ndarray]
    '''
    
    # Initialize list to store the quadrilaterals
    quad_list = []

    # Make the random number generator
    if random_seed is None:
        rand = np.random.default_rng()
    else:
        rand = np.random.default_rng(random_seed)
    
    # TODO: Calculate the number of iterations for RANSAC
    N = 500
    
    # Iterate through the number of contours
    for contour in contour_list:

        # Set up values for rolls if not random
        curr_roll_val: np.int64 = starting_roll_val % 4294967296 # So that it's in the range of a u32 in Rust

        # Extract the contour points from the list
        point_rows = contour[1::2]
        point_cols = contour[2::2]

        # Convert to numpy array
        contour_points = np.array([point_rows, point_cols]).T
        [num_points, _]  = contour_points.shape

        # Calculate how many points are used for each line fit, ensuring 
        # that at least 2 points are used, as that is the minimum needed
        points_for_line = max(2, int(np.round(points_per_line_ratio * num_points)))

        # Set up variables to track the best solution from RANSAC
        num_iterations = 0
        best_inliers = []
        best_inlier_count = 0
        best_quad = []
        attempt_improvement = False
        # Run RANSAC, fitting a quadrilateral using a random sample of points
        while num_iterations <= N:
            lines = []
            slopes = []
            temp_points = contour_points.copy()

            for i in range(4):
                # Normal case, randomly select points for the lines
                if not attempt_improvement or len(best_inliers) == 0:
                    [solve_for, slope, intercept, temp_points] = get_random_point_and_fit_line(temp_points, points_for_line, use_same_random_vals, rand, curr_roll_val)
                    curr_roll_val = (curr_roll_val + 1) % 4294967296
                # Special case, on the last run of this loop we want to try to refine the lines based on the best inlier sets found
                else:
                    solve_for, slope, intercept = least_squares_line_fit(np.array(best_inliers[i]))

                # Round slopes and intercepts to 8 decimal places to ensure identical functionality to Rust code
                slope = np.round(slope, 8)
                intercept = np.round(intercept, 8)

                lines.append([solve_for, slope, intercept])
                # Handle equations of the form row = slope column + intercept (x = m y + b)
                if solve_for == 'row':
                    # Set slope to a very large number when the line is vertical
                    if abs(slope) < 1e-6:
                        slopes.append(1e6)
                    # Otherwise just invert the slope
                    else:
                        slopes.append(1 / slope)
                # For column = slope row + intercept (y = m x + b) just append the slope as is
                else:
                    slopes.append(slope)

            # Sort based on slope to identify which lines are used to calculate the intersections
            for i in range(0, 3):
                for j in range(i+1,4):
                    if (slopes[j] < slopes[i]): 
                        # Swap the slopes
                        temp = slopes[i]
                        slopes[i] = slopes[j]
                        slopes[j] = temp

                        # Also swap the corresponding lines
                        temp = lines[i]
                        lines[i] = lines[j]
                        lines[j] = temp

            # If any of the slopes are identical, make sure to sort by intercept as well
            # THIS seems unnecessary, but required to guaranteed identical results with Python code
            for i in range(0, 3):
                for j in range(i+1, 4):
                    if (slopes[j] == slopes[i]):
                        if (lines[j][2] < lines[i][2]):
                            # Swap the slopes
                            temp = slopes[i]
                            slopes[i] = slopes[j]
                            slopes[j] = temp

                            # Also swap the corresponding lines
                            temp = lines[i]
                            lines[i] = lines[j]
                            lines[j] = temp
            
            # Finally, if any of the slopes and intercepts are identical, sort by row/column form
            # THIS seems unnecessary, but required to guaranteed identical results with Python code
            for i in range(0, 3):
                for j in range(i+1, 4):
                    if (slopes[j] == slopes[i] and lines[j][2] == lines[i][2]):
                        j_form_val = (lines[j][0] == 'row')
                        i_form_val = (lines[i][0] == 'row')
                        if j_form_val < i_form_val:
                            # Swap the slopes
                            temp = slopes[i]
                            slopes[i] = slopes[j]
                            slopes[j] = temp

                            # Also swap the corresponding lines
                            temp = lines[i]
                            lines[i] = lines[j]
                            lines[j] = temp

            # Solve for the intersections, getting the four corners of the quadrilateral
            intersections = []
            
            # The order is such that we should get the four points in either clockwise or counterclockwise order around the quadrilateral
            try:
                intersections.append(get_intersection_of_lines(lines[0], lines[2]))
                intersections.append(get_intersection_of_lines(lines[0], lines[3]))
                intersections.append(get_intersection_of_lines(lines[1], lines[3]))
                intersections.append(get_intersection_of_lines(lines[1], lines[2]))
                intersections_array = np.array(intersections)
            except ValueError:
                num_iterations += 1
                continue

            # Move on to the next iteration if any of the intersection points lie outside of the image bounds
            if np.any(intersections_array < 0) or np.any(intersections_array >= img_shape):
                num_iterations += 1
                continue

            # If any of the intersections are the same point, this isn't a quadrilateral.
            # Go on to the next iteration
            intersections_match = False
            for i in range(0, 4):
                for j in range(i+1, 4):
                    if intersections[i][0] == intersections[j][0] and intersections[i][1] == intersections[j][1]:
                        intersections_match = True
            if intersections_match:
                num_iterations += 1
                continue

            # Calculate the distance between each point and each of the four sides of the quadrilateral
            distances = np.zeros((num_points, 4))
            for i, point in enumerate(contour_points):
                distances[i, 0] = distance_between_point_and_line_segment(point, intersections[0], intersections[1])
                distances[i, 1] = distance_between_point_and_line_segment(point, intersections[1], intersections[2])
                distances[i, 2] = distance_between_point_and_line_segment(point, intersections[2], intersections[3])
                distances[i, 3] = distance_between_point_and_line_segment(point, intersections[3], intersections[0])

            # Determine which line segment is the closest for each point, and determine the inliers
            mins = np.min(distances, axis=1)
            mins_idxs = np.argmin(distances, axis=1)
            inliers = [[], [], [], []]
            num_inliers = 0
            # Determine the number of inliers, and split them up according to which line segment they belong to
            for i, (min, idx) in enumerate(zip(mins, mins_idxs)):
                if min <= dist_for_inlier:
                    inliers[idx].append(contour_points[i,:])
                    num_inliers += 1

            # Only accept solutions that contain enough points for four unique lines (a valid quadrilateral)
            for i in inliers:
                if len(i) < 2:
                    num_inliers = 0

            # Update tracking of best solution if this is a better solution
            if num_inliers > best_inlier_count:
                best_inlier_count = num_inliers
                best_quad = intersections_array.copy()
                best_inliers = inliers.copy()
                # Continue attempting to improve the quadrilateral until we no longer see an increase in the number of inliers
                if attempt_improvement:
                    num_iterations = N - 1
            # Increment the number of iterations
            num_iterations += 1
            # Check for early termination criteria
            if num_inliers / num_points >= desired_inlier_ratio and attempt_improvement == False:
                attempt_improvement = True
                num_iterations = N
            # If we reach the maximum number of iterations, begin attempting improvement on the best solution found thus far
            elif num_iterations == N:
                if not attempt_improvement:
                    attempt_improvement = True

        # Check if the best fit we found meets the criteria to be a good enough fit to call it a probable quadrilateral
        if best_inlier_count / num_points >= required_inlier_ratio:
            # Check if the quadrilateral is a parallelogram by computing the dot product of the two sets
            # of unit vectors that should be parallel
            # TODO: Update how we check for a parallelogram to be more robust
            v1 = best_quad[0,:] - best_quad[1,:]
            v1 = v1 / np.sqrt(np.sum(v1**2))
            v2 = best_quad[3,:] - best_quad[2,:]
            v2 = v2 / np.sqrt(np.sum(v2**2))
            v3 = best_quad[1,:] - best_quad[2,:]
            v3 = v3 / np.sqrt(np.sum(v3**2))
            v4 = best_quad[0,:] - best_quad[3,:]
            v4 = v4 / np.sqrt(np.sum(v4**2))
            v1v2 = abs(np.dot(v1, v2))
            v3v4 = abs(np.dot(v3, v4))

            # Only accept the quadrilateral if the dot product is above the specified threshold that we consider to be a parallelogram
            if v1v2 >= parallel_threshold and v3v4 >= parallel_threshold:
                # Check if the points are in clockwise or counterclockwise order around the quadrilateral (sum over edges (r2-r1)(c2+c1))
                temp_quad = np.roll(best_quad, -1, axis=0)
                temp_quad = np.vstack([temp_quad[:, 0] - best_quad[:, 0], temp_quad[:, 1] + best_quad[:, 1]]).T
                # If the sum is negative, the points are in counter-clockwise order, and we need to reorder them to be clockwise
                if np.sum(temp_quad[:, 0] * temp_quad[:, 1]) < 0:
                    best_quad[[1, 3]] = best_quad[[3, 1]]
                # Append the quadrilateral to the list of quadrilaterals
                quad_list.append(best_quad)
    # Return the list of any quadrilaterals found
    return quad_list

# First, a number of helper functions, to make the code more readable and compact
def get_intersection_of_lines(line_1, line_2) -> list[int, int]:
    '''
    Calculate the intersection of two lines.
    
    :param line_1: The first line. It is stored as a list of the form ``[solve_for, slope, intercept]``, where ``solve_for`` is the equation form used to represent the line (``row`` or ``column``).
    :type line_1: list[str, float, float]
    :param line_2: The second line, formatted in the same way as ``line_1``.
    :type line_2: list[str, float, float]
    :return: The intersection of the two lines in the form ``[row, column]``, rounded to the nearest row and column index.
    :rtype: list[int, int]
    :raises ValueError: Raised if the two lines have the same slope, resulting in either 0 or infinite intersections.
    '''

    # Get the intersection of the two lines
    [solve_for_1, slope_1, intercept_1] = line_1
    [solve_for_2, slope_2, intercept_2] = line_2

    # Make sure they don't have the same slope
    if slope_1 == slope_2:
        raise ValueError("0 or infinite intersections for lines with identical slope")

    # Both equations are in the form column = slope row + intercept (y = m x + b)
    if solve_for_1 == 'column' and solve_for_2 == 'column':
        row = (intercept_2 - intercept_1) / (slope_1 - slope_2)
        column = slope_1 * row + intercept_1
    # Both equations are in the form row = slope column + intercept (x = m y + b)
    elif solve_for_1 == 'row' and solve_for_2 == 'row':
        column = (intercept_2 - intercept_1) / (slope_1 - slope_2)
        row = slope_1 * column + intercept_1
    # The first equation is in the form column = slope row + intercept (y = m x + b) 
    # while the second is in the form row = slope column + intercept (x = m y + b)
    elif solve_for_1 == 'column' and solve_for_2 == 'row':
        column = (slope_1 * intercept_2 + intercept_1) / (1 - slope_1 * slope_2)
        row = slope_2 * column + intercept_2
    # The first equation is in the form row = slope column + intercept (x = m y + b)
    # while the second is in the form column = slope row + intercept (y = m x + b)
    else:
        row = (slope_1 * intercept_2 + intercept_1) / (1 - slope_1 * slope_2)
        column = slope_2 * row + intercept_2

    # Return the intersection point
    return np.round([row, column]).astype(int)

def distance_between_point_and_line_segment(point: np.ndarray, line_point_1: np.ndarray, line_point_2: np.ndarray) -> float:
    '''
    Calculates the smallest distance between a point and a line. The line is represented by two points.

    :param point: 2D point, of the form ``[row, column]``.
    :type point: np.ndarray
    :param line_point_1: A point on the line segment, of the form ``[row, column]``.
    :type line_point_1: np.ndarray
    :param line_point_2: A different point on the line segment, in the same form as ``line_point_1``.
    :type line_point_2: np.ndarray
    :return: The smallest distance between the point and any point on the line.
    :rtype: float
    '''    

    [rp, cp] = point
    [r1, c1] = line_point_1
    [r2, c2] = line_point_2
    # Vector for the line segment
    vec_12 = [r2 - r1, c2 - c1]
    # Vector for line_point_1 to point
    vec_1p = [rp - r1, cp - c1]
    # Vector for line_point_2 to point
    vec_2p = [rp - r2, cp - c2]
    # Dot products
    dp_12_2p = np.dot(vec_12, vec_2p)
    dp_12_1p = np.dot(vec_12, vec_1p)
    # Handle when closest point is line_point_2
    if dp_12_2p > 0:
        dist = np.sqrt(vec_2p[0]**2 + vec_2p[1]**2)
    # Handle when the closest point is line_point_1
    elif dp_12_1p < 0:
        dist = np.sqrt(vec_1p[0]**2 + vec_1p[1]**2)
    # Handle when the closest point is between line_point_1 and line_point_2
    else:
        dist = abs(vec_12[0] * vec_1p[1] - vec_12[1] * vec_1p[0]) / np.sqrt(vec_12[0]**2 + vec_12[1]**2)
    # Return the distance
    return dist
    
def distance_between_point_and_line(point: np.ndarray, line_type: str, slope: float, intercept: float) -> float:
    '''
    Calculate the smallest distance between a point and a line. 
    The line is represented by an equation form (``row`` or ``column``), 
    a slope, and an intercept.

    :param point: 2D point, of the form ``[row, column]``.
    :type point: np.ndarray
    :param line_type: The form of the equation for the line, either ``row`` or ``column``.
    :type line_type: str
    :param slope: The slope of the line.
    :type slope: float
    :param intercept: The intercept of the line.
    :type intercept: float
    :return: The smallest distance between the point and any point on the line.
    :rtype: float
    '''
    [r, c] = point
    # For r = slope c + intercept (x = m y + b)
    if line_type == 'row':
        return abs(intercept + slope * c - r) / np.sqrt(1 + slope**2)
    # For c = slope r + intercept (y = m x + b)
    else:
        return abs(intercept + slope * r - c) / np.sqrt(1 + slope**2)
        
def least_squares_line_fit(line_points: np.ndarray) -> tuple[str, float, float]:
    '''
    Fit a line to the points using least squares, trying forms ``y = m * x + b`` and 
    ``x = m * y + b`` to see which is the better fit. This is necessary because linear 
    regression for vertical lines is very poor with the form ``y = m * x + b``; likewise 
    for horizontal lines we do not want to use the form ``x = m * y + b``.
    
    :param line_points: The points used to fit a line to.
    :type line_points: np.ndarray
    :return: The form of the line fit (``row`` or ``column``), the slope, and the intercept.
    :rtype: tuple(str, float, float)
    '''

    try:
        [num_points, _] = line_points.shape
    except ValueError:
        print(line_points)
        exit()
    # y = m x + b
    A1 = np.vstack([line_points[:, 0], np.ones(num_points)]).T
    slope_x, intercept_x = np.linalg.lstsq(A1, line_points[:, 1], rcond=None)[0]
    
    # x = m y + b
    A2 = np.vstack([line_points[:, 1], np.ones(num_points)]).T
    slope_y, intercept_y = np.linalg.lstsq(A2, line_points[:, 0], rcond=None)[0]
    
    # Calculate sum of squares for each line approximation
    err_x = 0
    err_y = 0
    for r, c in line_points:
        err_x += (c - (slope_x * r + intercept_x))**2
        err_y += (r - (slope_y * c + intercept_y))**2

    # Round to 8 decimal places, so that functionality is identical to Rust
    err_x = np.round(err_x, 8)
    err_y = np.round(err_y, 8)

    # Determine which line is a better approximation, and set the output values appropriately
    if err_x < err_y:
        solve_for = 'column'
        slope = slope_x
        intercept = intercept_x
    else:
        solve_for = 'row'
        slope = slope_y
        intercept = intercept_y
    # Return the line parameters
    return solve_for, slope, intercept

def get_random_point_and_fit_line(shape_points: np.ndarray, points_per_line: int, 
                                  use_same_random_vals: bool, rand: np.random.Generator, 
                                  supplied_roll_value: int=0) -> list[float, float, float, np.ndarray]:
    '''
    Randomly sample ``points_per_line`` points from ``shape_points`` with the provided random 
    number generator, and then fit a line to these points using least squares. If ``use_same_random_vals`` 
    is true, roll to the ``supplied_roll_value`` instead of randomly.

    :param shape_points: All of the points in the contour to randomly sample from.
    :type shape_points: np.ndarray
    :param points_per_line: The number of points to sample for a line.
    :type points_per_line: int
    :param use_same_random_vals: If true, sample points starting at the ``supplied_roll_value`` instead of randomly.
    :type use_same_random_vals: bool
    :param rand: A random number generator used to sample randomly.
    :type rand: np.random.Generator
    :param supplied_roll_value: If ``use_same_random_vals`` is true, then sample points starting from this value.
    :type supplied_roll_value: int
    :return: A list, containing the form of the line (``row`` or ``column``), the slope, the intercept, and any leftover points from shape_points that were not used to fit this line.
    :rtype: list[float, float, float, np.ndarray]
    '''

    # Get the total number of points in the shape
    [rows, _] = shape_points.shape

    # If there are not enough points to fit a line, return an empty list, and a line at y = 0
    if rows < points_per_line:
        return ['column', 0, 0, np.array([])]
    
    # Randomly roll the points and take the first 'points_for_line' points.
    if not use_same_random_vals:
        roll_value = rand.integers(0, rows) # Uses the Discrete Uniform distribution
    else:
        roll_value = supplied_roll_value % rows
    
    temp_points = np.roll(shape_points, roll_value, axis=0)
    line_points = temp_points[0:points_per_line, :]
    leftover_points = temp_points[points_per_line:, :]

    # Get approximate line for the points
    solve_for, slope, intercept = least_squares_line_fit(line_points)

    # Return the line parameters and the leftover points.
    return [solve_for, slope, intercept, leftover_points]

def add_quads_to_axis(ax: matplotlib.pyplot.Axes, quads: list) -> matplotlib.pyplot.Axes:
    '''
    Draw a set of quadrilaterals onto a matplotlib axis for visualization.

    :param ax: The matplotlib axes to draw the quadrilaterals onto.
    :type ax: matplotlib.pyplot.Axes
    :param quads: A list of quadrilaterals to draw onto the axis.
    :type quads: list
    :return: The matplotlib axes with the quadrilaterals drawn onto it.
    :rtype: matplotlib.pyplot.Axes
    '''

    for quad_pts in quads:
        for pt in quad_pts:
            ax.scatter(pt[1], pt[0], c='g', s=2)
        for i in range(4):
            ax.plot([quad_pts[i][1], quad_pts[(i+1)%4][1]], [quad_pts[i][0], quad_pts[(i+1)%4][0]], c='g')
    return ax