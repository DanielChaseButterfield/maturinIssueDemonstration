import numpy as np

class SonarParams:
    '''
    Class that contains the parameters of the sonar.    
        
    :param min_range: Minimum range of the sonar in meters.
    :type min_range: float
    :param max_range:  Maximum range of the sonar in meters.
    :type max_range: float
    :param horizontal_aperture: Horizontal aperture of the sonar in radians.
    :type horizontal_aperture: float
    '''
    def __init__(self, min_range: float, max_range: float, horizontal_aperture: float) -> None:
        # Set the input parameters as attributes of the class
        self.min_range = min_range
        self.max_range = max_range
        self.horizontal_aperture = horizontal_aperture
        # Calculate additional parameters based on the input parameters
        self.min_azimuth = -horizontal_aperture / 2
        self.max_azimuth = horizontal_aperture / 2