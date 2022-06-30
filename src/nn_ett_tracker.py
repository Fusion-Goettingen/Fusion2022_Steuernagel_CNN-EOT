"""
In this file, an example CNN-based Extended Target Tracker is implemented.
The tracker requires an external CNN (passed during instance initialization) to predict object sizes.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.ndimage import gaussian_filter
from copy import deepcopy

from utils import Rot, sigma_from_R, calc_R_scale_for_image_conversion, point_to_pixel
from base_classes import AbstractTracker


class TrackerNN(AbstractTracker):
    """
    Implements a CNN based Extended Target Tracker that produces a two-channel intensity measurement image with:
        ch1: collected measurements
        ch2: collected blurred measurements based on covariance information

    Handles Kinematic Tracking by a simple linear Kalman Filter and Orientation tracking strictly following the
    estimated velocity.
    """
    # normalized images will be scaled to have intensity in [0, ...]:
    # (usually, 255 should be chosen)
    IMAGE_INTENSITY_UPSCALE_FACTOR = 255

    def __init__(self,
                 x_init,
                 P_init,
                 R,
                 Q,
                 model,
                 image_size,
                 maximum_scaleable_distance,
                 gamma,
                 update_kinematics_from_statistics=True,
                 z_scale=0.25,
                 T=1):
        """
        Create a new instance of the tracker.

        :param x_init: Initial 4D state x: x, y, velocity_x, velocity_y
        :param P_init: Initial 4x4 state covariance corresponding to x_init.
        :param R: Measurement Noise Covariance R (2x2)
        :param Q: Process Noise Covariance Q (4x4)
        :param model: Neural Network that takes images and returns 2D size estimate in [0, 1]
        :param image_size: Size of internal measurement storage image passed to NN. E.g.: 300 for 300x300 images.
        :param maximum_scaleable_distance: A NN size estimate of '1' corresponds to this value in real-world size.
        Represents the maximum distance from the object center which is included in the discretized and scaled image,
        horizontally and vertically.
        :param gamma: Forgetting factor for previous measurements used in the predict step.
        :param update_kinematics_from_statistics: If False (default), the kinematic state will be updated sequentially
        from all measurements. Otherwise, the mean or median (depending on center_via_median) will be used to calculate
        a single measurement point
        :param z_scale: Scaling factor for cov mat from shape. Choose 1/4 if objects are assumed to be elliptical and
        1/3 for rectangular objects.
        :param T: Time Step length
        """
        # parameter assertions
        x_init = np.array(x_init)
        assert R.shape == (2, 2), "R is not of shape (2,2) (is {})".format(R.shape)
        assert x_init.shape == (4,), "x_init is not of shape (4,) (is {})".format(x_init.shape)
        assert P_init.shape == (4, 4), "P_init is not of shape (4,4) (is {})".format(P_init.shape)
        assert Q.shape == (4, 4), "Q is not of shape (4,4) (is {})".format(Q.shape)
        assert 0 <= gamma <= 1, "Gamma is {} but has to be in [0, 1]".format(gamma)

        # instance variables
        self.T = T
        self._z_scale = z_scale
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])

        self.update_kinematics_from_statistics = update_kinematics_from_statistics

        # internal KF for kinematic state
        self._KF_kin = KalmanFilter(dim_x=4, dim_z=2)
        self._KF_kin.x = x_init
        self._KF_kin.P = P_init
        self._KF_kin.H = self.H
        self._KF_kin.F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self._KF_kin.Q = Q
        self._KF_kin.R = R

        # size estimator
        self._image_size = image_size
        self._maximum_scaleable_distance = maximum_scaleable_distance
        self._model = model
        self._model.eval()
        self._gamma = gamma
        self._collected_measurements = np.zeros((self._image_size, self._image_size))
        self._collected_blurry_measurements = np.zeros((self._image_size, self._image_size))

        # define current state vars and extract them from internal KFs
        #   Note that length and width stay None until the first .update() call
        self.x = None
        self.P = None
        self.length = None
        self.width = None
        self._extract()

    def _extract(self):
        """
        Internal function that updates self. variables corresponding to estimates and covariances based on the internal
        Kalman Filter states.
        """
        self.x = self._KF_kin.x
        self.P = self._KF_kin.P

    def get_alpha(self):
        """
        Calculate the current heading information from velocity.
        :return: Orientation estimate in radians
        """
        return np.arctan2(self._KF_kin.x[3], self._KF_kin.x[2])

    def get_state(self):
        """
        Returns the seven dimensional state vector as a numpy array:
        x, y, velocity_x, velocity_y, orientation, length, width
        :return: 7D state (location, velocity, orientation, size)
        """
        self._extract()
        return np.array([*self.x, self.get_alpha(), self.length, self.width])

    def set_R(self, R):
        """
        Update internal measurement noise covariance with equally shaped new measurement noise cov. matrix
        :param R: New covariance matrix
        """
        R = np.array(R)
        assert np.array(self._KF_kin.R).shape == R.shape, "Old ({}) and new ({}) R.shape are different".format(
            np.array(self._KF_kin.R).shape,
            R.shape)
        self._KF_kin.R = R

    def predict(self):
        """
        Performs a predict step with all internal Kalman filters and applied the forgetting factor gamma to the
        previously collected measurements.
        :return: Predicted 7D state (location, velocity, orientation, size)
        """
        # kinematic predict
        self._KF_kin.predict()

        # downscale internal intensity measurement image
        self._collected_measurements *= self._gamma
        self._collected_blurry_measurements *= self._gamma

        # update internal variables
        self._extract()
        return self.get_state()

    def update(self, Z):
        """
        Update the tracker with a new set of measurements Z.

        :param Z: Measurements Points in shape (N, 2).
        :return: Updated 7D state (location, velocity, orientation, size)
        """
        # (1) Kinematic Update:
        if self.length is None or self.update_kinematics_from_statistics:
            # no size estimate exists so far - update kinematics based on center of measurement cloud
            Z = np.array(Z).reshape((-1, 2))
            z_bar = np.average(Z, axis=0)
            if self.length is None:
                R = self._KF_kin.R
            else:
                # additionally, use shape information
                R = self._KF_kin.R + \
                    (self._z_scale * (Rot(self.get_alpha()) @ np.diag([(self.length/2)**2, (self.width/2)**2]) @ Rot(self.get_alpha()).T))
                # based on number of measurements
                R /= Z.shape[0]
            self._KF_kin.update(z_bar, R=R)
        else:
            # update kinematics based on individual measurement points
            # with measurement noise adapted to take into account the current object size estimate
            for z in Z:
                temp_R = self._KF_kin.R + (Rot(self.get_alpha()) @ np.diag([self.length, self.width]) @ Rot(self.get_alpha()).T)
                self._KF_kin.update(z, R=temp_R)

        # _extract: update internal variables
        self._extract()

        # (2) Shape Update
        # size update:
        r_scale = calc_R_scale_for_image_conversion(image_size=self._image_size,
                                                    maximum_object_size=self._maximum_scaleable_distance)
        # update collected measurements
        new_measurements = np.zeros(self._collected_measurements.shape)
        for z in Z:
            # add each measurement to collections
            R_alpha = Rot(-self.get_alpha())
            z_centered_and_rotated = R_alpha @ (z - self.H @ self.x)
            # convert to pixel representation
            # note that this operation scales based on self._maximum_scaleable_distance
            x_px, y_px = self._to_pixel(z_centered_and_rotated)
            new_measurements[x_px, y_px] += 1

        # blurry measurements
        # calculate R based on kinematic R and P
        R = self.P[:2, :2] + self._KF_kin.R
        new_blurry_measurements = gaussian_filter(deepcopy(new_measurements), sigma_from_R(R * r_scale))

        # update internal measurement images
        self._collected_measurements = self._collected_measurements + new_measurements
        self._collected_blurry_measurements = self._collected_blurry_measurements + new_blurry_measurements

        # generate normalized size estimate to pass to model
        normalized_measurement_map = self.get_current_measurement_image()

        # pass to NN and estimate length and width of measured data
        model_output = self._model(normalized_measurement_map).cpu().detach().numpy().reshape((2,))

        # upscale model output based on self._maximum_scaleable_distance again
        self.length, self.width = self._maximum_scaleable_distance * model_output

        # return results
        return self.get_state()

    def get_current_measurement_image(self):
        """
        Returns a normalized version of the internal intensity measurement image, normalized individually across both
        image channels.
        :return: Current Normalized Internal Intensity Measurement Image
        """
        # normalize measurement map
        normalized_measurement_map = self._collected_measurements / np.max(self._collected_measurements)
        normalized_measurement_map *= self.IMAGE_INTENSITY_UPSCALE_FACTOR

        # normalize blurred measurement map
        normalized_blurry_measurements = self._collected_blurry_measurements / np.max(self._collected_blurry_measurements)
        normalized_blurry_measurements *= self.IMAGE_INTENSITY_UPSCALE_FACTOR

        # build current measurement image from normalized blurry+normal measurement map
        current_image = np.zeros((2, self._image_size, self._image_size))
        current_image[0, :, :] = normalized_measurement_map
        current_image[1, :, :] = normalized_blurry_measurements
        return current_image

    def _to_pixel(self, point):
        """
        Convert a point centered around the object to pixel coordinates.
        :param point: x,y point that is centered around the object center and aligned with the object orientation
        :return: [x_px, y_px] Point coordinates in internal image space
        """
        return point_to_pixel(point, self._maximum_scaleable_distance, self._image_size)

    def get_kin_P(self):
        """
        Quickly access the kinematic covariance ('P')
        :return: current kinematic state covariance
        """
        return self._KF_kin.P

