"""Contains general utility function used throughout the project"""
import numpy as np


def Rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return r.reshape((2, 2))


def get_ellipse_measurements(loc, length, width, theta, n_measurements, R, internal_RNG=None):
    """
    Given an elliptical extended object, returns n_measurements many measurement sources and noise-corrupted
    measurements across the entire object surface, based on measurement covariance R.

    Uses accept-reject sampling to uniformly generate measurement sources across the ellipse.
    :param loc: [x,y] location of object
    :param length: length of object in direction of orientation
    :param width: width of object, orthogonal to direction of orientation
    :param theta: orientation of object in radians
    :param n_measurements: number of measurements to draw. If <1, will be set to 1 instead!
    :param R: 2x2 measurement noise matrix
    :param internal_RNG: np random Generator or None. If None, a new generator will be created, without seed.
    :return: Y, Z: measurement sources and measurements, both as np arrays of shape (n_measurements, 2)
    """
    if n_measurements < 1:
        n_measurements = 1

    if internal_RNG is None:
        internal_RNG = np.random.default_rng()

    # half axis length
    half_length = length / 2
    half_width = width / 2
    Y = []
    while len(Y) < n_measurements:
        # draw new candidate point [x,y]
        x = internal_RNG.uniform(low=-half_length, high=half_length)
        y = internal_RNG.uniform(low=-half_width, high=half_width)

        # determine whether to check for <=1 or <1 (entire surface or not)
        # check if it matches ellipse equation:
        if (x ** 2 / half_length ** 2) + (y ** 2 / half_width ** 2) <= 1:
            # measurement y
            y = np.array([x, y])
            # rotate to match orientation
            y = Rot(theta) @ y
            # offset based on location of ellipse center
            y += loc
            # save
            Y.append(y)
    Y = np.vstack(Y)
    # apply gaussian noise with cov. matrix R to all measurements
    if R is not None:
        Z = np.vstack([internal_RNG.multivariate_normal(y, R) for y in Y])
    else:
        Z = Y
    return Y, Z


def velocity_to_orientation(velocity):
    """
    Convert a cartesian velocity vector to orientation
    :param velocity: Cartesian velocity vector [velocity_x, velocity_y]
    :return: Orientation in radians
    """
    return np.arctan2(*velocity[::-1])


def sigma_from_R(R):
    """
    Given measurement noise covariance matrix, approximate sigma values for a diagonal approximation of R.
    Use-case: efficient gaussian filtering in imagespace with non-diagonal R.

    :param R: measurement noise covariance
    :return: diagonal approx. covariance
    """
    # simple case: simply take diag indices
    sigma = R[np.diag_indices_from(R)]

    # alt/better: perform eigenvalue decomposition
    eigvals, _ = np.linalg.eig(R)
    # return average of eigvals |eigvals| many times
    return np.average(eigvals).repeat(len(eigvals))


def calc_R_scale_for_image_conversion(image_size, maximum_object_size):
    s = 0.05
    s *= image_size / maximum_object_size
    return s


def point_to_pixel(point, max_size, image_size, bound_xy=False):
    """
    Convert a point centered around the object to pixel coordinates
    :param point: x,y point that is centered around the object center and aligned with the object orientation
    :param max_size: Maximum Object Size that "fits" in the image
    :param image_size: Length/Width of the image, e.g. 300 for a 300x300 pixel image.
    :param bound_xy: Bool: Whether to bound x and y within [-self.max_size, self.max_size]
    :return: [x_px, y_px] Point coordinates in internal image space
    """
    # unpack point
    x, y = point

    # Bound x and y within [-self.max_size, self.max_size]
    if bound_xy:
        x = max(max_size, x)
        x = min(max_size, x)
        y = max(max_size, y)
        y = min(max_size, y)

    # map x from [-self.max_size, self.max_size] -> [0, image_size]
    #   add self.max_size
    #   multiply by image_size/(2*self.max_size)
    #   round to int
    scale = image_size / (2 * max_size)
    x_px = (x + max_size) * scale
    y_px = (y + max_size) * scale

    # pixel coord:
    #   integer between 0 and self._image_size - 1
    x_px = max(0, min(image_size - 1, int(x_px)))
    y_px = max(0, min(image_size - 1, int(y_px)))

    return np.array([x_px, y_px])
