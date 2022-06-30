from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from utils import Rot, sigma_from_R, calc_R_scale_for_image_conversion, point_to_pixel
from scenario_generation import get_random_walk_data
from scipy.ndimage import gaussian_filter


class RandomWalkEllipseDataset(Dataset):
    """
    Performs a random walk with objects of varying sizes, and saves randomly generated measurements as a data set
    Image data can be built in different ways:
    1. Only using the original measurements (produces single-channel images)
    2. Additionally applying gaussian blur based on measurement noise (produces 2channel images)
    3. Using a KF based formulation (3channels: measurements-only (like (1)), KF state, KF variance)
    """

    def __init__(self,
                 n_scenes,
                 n_steps_per_scene,
                 skip_first_n_steps=0,
                 min_size=1,
                 max_size=10,
                 image_size=300,
                 noise_scaling_factor=1.,
                 uniformly_vary_noise=True,
                 measurement_lambda_min_max=None,
                 gamma=0.95,
                 uniformly_vary_gamma=False,
                 transform=None,
                 gaussian_blur=False,
                 RNG=None,
                 max_size_generation_factor=1.0
                 ):
        """
        Create a new data set.

        :param n_scenes: Number of random walk scenes to generate
        :param n_steps_per_scene: Number of steps per generated scene
        :param skip_first_n_steps: The first skip_first_n_steps will not be included, but processed as part of the
        intensity images.
        :param min_size: Minimum Object Size
        :param max_size: Maximum Object Size = Scaling Parameter. See max_size_generation_factor too
        :param image_size: Image size. Images will be square images of shape (image_size, image_size)
        :param noise_scaling_factor: Measurement noise will be N(0, np.eye(2)*noise_scaling_factor). See
        uniformly_vary_noise too.
        :param uniformly_vary_noise: If True, for every scene, the noise_scaling_factor will be uniformly picked from
        [0, noise_scaling_factor]
        :param measurement_lambda_min_max: Minimum and Maximum lambda (mean) for the poisson distributed number of
        measurements per step. Or None for default value.
        :param gamma: Gamma to use for downscaling previous time-step's measurement data when building measurement
        images
        :param uniformly_vary_gamma: Bool: Whether gamma should be uniformly drawn from [0, gamma] for each individual
        scene
        :param transform: Transform to apply to images or None
        :param RNG: Random Number Generator or None to create a new one.
        :param max_size_generation_factor: Maximum sizes of actually generated objects is
        'max_size_generation_factor * max_size'. E.g.: Set to 1.5 and maxsize=10 to generate objects of up to 15m size
        while the scaling factor for transformation into images is fixed at 10.
        """
        self._n_scenes = n_scenes
        self._n_steps_per_scene = n_steps_per_scene
        self._skip_first_n_steps = skip_first_n_steps
        self.min_size = min_size
        self.max_size = max_size
        self._image_size = image_size
        self.transform = transform
        self._r_factor = noise_scaling_factor
        self._uniformly_vary_noise = uniformly_vary_noise
        self._measurement_lambda_min_max = [5, 50] if measurement_lambda_min_max is None else measurement_lambda_min_max
        self._gamma = gamma
        self._uniformly_vary_gamma = uniformly_vary_gamma
        self._max_size_generation_factor = max_size_generation_factor

        self.RNG = np.random.default_rng() if RNG is None else RNG

        self._apply_gaussian_blur = gaussian_blur

        if self._apply_gaussian_blur:
            self.n_channels = 2
        else:
            self.n_channels = 1

        # -------
        # FINALLY:
        self._data = self._make_images()
        self._n_image = len(self._data)

    def __len__(self):
        return self._n_image

    def _to_pixel(self, point):
        """
        Convert a point from actual size into [0, image_size] as integer, to be placed into the raster image.
        :param point: x,y point that is centered around the object center and aligned with the object orientation
        :return: [x_px, y_px] Point coordinates in internal image space
        """
        return point_to_pixel(point, self.max_size, self._image_size)

    def __getitem__(self, idx):
        """
        Return image and label corresponding to index
        :param idx: index of data set entry
        :return: image, label: image with corresponding length, width label
        """
        sample = self._data[idx]
        image, label = sample
        image: np.ndarray
        image = image.reshape((self.n_channels, 300, 300))
        if self.transform:
            image = self.transform(image)
        return image, label

    def _make_images(self):
        """Create images based on parameters set in instance of data set, and return them for saving"""
        images = []

        for traj_count in range(self._n_scenes):
            poisson_low, poisson_high = self._measurement_lambda_min_max
            length = np.random.uniform(low=self.min_size, high=self.max_size*self._max_size_generation_factor)
            width = np.random.uniform(low=self.min_size, high=self.max_size*self._max_size_generation_factor)

            # generate noise with randomization across x/y dim
            if self._uniformly_vary_noise:
                current_R = np.diag(self.RNG.uniform(low=0, high=self._r_factor, size=2))
            else:
                current_R = np.eye(2) * self._r_factor
            original_data = get_random_walk_data(time_step_length=1,
                                                 poisson_lambda=self.RNG.integers(low=poisson_low,
                                                                                  high=poisson_high + 1),
                                                 R=current_R,
                                                 initial_velocity=[4, 0],
                                                 velocity_change_max=0,
                                                 min_measurements=1,
                                                 length=length,
                                                 width=width,
                                                 n_steps=self._n_steps_per_scene)
            gamma = self._gamma if not self._uniformly_vary_gamma else self.RNG.uniform(low=0, high=self._gamma)
            images.extend(self._trajectory_data_to_images(original_data, gamma, current_R)[self._skip_first_n_steps:])

        np.random.shuffle(images)
        return images

    def _trajectory_data_to_images(self, trajectory, gamma, R=None):
        """
        Convert data in trajectory (list of dicts<gt, measurements>) format to image data.

        Builds internal measurement intensity images sequentially from the data
        :param trajectory: Trajectory data
        :param gamma: Forgetting factor used for building internal measurement images
        :return: Image data corresponding to trajectory data
        """
        image_label_data = []

        blurred_intensity_image = np.zeros((self._image_size, self._image_size))
        measurement_intensity_image = np.zeros((self._image_size, self._image_size))
        for i, data in enumerate(trajectory):
            n_steps = len(data["measurements"])

            # update intensity image from last step using forgetting factor gamma
            blurred_intensity_image *= gamma
            measurement_intensity_image *= gamma

            blurred_intensity_image_update = np.zeros(blurred_intensity_image.shape)
            for point in data["measurements"]:
                # ensure point is numpy
                point = np.array(point)

                # center point on object location
                point -= np.array(data["gt"][:2])

                # rotate point according to object rotation (gt[4])
                point = Rot(data["gt"][4]) @ point

                # convert to pixel coordinates
                x, y = self._to_pixel(point)
                blurred_intensity_image_update[x, y] += 1
                measurement_intensity_image[x, y] += 1

            if self._apply_gaussian_blur and R is not None:
                r_scale = calc_R_scale_for_image_conversion(image_size=self._image_size,
                                                            maximum_object_size=self.max_size)
                blurred_intensity_image_update = gaussian_filter(blurred_intensity_image_update,
                                                                 sigma_from_R(R * r_scale))

            blurred_intensity_image = blurred_intensity_image + blurred_intensity_image_update

            # After adding all new points:
            # Re-Normalize intensity image to be in [0, 255]
            normalized_blurred_intensity_image = 255 * (blurred_intensity_image / np.max(blurred_intensity_image))

            if self._apply_gaussian_blur:
                output_image = np.zeros((2, self._image_size, self._image_size))
                output_image[0, :, :] = measurement_intensity_image
                output_image[1, :, :] = normalized_blurred_intensity_image
            else:
                output_image = deepcopy(measurement_intensity_image)

            image_label_data.append(
                (output_image,
                 (data["gt"][5] / self.max_size, data["gt"][6] / self.max_size)
                 )
            )
        return image_label_data


if __name__ == '__main__':
    pass
