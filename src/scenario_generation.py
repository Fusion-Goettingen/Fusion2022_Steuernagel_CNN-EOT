"""
Contains function that provide data generation utility for different scenarios.

    Data will be produced in the following format:
    Array of dicts, where every dict represents a time step and contains:
    {time: time at which this step took place,
     gt: [x, y, velocity, alpha, length, width]
     measurements: measurement data as an array of shape N,2 containing all measured data points
    }
"""
import numpy as np
from utils import get_ellipse_measurements, Rot, velocity_to_orientation
from copy import deepcopy


def get_slow_turn_data(time_step_length,
                       poisson_lambda,
                       R,
                       length=3,
                       width=1,
                       initial_velocity=1,
                       rotation_speed=0.03,
                       min_measurements=1,
                       n_steps=50,
                       RNG=None):
    # ensure velocity is float
    initial_velocity = float(initial_velocity)

    # make sure min_measurements is a positive int
    min_measurements = max(1, int(min_measurements))

    # set up internal RNG
    if RNG is None:
        RNG = np.random.default_rng()

    # Prepare ground truth (6D)
    gt_start = np.array([0, 0, initial_velocity, 0, 0, length, width])
    gt = gt_start

    # set start time
    current_time = 0

    # prepare output array and iterate over steps
    output = []
    for i in range(n_steps):
        current_time += time_step_length
        # 1) ground truth update
        # change orientation
        gt[4] += rotation_speed
        # ensure that this stays below 2pi
        gt[4] = gt[4] % (2 * np.pi)
        # add velocity to location
        gt[:2] += Rot(gt[4]) @ (time_step_length * gt[2:4])

        # adapt velocity
        gt_rot_velo = deepcopy(gt)
        gt_rot_velo[2:4] = Rot(gt[4]) @ np.array([np.linalg.norm(gt[2:4]), 0])

        # 2) measurement generation
        _, measurements = get_ellipse_measurements(loc=gt[:2],
                                                   length=gt[5],
                                                   width=gt[6],
                                                   theta=gt[4],
                                                   n_measurements=max(np.random.poisson(lam=poisson_lambda),
                                                                      min_measurements),
                                                   R=R,
                                                   internal_RNG=RNG)
        output.append(
            {"time": current_time,
             "gt": deepcopy(gt_rot_velo),
             "measurements": measurements}
        )

    return output


def get_random_walk_data(time_step_length,
                         poisson_lambda,
                         R,
                         length=3,
                         width=1,
                         initial_velocity=None,
                         velocity_change_max=None,
                         min_measurements=1,
                         n_steps=50,
                         RNG=None):
    # parse None params
    initial_velocity = [1, 0] if initial_velocity is None else initial_velocity
    velocity_change_max = max(initial_velocity) * 0.2 if velocity_change_max is None else velocity_change_max
    RNG = np.random.default_rng() if RNG is None else RNG

    x = [0, 0]
    v = initial_velocity

    output = []
    current_time = 0

    for i in range(n_steps):
        # advance to next time step
        current_time += time_step_length
        x[0] += v[0]
        x[1] += v[1]
        v[0] += RNG.uniform(low=-velocity_change_max, high=velocity_change_max)
        v[1] += RNG.uniform(low=-velocity_change_max, high=velocity_change_max)

        # set up current ground truth
        gt = [x[0], x[1], v[0], v[1], velocity_to_orientation(v), length, width]

        # generate measurements
        _, measurements = get_ellipse_measurements(loc=gt[:2],
                                                   length=gt[5],
                                                   width=gt[6],
                                                   theta=gt[4],
                                                   n_measurements=max(RNG.poisson(lam=poisson_lambda),
                                                                      min_measurements),
                                                   R=R,
                                                   internal_RNG=RNG)

        # save data
        output.append(
            {"time": current_time,
             "gt": deepcopy(gt),
             "measurements": measurements}
        )
    return output
