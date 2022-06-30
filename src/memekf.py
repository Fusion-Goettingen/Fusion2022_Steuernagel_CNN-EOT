"""
Contains a simple MEM-EKF* Implementation, presented in
    Tracking the Orientation and Axes Lengths of an Elliptical Extended Object - Yang et al.
see:
    https://ieeexplore.ieee.org/document/8770112
"""
import numpy as np
from utils import Rot
from base_classes import AbstractTracker


def vect(M):
    """
    From original MEM-EKF* paper:
    Constructs a column vector from a matrix M by stacking its column vectors
    """
    v = M.flatten(order="F")  # just use ndarray.flatten(), pass `order='F'` for column-major order
    v = np.reshape(v, (len(v), 1))  # ensure output is column vector
    return v


class TrackerMEMEKF(AbstractTracker):
    def __init__(self,
                 m_init,
                 p_init,
                 C_m_init,
                 C_p_init,
                 R,
                 H=None,
                 C_h=None,
                 Q=None,
                 Q_extent=None):
        """
        Initialize a new tracker

        :param m_init: Initial kinematic estimate in the form [loc_x, loc_y, speed] where speed is the velocity in
        direction of orientation.
        :param p_init: Initial extent estimate in the form [orientation, length, width] where l,w are semi-axis length
        :param C_m_init: Initial covariance matrix of the kinematic estimate (3x3)
        :param C_p_init: Initial covariance matrix of the extent estimate (3x3)
        :param R: Measurement Noise covariance matrix
        :param H: Measurement Matrix or None. If None, will assume 3D state and use standard H
        :param C_h: Covariance matrix of multiplicative Gaussian noise, or None. If none will be I*0.25
        :param Q: Process Noise Covariance Matrix or None (in which case Q=0)
        :param Q_extent: Extent Process Noise Covariance matrix or None (in which case Q_extent = 0)
        """
        self.m = np.array(m_init)
        self.p = np.array(p_init)

        self.C_m = np.array(C_m_init)
        self.C_p = np.array(C_p_init)

        self.H = np.array(H) if H is not None else np.hstack([np.eye(2), np.zeros((2, 3-2))])
        self.R = np.array(R)
        self.C_h = np.array(C_h) if C_h is not None else 0.25 * np.eye(2, 2)
        self.Q = Q if Q is not None else np.diag([0, 0, 0])
        self.Q_extent = Q_extent if Q_extent is not None else np.diag([1e-3, 1e-3, 1e-3])

    def predict(self):
        """
        Perform a predict (time update) step.
        """
        self.m, self.p, self.C_m, self.C_p = self.predict_memekf(self.m, self.p, self.C_m, self.C_p)

    def update(self, Z):
        """
        Update the tracker given a set of Measurements Z as a (N, 2) array.
        :param Z: (N, 2) array of measurements
        """
        for z in Z:
            self.m, self.p, self.C_m, self.C_p = self.update_memekf(z, self.m, self.p, self.C_m, self.C_p)

    def set_R(self, R):
        """
        Set the Measurement Noise Covariance R

        :param R: Measurement Noise Covariance
        """
        R = np.array(R)
        assert np.array(self.R).shape == R.shape, "Old ({}) and new ({}) R.shape are different".format(np.array(self.R).shape,
                                                                                                       R.shape)
        self.R = R

    def get_state(self):
        """
        Get the current state estimate.

        :return: 7D State: [x, y, velocity_x, velocity_y, orientation, length, width]
        """
        # location
        x, y, speed = self.m.reshape((3, )).astype(float)

        # extent
        theta, length, width = self.p.reshape((3, )).astype(float)
        length, width = length*2, width*2

        # cartesian velocity
        vel_x, vel_y = 0, 0  # TODO calc cartesian velo from speed and theta
        return np.array([x, y, vel_x, vel_y, theta, length, width])

    def update_memekf(self, z, m_minus, p_minus, C_m_minus, C_p_minus):
        """
        Update function for a single measurement.

        :param z: Single measurement
        :param m_minus: Kinematic state
        :param p_minus: Extent state
        :param C_m_minus: Kinematic state covariance
        :param C_p_minus: Extent state covariance
        :return: m_plus, p_plus, C_m_plus, C_p_plus : Kinematic State, Extent State, Kinematic state covariance,
        Extent state covariance - after incorporating z
        """
        # unpack p_minus
        alpha_minus, l1_minus, l2_minus = p_minus.reshape((3,))

        # F
        F = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        # F tilde
        Ft = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        S = Rot(alpha_minus) @ np.diag([l1_minus, l2_minus])
        S_1 = S[0, :].reshape((1, 2))
        S_2 = S[1, :].reshape((1, 2))

        J_1 = np.block([
            [-1 * l1_minus * np.sin(alpha_minus), np.cos(alpha_minus), 0],
            [-1 * l2_minus * np.cos(alpha_minus), 0, -1 * np.sin(alpha_minus)]
        ])
        J_2 = np.block([
            [l1_minus * np.cos(alpha_minus), np.sin(alpha_minus), 0],
            [-1 * l2_minus * np.sin(alpha_minus), 0, np.cos(alpha_minus)]
        ])
        J = [J_1, J_2]

        C_I = S @ self.C_h @ S.T
        C_II = np.block([
            [np.trace(C_p_minus @ J[m].T @ self.C_h @ J[n]) for m in range(2)] for n in range(2)
        ]).reshape(2, 2)

        M = np.array([
            [2 * S_1 @ self.C_h @ J_1],
            [2 * S_2 @ self.C_h @ J_2],
            [S_1 @ self.C_h @ J_2 + S_2 @ self.C_h @ J_1]
        ])
        M = np.reshape(M, (3, -1))

        C_mz = C_m_minus @ self.H.T
        C_z = self.H @ C_m_minus @ self.H.T + C_I + C_II + self.R
        Z = F @ np.kron(z - self.H @ m_minus, z - self.H @ m_minus)
        Z = np.reshape(Z, (-1, 1))
        Z_bar = F @ vect(C_z)

        C_pZ = C_p_minus @ M.T
        C_Z = F @ np.kron(C_z, C_z) @ (F + Ft).T

        # prepare for final calculations - invert C_z and C_Z
        C_z_inv = np.linalg.inv(C_z)
        C_Z_inv = np.linalg.inv(C_Z)
        p_minus = p_minus.reshape((-1, 1))

        # finally: calculate m, p, C_m, C_p
        m_plus = m_minus + C_mz @ C_z_inv @ (z - self.H @ m_minus)
        C_m_plus = C_m_minus - C_mz @ C_z_inv @ C_mz.T
        p_plus = p_minus + C_pZ @ C_Z_inv @ (Z - Z_bar)
        C_p_plus = C_p_minus - C_pZ @ C_Z_inv @ C_pZ.T

        # enforce symmetry of covariance:
        #     in some scenarios it may be useful to ensure that C^p and C^m are symmetric
        #     the following (optional) two lines enforce the symmetry before returing the matrices
        #     in this simple scenario, this is not necessary
        C_p_plus = (C_p_plus + C_p_plus.T) * 0.5
        C_m_plus = (C_m_plus + C_m_plus.T) * 0.5
        return m_plus, p_plus, C_m_plus, C_p_plus

    def predict_memekf(self, m_minus, p_minus, C_m_minus, C_p_minus):
        """
        Predict step for the MEM-EKF*

        :param m_minus: Kinematic state
        :param p_minus: Extent state
        :param C_m_minus: Kinematic state covariance
        :param C_p_minus: Extent state covariance
        :return: m_plus, p_plus, C_m_plus, C_p_plus : Kinematic State, Extent State, Kinematic state covariance,
        Extent state covariance - after prediction
        """
        # update kinematics
        m_plus = m_minus.astype(float)
        m_plus[:2] += Rot(p_minus[0]) @ np.array([m_minus[2], 0])

        C_m_plus = C_m_minus + self.Q

        # predict extent
        p_plus = p_minus

        C_p_plus = C_p_minus + self.Q_extent
        return m_plus, p_plus, C_m_plus, C_p_plus
