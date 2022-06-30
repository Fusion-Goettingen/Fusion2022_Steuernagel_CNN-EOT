"""
Implements the IAE tracker, presented in
    On Independent Axes Estimation for Extended Target Tracking - Felix Govaers
see:
    https://ieeexplore.ieee.org/abstract/document/8916660
"""
import numpy as np
from base_classes import AbstractTracker
from utils import Rot


class TrackerIAE(AbstractTracker):
    """
    IAE Tracker Implementation

    x: 4D state
    P: 4D state cov
    l: semi axis length
    c: semi axis uncertainty
    """
    def __init__(self,
                 x_init,
                 P_init,
                 l_init,
                 c_init,
                 R=None,
                 time_step_length=1,
                 H=None,
                 Q=None,
                 F=None,
                 q_c=0,  # Noise parameter modeling object size change over time. Set to 0 due to fixed object size.
                 c_scaling=0.25):
        """
        Initialize a new IAE Tracker Instance

        :param x_init: Initial Location
        :param P_init: Initial Covariance
        :param l_init: Initial axes length
        :param c_init: Initial axes length uncertainty
        :param R: Measurement noise covariance matrix
        :param time_step_length: Length of a single time step
        :param H: Measurement Matrix
        :param Q: Process Noise Covariance Matrix
        :param F: State Transition Model
        :param q_c: Parameter controlling change of axes length over time. Set to 0 for fixed object size
        :param c_scaling: scaling parameter c
        """
        self.x = np.array(x_init)
        self.P = np.array(P_init)
        self.len_semi_axis = np.array(l_init)
        self.c = np.array(c_init)

        self.R = np.eye(2) if R is None else R
        self.q_c = q_c
        self.H = self.H if H is not None else np.block([np.eye(2), np.zeros((2, 2))])
        self.Q = np.eye(4) * 0.001 if Q is None else Q
        self.F = np.array([
            [1, 0, time_step_length, 0],
            [0, 1, 0, time_step_length],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) if F is None else F
        self.c_scaling = c_scaling

    def predict(self):
        """
        Perform a predict (time update) step.
        """
        self.x, self.P, self.len_semi_axis, self.c = self.predict_iae(self.x, self.P, self.len_semi_axis, self.c)

    def update(self, Z):
        """
        Update the tracker given a set of Measurements Z as a (N, 2) array.
        :param Z: (N, 2) array of measurements
        """
        self.x, self.P, self.len_semi_axis, self.c = self.update_iae(Z, self.x, self.P, self.len_semi_axis, self.c)

    def set_R(self, R):
        """
        Set the Measurement Noise Covariance R

        :param R: Measurement Noise Covariance
        """
        self.R = R

    def get_state(self):
        """
        Get the current state estimate.

        :return: 7D State: [x, y, velocity_x, velocity_y, orientation, length, width]
        """
        state = np.zeros((7, ))
        state[:4] = self.x
        state[4] = np.arctan2(self.x[3], self.x[2])
        state[5:] = np.array(self.len_semi_axis) * 2  # convert semi to full axis length
        return state

    def predict_iae(self, x_minus, P_minus, l_minus, c_minus):
        """
        Predict function for the IAE algorithm. Parameters:
        x_minus: Prior kinematic state estimate
        P_minus: Prior kinematic state covariance
        l_minus: 2D Array of Estimated semi-axis lengths
        c_minus: 2D Arrayof semi-axis length variances.
        """
        x_minus = np.array(x_minus)
        P_minus = np.array(P_minus)
        l_minus = np.array(l_minus)
        c_minus = np.array(c_minus)
        x_plus = self.F @ x_minus
        P_plus = self.F @ P_minus @ self.F.T + self.Q
        l_plus = l_minus
        # q_c: defined in "VARIABLES" section above
        c_plus = c_minus + self.q_c
        return x_plus, P_plus, l_plus, c_plus

    def update_iae(self, Z, x_minus, P_minus, l_minus, c_minus):
        """
        Update ('filtering') function for the IAE algorithm. Parameters:
        Z: measurements
        x_minus: Prior kinematic state estimate
        P_minus: Prior kinematic state covariance
        l_minus: 2D Array of Estimated semi-axis lengths
        c_minus: 2D Array of semi-axis length variances.
        """
        x_minus = np.array(x_minus)
        P_minus = np.array(P_minus)
        l_minus = np.array(l_minus)
        c_minus = np.array(c_minus)

        # (1) Kinematic Update
        n = len(Z)
        z_avg = np.average(Z, axis=0)
        innov = z_avg - self.H @ x_minus

        alpha = np.arctan2(x_minus[3], x_minus[2])
        L = np.diag(l_minus)
        # R: measurement noise covariance
        # c: defined in "VARIABLES" section above
        R_bar = (1 / n) * (self.c_scaling * (Rot(alpha) @ L ** 2 @ Rot(alpha).T) + self.R)
        S = self.H @ P_minus @ self.H.T + R_bar  # innovation covariance

        W = P_minus @ self.H.T @ np.linalg.inv(S)  # gain

        # update parameters:
        x_plus = x_minus + W @ innov
        P_plus = P_minus - W @ S @ W.T

        # (2) Shape Update
        # measurement observation of half axis length d with corresponding variance v (both as 2D array)
        if len(Z) > 2:
            d, v = self.half_axis_observation(Z=Z, R_k=self.R, x_plus=x_plus)

            s_l = c_minus + v
            w_l = c_minus / s_l

            # TODO: w_l[1] is always close to 1?
            # print("c=", c_minus, "\tv=", v, "w_l=", w_l)

            l_plus = l_minus + w_l * (d - l_minus)
            # NOTE: paper uses /s_l - causes huge problems - use *s_l instead
            c_plus = c_minus - ((w_l ** 2) * s_l)
        else:
            # not enough measurements for half axis observation model, no change
            l_plus = l_minus
            c_plus = c_minus

        return x_plus, P_plus, l_plus, c_plus

    def half_axis_observation(self, Z, R_k, x_plus):
        """
        Computing the half axis measurement from given sensor data Z and noise with covariance R_k
        Returns the observation of half axis length d with corresponding variance v, each as 2D arrays.
        """
        n = len(Z)
        if n < 3:
            raise ValueError("IAE half axis observation cant estimate anything for n<3")

        # spread matrix of measurements
        Z_spread = Z - np.average(Z, axis=0).reshape((-1, 2))
        Z_spread = (Z_spread.T @ Z_spread) / (n - 1)

        # calculation of eigenvalues - note that we use the rescaled version Z
        w, V = np.linalg.eig(Z_spread * (1 / self.c_scaling))

        # ---
        # [Kolja Thormann]
        # Code to check if switching eigenvalues is necessary
        alpha = np.arctan2(self.x[3], self.x[2])
        eig0_or_diff = np.minimum(abs(((np.arctan2(V[1, 0], V[0, 0]) - alpha) + np.pi) % (2 * np.pi) - np.pi),
                                  abs(((np.arctan2(-V[1, 0], -V[0, 0]) - alpha) + np.pi) % (2 * np.pi) - np.pi))
        eig1_or_diff = np.minimum(abs(((np.arctan2(V[1, 1], V[0, 1]) - alpha) + np.pi) % (2 * np.pi) - np.pi),
                                  abs(((np.arctan2(-V[1, 1], -V[0, 1]) - alpha) + np.pi) % (2 * np.pi) - np.pi))
        if eig0_or_diff > eig1_or_diff:  # switch eigenvalues to make R==V assumption possible
            eig_save = w[0]
            w[0] = w[1]
            w[1] = eig_save
        # ---

        # approx V by rot based on velocity
        V = Rot(np.arctan2(x_plus[3], x_plus[2]))
        K = (1 / self.c_scaling) * (V.T @ R_k @ V)
        k = np.diag(K)

        subtracted_noise = np.array(w - k)

        # TODO: eigenvalues are sometimes smaller than noise to be subtracted, setting those entries to 1e-2 before sqrt
        subtracted_noise[subtracted_noise < 1e-2] = 1e-2

        d = np.sqrt(subtracted_noise)
        v = ((d ** 2 + k) ** 2) / (2 * (n - 1) * d ** 2)
        return d, v

