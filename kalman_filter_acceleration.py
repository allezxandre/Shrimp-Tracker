# Import python libraries
import numpy as np
from numpy.linalg import inv

CX, CY, ANGLE, AREA, LAMBDA1, LAMBDA2, VEL, ANGULAR_VEL, ACCELERATION = 0, 1, 2, 3, 4, 5, 6, 7, 8


class KalmanFilterConstantAcceleration(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality

    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self, first_observation):
        """
        Initialize variable used by Kalman Filter class
        """
        self.dt = 1 / 24  # delta time

        self.x = np.append(np.asarray(first_observation), [0, 0, 0])  # state vector
        self.P = np.diag((9.0, 9.0, np.math.pi, 10., 10., 10., 9., 9., 10.))  # covariance matrix

        self.Q = np.diag((3.0, 3.0, np.math.pi / 6, 35., 30., 5., 5., 0.2, 10.))  # process noise matrix
        self.Q = self.Q ** 2
        self.R = np.diag((3.0, 3.0, np.math.pi / 6, 10., 10, 5.))  # observation noise matrix

        self.H = np.eye(6, 9)

    @property
    def F(self):
        # state transition mat
        cos_t = np.cos(self.x[ANGLE])
        sin_t = np.sin(self.x[ANGLE])
        v = self.x[VEL]
        a = self.x[ACCELERATION]
        #                CX,   CY, ANGLE, AREA, LAMBDA1, LAMBDA2        VEL,      ANGULAR_VEL
        return np.array([[1, 0, -sin_t * v * self.dt - sin_t * a * 0.5 * self.dt * self.dt, 0, 0, 0, cos_t * self.dt, 0,
                          0.5 * cos_t * self.dt * self.dt],
                         [0, 1, cos_t * v * self.dt + cos_t * a * 0.5 * self.dt * self.dt, 0, 0, 0, sin_t * self.dt, 0,
                          0.5 * sin_t * self.dt * self.dt],
                         [0, 0, 1, 0, 0, 0, 0, self.dt, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, self.dt],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         ])

    @property
    def state(self):
        return self.x

    def predict(self):
        F = self.F
        self.x = np.asarray(F @ self.x)
        self.P = np.asarray(F @ self.P @ F.T + self.Q)
        self.x[ANGLE] %= np.pi
        return self.x

    def accuracy(self):
        return self.P[CX, CX], self.P[CY, CY]

    def correct(self, z, flag):
        y = z - np.dot(self.H, self.x)
        S = self.R + self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ inv(S)
        self.x = np.asarray(self.x + K @ y)
        self.x[ANGLE] %= np.pi
        I_KH = (np.eye(9) - K @ self.H)
        self.P = np.asarray(I_KH @ self.P @ I_KH.T + K @ self.R @ K.T)
        return self.state
