# Import python libraries
import math
import numpy as np
from numpy.linalg import inv

CX, CY, ANGLE, AREA, LAMBDA1, LAMBDA2, VEL, ANGULAR_VEL = 0, 1, 2, 3, 4, 5, 6, 7


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality

    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self, first_observation, observation_matrix):
        """
        Initialize variable used by Kalman Filter class
        """
        self.dt = 1 / 24  # delta time

        self.x = np.append(np.asarray(first_observation), [0, 0])  # state vector
        self.P = np.diag((9.0, 9.0, np.pi, 10., 10., 10., 9., 9.))  # covariance matrix

        self.Q = np.diag((3.0, 3.0, np.pi / 6, 35., 30., 30., 5., 0.2))  # process noise matrix
        self.Q = self.Q ** 2

        self.R = observation_matrix  # observation noise matrix

        self.H = np.eye(6, 8)

    def F(self, x):
        # state transition mat
        x_t = x[CX]
        y_t = x[CY]
        cos_t = np.cos(x[ANGLE])
        sin_t = np.sin(x[ANGLE])
        v_t = x[VEL]
        #                CX,   CY, ANGLE, AREA, LAMBDA1, LAMBDA2        VEL,      ANGULAR_VEL
        return np.array([x_t + v_t * cos_t * self.dt,
            y_t + v_t * sin_t * self.dt,
            x[ANGLE] + x[ANGULAR_VEL] * self.dt,
            x[AREA], x[LAMBDA1], x[LAMBDA2], x[VEL], x[ANGULAR_VEL]]).T
    
    def dF_dX(self, x):
        # state transition mat
        cos_t = np.cos(x[ANGLE])
        sin_t = np.sin(x[ANGLE])
        v_t = x[VEL]
        #                CX,   CY, ANGLE, AREA, LAMBDA1, LAMBDA2        VEL,      ANGULAR_VEL
        return np.eye(x.shape[0]) + \
               np.array([[0, 0, -sin_t * v_t * self.dt, 0, 0, 0, cos_t * self.dt, 0],
                         [0, 0, cos_t * v_t * self.dt, 0, 0, 0, sin_t * self.dt, 0],
                         [0, 0, 0, 0, 0, 0, 0, self.dt],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         ])

    @property
    def state(self):
        return self.x

    @property
    def uncertainty(self):
        return np.diag(self.P)

    def predict(self):
        dF_dX = self.dF_dX(self.x)
        self.x = np.asarray(self.F(self.x))
        self.P = np.asarray(dF_dX @ self.P @ dF_dX.T + self.Q)
        self.x[ANGLE] %= 2*np.pi
        return self.x

    def accuracy(self):
        return self.P[CX, CX], self.P[CY, CY]

    def correct(self, z, flag):
        y = z - np.dot(self.H, self.x)
        # angle_t = y[ANGLE]
        y[ANGLE] = math.fmod(y[ANGLE]+5*np.pi/2,np.pi)-np.pi/2
        # print("%.1f,%.1f,%.1f,%.1f,%.1f"%(self.x[2]*180/np.pi,z[2]*180/np.pi,angle_t*180/np.pi,y[2]*180/np.pi,self.x[VEL]))
        # print("%.1f,%.1f,%.1f,%.1f"%(self.x[LAMBDA1],self.x[LAMBDA2],z[LAMBDA1],z[LAMBDA2]))
        S = self.R + self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ inv(S)
        self.x = np.asarray(self.x + K @ y)
        if self.x[VEL] < -10:
            self.x[VEL] = -self.x[VEL]
            self.x[ANGLE] = self.x[ANGLE] + np.pi
        self.x[ANGLE] %= 2*np.pi
        I_KH = (np.eye(8) - K @ self.H)
        self.P = np.asarray(I_KH @ self.P @ I_KH.T + K @ self.R @ K.T)
        return self.state
