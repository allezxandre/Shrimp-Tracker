import numpy as np
import pandas as pd

from kalman_filter import KalmanFilter

CX, CY, ANGLE, AREA, LAMBDA1, LAMBDA2 = 0, 1, 2, 3, 4, 5


class Shrimp:

    def __init__(self, prediction, id, frame_id, observation_matrix):
        self._KF = KalmanFilter(np.asarray(prediction), observation_matrix)
        self.id = id
        self.__skipped_frames = 0  # number of frames skipped undetected
        self.__nb_frames = frame_id  # The index of the frame
        trace, idx = self.__trace_from_state()
        self._trace = pd.DataFrame(trace, index=[idx])

    def predict(self):
        return self._KF.predict()

    def new_frame(self):
        self.__nb_frames += 1
        self.__skipped_frames += 1

    def accuracy(self):
        return tuple(map(int, self._KF.accuracy()))

    def save(self):
        trace, idx = self.__trace_from_state()
        self._trace.loc[idx, trace.keys()] = list(trace.values())
        pass

    def __trace_from_state(self):
        cx, cy, angle, area, lambda1, lambda2, vel, angular_vel = self.state
        p_cx, p_cy, p_angle, p_area, p_lambda1, p_lambda2, p_vel, p_angular_vel = self._KF.uncertainty
        return {
            "Track ID": self.id,
            "Timestep": self.__nb_frames,
            "Skipped Frames": self.__skipped_frames,
            "Lost track": self.__skipped_frames > 0,
            "CX": cx,
            "CY": cy,
            "ANGLE": angle,
            "AREA": area,
            "LAMBDA1": lambda1,
            "LAMBDA2": lambda2,
            "VEL": vel,
            "ANGULAR_VEL": angular_vel,
            "CX Uncertainty": p_cx,
            "CY Uncertainty": p_cy,
            "ANGLE Uncertainty": p_angle,
            "AREA Uncertainty": p_area,
            "LAMBDA1 Uncertainty": p_lambda1,
            "LAMBDA2 Uncertainty": p_lambda2,
            "VEL Uncertainty": p_vel,
            "ANGULAR_VEL Uncertainty": p_angular_vel,
        }, self.__nb_frames

    @property
    def trace_df(self):
        return self._trace

    def trace(self, max_trace_length):
        length = min(max_trace_length, self._trace.shape[0])
        return self._trace[["CX", "CY", "ANGLE", "AREA", "LAMBDA1", "LAMBDA2", "VEL", "ANGULAR_VEL"]][-length:]


    @property
    def center(self):
        return int(self.state[CX]), int(self.state[CY])

    @property
    def skipped_frames(self):
        return self.__skipped_frames

    def correct(self, z, flag=True):
        self.__skipped_frames = 0
        return self._KF.correct(z, flag=False)

    @property
    def state(self):
        return self._KF.state

    @property
    def lambda1(self):
        return self.state[LAMBDA1]

    @property
    def lambda2(self):
        return self.state[LAMBDA2]

    @property
    def cx(self):
        return self.state[CX]

    @property
    def cy(self):
        return self.state[CY]

    @property
    def angle(self):
        return self.state[ANGLE]
