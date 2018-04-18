import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanFilter
from kalman_filter_acceleration import KalmanFilterConstantAcceleration
from tracer import Tracer


class Shrimp:
    def __init__(self, prediction, id, tracer: Tracer, kalman):
        self._KF = KalmanFilter(np.asarray(prediction), kalman)
        self.id = id
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self._tracer = tracer
        self._tracer.register(self.state, self.id)

    def predict(self):
        return self._KF.predict()

    def accuracy(self):
        return tuple(map(int, self._KF.accuracy()))

    def save(self):
        self._tracer.add(self.state, self.id)

    @property
    def center(self):
        return int(self.state[CX]), int(self.state[CY])

    def correct(self, z, flag=True):
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

    def plot(self):
        self._KF.plot()


CX, CY, ANGLE, AREA, LAMBDA1, LAMBDA2 = 0, 1, 2, 3, 4, 5

class DecisionTree(object):
    def __init__(self, center_threshold, area_threshold, angle_threshold, body_ratio_range):
        self.center_threshold = center_threshold
        self.area_threshold = area_threshold
        self.body_ratio_range = body_ratio_range
        self.angle_threshold = angle_threshold

    def area_error(self, v1, v2):
        return abs(v2[AREA] / v1[AREA])

    def centroids_error(self, v1, v2):
        cx1, cy1 = v1[CX], v1[CY]
        cx2, cy2 = v2[CX], v2[CY]
        return np.linalg.norm(np.vstack((cx1, cy1)) - np.vstack((cx2, cy2)))

    def angle_error(self, v1, v2):
        angle1, angle2 = v1[ANGLE], v2[ANGLE]
        return abs(angle1 - angle2)

    def body_shape_error(self, v1, v2):
        lambda1_1, lambda2_1 = v1[LAMBDA1], v1[LAMBDA2]
        lambda1_2, lambda2_2 = v2[LAMBDA1], v2[LAMBDA2]
        ratio1 = lambda1_1 / lambda2_1
        ratio2 = lambda1_2 / lambda2_2
        # FIXME: Handle divisions by zero
        return abs(ratio2 / ratio1)

    def distance(self, v1, v2):
        return self.centroids_error(v1, v2)

    def matches(self, v1, v2):
        if self.centroids_error(v1, v2) < self.center_threshold:
            if 1 / self.area_threshold < self.area_error(v1, v2) < self.area_threshold:
                return True
        return False

class Tracker:

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, kalman, tracer=Tracer()):
        """
        Tracker object.

        :param dist_thresh: distance threshold. When distance exceeds this threshold,
                            tracks are deleted and a new track is created instead.
        :param max_frames_to_skip: maximum allowed frames to be skipped for
                                    the track object undetected
        :param max_trace_length: trace path history length
        """
        self.kalman=kalman
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = 1
        self.decision = DecisionTree(center_threshold=70, area_threshold=2,
                                     angle_threshold=np.math.pi / 2, body_ratio_range=(1 / 3, 1.5))
        self.tracer = tracer

    def update(self, vectors):
        """
        Updates the internal tracking state.

        :param vectors: List of contours [[]]
        """
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(vectors)):
                track = Shrimp(vectors[i], self.trackIdCount, tracer=self.tracer, kalman=self.kalman)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost
        n, m = len(self.tracks), len(vectors)
        cost = np.zeros(shape=(n, m))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(vectors)):
                predicted = self.tracks[i].state
                detected = vectors[j]
                cost[i][j] = self.decision.distance(detected, predicted)

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = np.full(n, -1).tolist()
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                j = assignment[i]
                predicted = self.tracks[i].state
                detected = vectors[j]
                if (not self.decision.matches(predicted, detected)):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(vectors)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if (len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Shrimp(vectors[un_assigned_detects[i]],
                               self.trackIdCount, tracer=self.tracer)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].predict()
            self.tracks[i].save()

            if (assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                # FIXME: Angle workaround
                previous_angle = self.tracks[i].angle
                self.tracks[i].correct(vectors[assignment[i]], flag=True)

            if (len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].state)
            self.tracks[i].save()
