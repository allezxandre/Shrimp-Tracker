import numpy as np
from scipy.optimize import linear_sum_assignment

from DecisionTree import DecisionTree
from Shrimp import Shrimp
from tracer import Tracer


class Tracker:

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, observation_matrix, tracer: Tracer):
        """
        Tracker object.

        :param dist_thresh: distance threshold. When distance exceeds this threshold,
                            tracks are deleted and a new track is created instead.
        :param max_frames_to_skip: maximum allowed frames to be skipped for
                                    the track object undetected
        :param max_trace_length: trace path history length
        """
        self.observation_matrix = observation_matrix
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = 1
        self.decision = DecisionTree(center_threshold=70, area_threshold=2,
                                     angle_threshold=np.math.pi / 2, body_ratio_range=(1 / 3, 1.5))
        self.nb_frame = 0
        self.tracer = tracer

    def increment_frame(self):
        self.nb_frame += 1
        for track in self.tracks: track.new_frame()

    def write(self):
        for i, shrimp in enumerate(self.tracks):
            self.tracer.add(shrimp.trace_df)
            del self.tracks[i]
        self.tracer.write()


    def update(self, contours):
        """
        Updates the internal tracking state.

        :param contours: List of contours [[]]
        """
        self.increment_frame()
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(contours)):
                track = Shrimp(contours[i], self.trackIdCount, self.nb_frame, observation_matrix=self.observation_matrix)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost
        n, m = len(self.tracks), len(contours)
        cost = np.zeros(shape=(n, m))  # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(contours)):
                predicted = self.tracks[i].state
                detected = contours[j]
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
                detected = contours[j]
                if (not self.decision.matches(predicted, detected)):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for idx in del_tracks:
                if idx < len(self.tracks):
                    self.tracer.add(self.tracks[idx].trace_df)
                    del self.tracks[idx]
                    del assignment[idx]
                else:
                    print("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(contours)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Shrimp(contours[un_assigned_detects[i]], self.trackIdCount, self.nb_frame,
                               observation_matrix=self.observation_matrix)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].predict()

            if (assignment[i] != -1):
                self.tracks[i].correct(contours[assignment[i]], flag=True)

            self.tracks[i].save()
