import numpy as np

from Shrimp import AREA, CX, CY, ANGLE, LAMBDA1, LAMBDA2


class DecisionTree(object):
    """
    This class determines whether or not a Shrimp is similar to another.
    """

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