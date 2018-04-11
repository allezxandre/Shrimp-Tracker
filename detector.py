import math

import cv2
import numpy as np

from Circle_Detection.circle_crop import CircleCrop
from Shrimp_Tracker.utils import side_by_side


class Detector(object):
    def __init__(self, minimum_area, maximum_area, kernel=np.ones((3, 3)), circle=None, debug=False):
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area

        self.kernel = kernel
        if self.kernel is not None:
            self.kernel = self.kernel.astype(np.uint8)

        self.debug = debug
        self._circle = circle

    @staticmethod
    def compute_lambdas(moments):
        cov_xx = moments['mu20'] / moments['m00']
        cov_xy = moments['mu11'] / moments['m00']
        cov_yy = moments['mu02'] / moments['m00']
        T = cov_xx + cov_yy
        D = cov_xx * cov_yy - cov_xy * cov_xy
        delta = T * T / 4 - D
        assert (delta > -1e-8);
        if abs(delta) <= 1e-8:
            delta = 0
        lambda1 = T / 2 + math.sqrt(delta)
        lambda2 = T / 2 - math.sqrt(delta)
        return lambda1, lambda2

    def circle(self, frame):
        if self._circle is None:
            diameter = min(frame.shape[:2])
            circle = (frame.shape[0] // 2, frame.shape[1] // 2, diameter // 2)
            self._circle = circle
        return self._circle

    def detect(self, frame):
        """
        Returns contours in the frame

        :param frame: Frame to process
        :return: [cx, cy, angle, area, lambda1, lambda2] vectors found in the frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Clean the image
        threshold, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ret, frame = cv2.threshold(frame, 120, 255, 0)
        # Remove the circle
        frame_before = CircleCrop.value_around_circle(frame, value=0, circle=self.circle(frame))
        # Erode/dilate
        if self.kernel is not None:
            frame = cv2.morphologyEx(frame_before, cv2.MORPH_OPEN, self.kernel)
        else:
            frame = frame_before
        if self.debug:
            cv2.imshow("Threshold", side_by_side(frame_before, frame, separator_line_width=3))

        # Find contours
        vectors = []
        _, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            m = cv2.moments(cnt, True)
            area = m['m00']
            if self.maximum_area > area > self.minimum_area:
                cov_xx1 = m['mu20'] / m['m00']
                cov_xy1 = m['mu11'] / m['m00']
                cov_yy1 = m['mu02'] / m['m00']
                angle = 0.5 * math.atan2(2 * cov_xy1, (cov_xx1 - cov_yy1)) + np.pi / 2
                cx = m['m10'] / m['m00']
                cy = m['m01'] / m['m00']
                lambda1, lambda2 = Detector.compute_lambdas(m)
                v = np.array([cx, cy, angle, area, lambda1, lambda2])
                vectors.append(v)
        return vectors
