import math

import cv2
import numpy as np

from Circle_Detection.circle_crop import CircleCrop
from utils import side_by_side


class Detector(object):
    def __init__(self, minimum_area, maximum_area, kernel=np.ones((3, 3)), circle=None, mask=None, debug=False):
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area

        self.kernel = kernel
        if self.kernel is not None:
            self.kernel = self.kernel.astype(np.uint8)

        self.debug = debug
        self._circle = circle
        self.mask = mask

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

    def reset_mask(self):
        self.mask = None

    def finalize_mask(self, mask_threshold):
        if not self.mask is None:
            mask_before = cv2.bitwise_not(255*(self.mask > mask_threshold).astype(np.uint8))
            # Erode/dilate
            #self.mask = cv2.morphologyEx(mask_before, cv2.MORPH_ERODE, np.ones((3,3)))
            #mask_before = self.mask
            self.mask = cv2.morphologyEx(mask_before, cv2.MORPH_DILATE, np.ones((3,3)))
            if self.debug:
                cv2.imshow("Mask", side_by_side(mask_before, self.mask, separator_line_width=3))

    def update_mask(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # ret, frame = cv2.threshold(frame, 120, 1, 0)
        # Remove the circle
        frame_before = CircleCrop.value_around_circle(frame, value=None, circle=self.circle(frame))
        # Erode/dilate
        if self.kernel is not None:
            frame = cv2.morphologyEx(frame_before, cv2.MORPH_OPEN, self.kernel)
        else:
            frame = frame_before
        cv2.imshow('Tracking', frame*255)
        if self.mask is None:
            self.mask = frame.astype(np.uint32)
        else:
            self.mask += frame

    def detect(self, frame):
        """
        Returns contours in the frame

        :param frame: Frame to process
        :return: [cx, cy, angle, area, lambda1, lambda2] vectors found in the frame
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _,_,frame_gray = cv2.split(frame)
        # Clean the image
        threshold, frame_thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # threshold, frame_thresh = cv2.threshold(frame_gray, 60, 255, cv2.THRESH_BINARY)
        if self.debug:
            cv2.imshow("Base Image", frame_gray )
            cv2.imshow("Base Threshold", frame_thresh )
        # Remove the circle
        frame_before = CircleCrop.value_around_circle(frame_thresh, value=0, circle=self.circle(frame_thresh))
        if self.mask is not None:
            frame_before = cv2.bitwise_not(cv2.bitwise_or(frame_before,self.mask))
        else:
            frame_before = cv2.bitwise_not(frame_before)
 
        # Erode/dilate
        if self.kernel is not None:
            frame = cv2.morphologyEx(frame_before, cv2.MORPH_OPEN, self.kernel)
        else:
            frame = frame_before

        sbs = cv2.cvtColor(side_by_side(frame_before, frame, separator_line_width=3),cv2.COLOR_GRAY2RGB)

        # Find contours
        vectors = []
        _, contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if self.debug:
            print("\n===================")
        for icnt,cnt in enumerate(contours):
            m = cv2.moments(cnt, True)
            area = m['m00']
            cx = m['m10'] / m['m00']
            cy = m['m01'] / m['m00']
            if self.debug:
                cv2.putText(sbs, str(icnt), (int(cx),int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
            if self.maximum_area > area > self.minimum_area:
                cov_xx1 = m['mu20'] / m['m00']
                cov_xy1 = m['mu11'] / m['m00']
                cov_yy1 = m['mu02'] / m['m00']
                angle = 0.5 * math.atan2(2 * cov_xy1, (cov_xx1 - cov_yy1)) 
                lambda1, lambda2 = Detector.compute_lambdas(m)
                r1=2*math.sqrt(lambda1)
                r2=2*math.sqrt(lambda2)
                if self.debug:
                    # print("%d,%.0f,%.0f,%.1f,%.1f,%.4f"%(icnt,area,math.pi*r1*r2,r1,r2,r2/r1))
                    cv2.ellipse(sbs, center=(int(cx),int(cy)), 
                            axes=(int(r1),int(r2)), 
                            angle=angle*180./math.pi, startAngle=0, endAngle=360, color=(0,0,255))
                if r2/r1 > 0.1 and math.pi*r1*r2/area < 2.0:
                    v = np.array([cx, cy, angle, area, lambda1, lambda2])
                    vectors.append(v)
            elif self.debug:
                #print("%d,%.0f"%(icnt,area))
                pass
        if self.debug:
            cv2.imshow("Threshold", sbs )
        return vectors
