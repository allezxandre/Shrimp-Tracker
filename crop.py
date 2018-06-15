import math

import cv2
import numpy as np

from tracker import Shrimp


class Crop(object):
    @staticmethod
    def crop_around_shrimp(frame, shrimp: Shrimp):
        length = 4 * math.sqrt(shrimp.lambda1) + 8
        thickness = 4 * math.sqrt(shrimp.lambda2) + 8
        cy = shrimp.cy
        cx = shrimp.cx
        angle = shrimp.angle 

        C = np.array([cx,cy])
        u = np.array([length/2 * math.cos(angle), length/2 * math.sin(angle)])
        v = np.array([-thickness/2 * math.sin(angle), thickness/2 * math.cos(angle)])

        # p1 = np.array([cx + length / 2 * math.cos(angle), cy + length / 2 * math.sin(angle)])
        # p2 = np.array([cx - length / 2 * math.sin(angle), cy - length / 2 * math.cos(angle)])
        # p3 = np.array([cx - length / 2 * math.cos(angle), cy - length / 2 * math.sin(angle)])
        # p4 = np.array([cx + length / 2 * math.sin(angle), cy + length / 2 * math.cos(angle)])

        # corner1 = np.array(p4) + np.array([length / 2 * math.cos(angle), length / 2 * math.sin(angle)])
        # corner2 = np.array(p2) + np.array([length / 2 * math.cos(angle), length / 2 * math.sin(angle)])
        # corner3 = np.array(p2) + np.array([-length / 2 * math.cos(angle), -length / 2 * math.sin(angle)])
        # corner4 = np.array(p4) + np.array([-length / 2 * math.cos(angle), -length / 2 * math.sin(angle)])

        corner1 = C + u + v
        corner2 = C - u + v
        corner3 = C - u - v
        corner4 = C + u - v

        box = [corner1, corner2, corner3, corner4]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs) - 5
        x2 = max(Xs) + 5
        y1 = min(Ys) - 5
        y2 = max(Ys) + 5


        # Center of rectangle in source image
        # center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # rect = (center, (int(abs(x1 - x2)), int(abs(y1 - y2))), angle * 180 / math.pi)
        center = (int(cx),int(cy))
        rect = (center, (int(length), int(thickness)), angle * 180 / math.pi)
        # Size of the upright rectangle bounding the rotated rectangle
        size = (int(np.ceil(x2 - x1)), int(np.ceil(y2 - y1)))

        M = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle * 180 / math.pi, 1)
        # Cropped upright rectangle
        cropped = cv2.getRectSubPix(frame, size, center)
        # Rotate and crop
        cropped = cv2.warpAffine(cropped, M, size)
        croppedW = (thickness if thickness > length else length) + 5
        croppedH = (thickness if thickness < length else length) + 5
        # Final cropped & rotated rectangle
        return cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2)), rect, box
