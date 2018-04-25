import copy

import cv2
import numpy as np

from Circle_Detection.circle_crop import CircleCrop
from Shrimp import CX, CY
from TrackWindow import TrackWindow
from crop import Crop
from detector import Detector
from tracer import TracerCSV
from tracker import Tracker
from utils import generate_color_set, side_by_side

PAUSE_KEY = ord(u'\r')
FRAME_BY_FRAME_KEY = ord(u' ')
ESC_KEY = 27

DISPLAY_ORIGINAL = True


def resizeFrame(frame, resize):
    return cv2.resize(frame, (0, 0), fx=resize, fy=resize) if resize is not None else frame


def main(filename, circle, resize=None, kalman=None, output_CSV_name=None):
    if len(filename) == 0:
        raise ValueError('Filename is empty')

    if circle is None or len(circle) != 3:
        raise ValueError('Circle is invalid.')

    cap = cv2.VideoCapture(filename)

    detector = Detector(minimum_area=100, maximum_area=500, debug=False)
    tracker = Tracker(dist_thresh=1000, max_frames_to_skip=30, max_trace_length=5, observation_matrix=kalman,
                      tracer=TracerCSV(output_CSV_path=output_CSV_name))

    pause = True

    # Infinite loop to process video frames
    tracks_window = TrackWindow()

    # Generate colors
    colors = generate_color_set(20)
    import random
    random.shuffle(colors)

    while True:
        tracks_window.reset()
        # Capture frame-by-frame (and resize)
        ret, frame = cap.read()
        if not ret:
            break
        frame = resizeFrame(frame, resize)

        uncropped_frame = copy.copy(frame)
        # Crop to the circle and add black pixels
        frame = CircleCrop.crop_circle(frame, circle)

        # Make copy of original frame
        orig_frame = copy.copy(frame)
        frame = CircleCrop.value_around_circle(frame)

        contours = detector.detect(frame)

        tracker.update(contours)

        # Display the original frame
        if DISPLAY_ORIGINAL: cv2.imshow('Original', uncropped_frame)

        for shrimp in tracker.tracks:
            color = colors[shrimp.id % len(colors)] + (1.0 / (1 + shrimp.skipped_frames),)
            trace = shrimp.trace(5)
            if (len(trace) > 1):
                for j in range(trace.shape[0] - 1):
                    # Draw trace line
                    x1 = trace.iloc[j, CX]
                    y1 = trace.iloc[j, CY]
                    x2 = trace.iloc[j + 1, CX]
                    y2 = trace.iloc[j + 1, CY]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Display the resulting tracking frame
            cropped, rect = Crop.crop_around_shrimp(copy.copy(orig_frame), shrimp)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, color, 2)
            cv2.ellipse(frame, center=shrimp.center, axes=shrimp.accuracy(), angle=0, startAngle=0, endAngle=360,
                        color=color)
            tracks_window.update_shrimp(cropped, shrimp.id, color)

        tracking_image = tracks_window.image(height=frame.shape[0])
        if tracking_image is None:
            cv2.imshow('Tracking', frame)
        else:
            cv2.imshow('Tracking', side_by_side(frame, tracking_image, separator_line_width=1))

        k = cv2.waitKey(10) & 0xff
        if k == PAUSE_KEY:
            pause = not pause
        if (pause is True):
            print("Code is paused.")
            while (pause is True):
                # stay in this loop until
                k = cv2.waitKey(30) & 0xff
                if k == PAUSE_KEY:
                    pause = False
                    print("Resume code")
                    break
                if k == FRAME_BY_FRAME_KEY:
                    pause = True
                    print("Resume code for one frame")
                    break
                if k == ESC_KEY:
                    break
        if k == ESC_KEY:  # 'esc' key has been pressed, exit program.
            break
    tracker.write()
    cv2.destroyAllWindows()
    cap.release()
