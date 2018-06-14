import copy

import cv2
import numpy as np
import os.path

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

DISPLAY_ORIGINAL = False


def resizeFrame(frame, resize):
    return cv2.resize(frame, (0, 0), fx=resize, fy=resize) if resize is not None else frame


def main(filename, circle, resize=None, kalman=None, output_CSV_name=None):
    if len(filename) == 0:
        raise ValueError('Filename is empty')

    if circle is None or len(circle) != 3:
        raise ValueError('Circle is invalid.')

    avi = None


    detector = Detector(minimum_area=100, maximum_area=600, debug=False)
    tracker = Tracker(dist_thresh=1000, max_frames_to_skip=30, max_trace_length=5, observation_matrix=kalman,
                      tracer=TracerCSV(output_CSV_path=output_CSV_name))

    pause = True

    # Infinite loop to process video frames
    tracks_window = TrackWindow()

    # Generate colors
    colors = generate_color_set(20)
    import random
    random.shuffle(colors)

    # First build a mask of static pixels
    cap = cv2.VideoCapture(filename)
    detector.reset_mask()
    frame_count = 0
    while True:
        # Capture frame-by-frame (and resize)
        ret, frame = cap.read()
        if not ret:
            break
        frame = resizeFrame(frame, resize)
        # Crop to the circle and add black pixels
        frame = CircleCrop.crop_circle(frame, circle)
        # Make copy of original frame
        frame = CircleCrop.value_around_circle(frame, None)
        detector.update_mask(frame)
        frame_count += 1
    detector.finalize_mask(max(1,(10*frame_count)/100))

    cap = cv2.VideoCapture(filename)
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
        frame = CircleCrop.value_around_circle(frame, None)

        contours = detector.detect(frame)

        tracker.update(contours)

        # Display the original frame
        if DISPLAY_ORIGINAL: 
            cv2.circle(uncropped_frame,(circle[0],circle[1]),circle[2],(0,0,255), 3)
            cv2.imshow('Original', uncropped_frame)

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
            size = rect[1][0]*rect[1][1]
            accuracy = shrimp.accuracy()
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if accuracy[0]*accuracy[1] < size:
                cv2.drawContours(frame, [box], 0, color, 2)
                cv2.ellipse(frame, center=shrimp.center, axes=accuracy, angle=0, startAngle=0, endAngle=360,
                        color=color)
                tracks_window.update_shrimp(cropped, shrimp.id, color)

        tracking_image = tracks_window.image(height=frame.shape[0])
        if avi is None:
            fileout,ext=os.path.splitext(filename)
            avi = cv2.VideoWriter(fileout+"_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, frame.shape[0:2])
        avi.write(frame)
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
    print("Shrimp tracking completed")
    tracker.write()
    cv2.destroyAllWindows()
    cap.release()
    if not avi is None:
        avi.release()
