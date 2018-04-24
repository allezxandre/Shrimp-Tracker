import copy
import time

import cv2
import numpy as np

from Circle_Detection.circle_crop import CircleCrop
from TrackWindow import TrackWindow
from crop import Crop
from detector import Detector
from tracer import TracerPlot, TracerCSV
from tracker import Tracker, CX, CY
from utils import generate_color_set, side_by_side

PAUSE_KEY = ord('p')
FRAME_BY_FRAME_KEY = ord(' ')
ESC_KEY = 27

DISPLAY_ORIGINAL = True


def resizeFrame(frame, resize):
    return cv2.resize(frame, (0, 0), fx=resize, fy=resize) if resize is not None else frame


def main(filename, resize=None, circle=None, kalman=None):
    if len(filename) == 0:
        raise ValueError('Filename is empty')

    if circle is None:
        # Detect circle
        print("Looking for circle")
        _, circle = CircleCrop.find_circle(filename, resize=resize)
        print("Found circle.")
    print(circle)

    cap = cv2.VideoCapture(filename)

    detector = Detector(minimum_area=100, maximum_area=500, debug=False)
    tracker = Tracker(dist_thresh=1000, max_frames_to_skip=30, max_trace_length=5, observation_matrix=kalman, tracer=TracerCSV())

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


        vectors = detector.detect(frame)

        tracker.update(vectors)

        # Display the original frame
        if DISPLAY_ORIGINAL: cv2.imshow('Original', uncropped_frame)

        for i in range(len(tracker.tracks)):
            shrimp = tracker.tracks[i]
            color = colors[shrimp.id % len(colors)] + (1.0 / (1 + shrimp.skipped_frames),)
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace) - 1):
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][CX]
                    y1 = tracker.tracks[i].trace[j][CY]
                    x2 = tracker.tracks[i].trace[j + 1][CX]
                    y2 = tracker.tracks[i].trace[j + 1][CY]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Display the resulting tracking frame
            cropped, rect = Crop.crop_around_shrimp(copy.copy(orig_frame), tracker.tracks[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, color, 2)
            cv2.ellipse(frame, center=shrimp.center, axes=shrimp.accuracy(), angle=0, startAngle=0, endAngle=360,
                        color=color)
            tracks_window.update_shrimp(cropped, tracker.tracks[i].id, color)

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
    tracker.tracer.write()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    # main(filename='../Resources/clip2.mp4', resize=0.7, circle=(667, 377, 274))
    main(filename='/Users/alexandre/PycharmProjects/OpenCV-Experiments/Resources/clip6.mp4', resize=0.7, circle=(703, 361, 325))
    #CircleCrop.find_circle('/Users/alexandre/PycharmProjects/OpenCV-Experiments/Resources/clip6.mp4',resize=0.7)
