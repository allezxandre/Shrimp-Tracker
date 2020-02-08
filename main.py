import copy

import cv2
import numpy as np
import os.path
import math

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

DISPLAY_CONTOURS = True
DISPLAY_ORIGINAL = True
BUILD_MASK = True
SAVE_THUMBS = True


def resizeFrame(frame, resize):
    return cv2.resize(frame, (0, 0), fx=resize, fy=resize) if resize is not None else frame


def main(filename, settings, circle, resize=None, kalman=None, output_CSV_name=None):
    if len(filename) == 0:
        raise ValueError('Filename is empty')

    if not os.path.exists(filename):
        raise ValueError('Filename does not exist')

    if circle is None or len(circle) != 3:
        raise ValueError('Circle is invalid.')

    avi = None


    mask = None
    try:
        mask = settings.read_from_cache(filename).mask
    except AttributeError:
        pass
    detector = Detector(minimum_area=500, maximum_area=3500, 
            mask=mask, debug=False)
    tracker = Tracker(dist_thresh=1000, max_frames_to_skip=30, max_trace_length=5,
            observation_matrix=kalman, tracer=TracerCSV(output_CSV_path=output_CSV_name))

    pause = False

    # Infinite loop to process video frames
    tracks_window = TrackWindow()

    # Generate colors
    colors = generate_color_set(20)
    import random
    random.shuffle(colors)

    if BUILD_MASK and (detector.mask is None):
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
                frame = CircleCrop.value_around_circle(frame, 0)
                detector.update_mask(frame)
                frame_count += 1
                cv2.waitKey(10)
                if frame_count > 60:
                    break
            detector.finalize_mask(max(1,(10*frame_count)/100))
            mask = settings.add_to_cache(filename, mask=detector.mask)
    
    fout=open("paths.csv","w")
    cap = cv2.VideoCapture(filename)
    frame_count = 0
    while True:
        tracks_window.reset()
        # Capture frame-by-frame (and resize)
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = resizeFrame(frame, resize)

        uncropped_frame = copy.copy(frame)
        # Crop to the circle and add black pixels
        cropped_frame = CircleCrop.crop_circle(frame, circle)

        # Make copy of original frame
        frame = CircleCrop.value_around_circle(cropped_frame, 0, mask=detector.mask)
        orig_frame = copy.copy(frame)

        contours = detector.detect(frame)

        tracker.update(contours)

        # Display the original frame
        if DISPLAY_ORIGINAL: 
            cv2.circle(uncropped_frame,(circle[0],circle[1]),circle[2],(0,0,255), 3)
            cv2.imshow('Original', uncropped_frame)

        if DISPLAY_CONTOURS:
            for C in contours:
                r1=2*math.sqrt(C[4])
                r2=2*math.sqrt(C[5])
                cv2.line(cropped_frame,(int(C[0]),int(C[1])),
                        (int(C[0]+r1*math.cos(C[2])),int(C[1]+r1*math.sin(C[2]))),
                        (0,0,0),1) 
                cv2.ellipse(cropped_frame, center=(int(C[0]),int(C[1])), 
                        axes=(int(r1),int(r2)), 
                        angle=C[2]*180./math.pi, startAngle=0, endAngle=360,
                        color=(0,0,0))

        # print()
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
                    cv2.line(cropped_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Display the resulting tracking frame
            cropped, rect, box = Crop.crop_around_shrimp(copy.copy(orig_frame), shrimp)
            if SAVE_THUMBS:
                try:
                    os.makedirs("thumbs/%04d"%shrimp.id)
                except:
                    pass
                cv2.imwrite("thumbs/%04d/%04d.png"%(shrimp.id,frame_count),cropped)
            size = rect[1][0]*rect[1][1]
            accuracy = shrimp.accuracy()
            # box = cv2.boxPoints(rect)
            box = np.int0(box)
            if accuracy[0]*accuracy[1] < size:
                cv2.drawContours(cropped_frame, [box], 0, color, 2)
                
                cx,cy = shrimp.center
                fout.write("%d,%d,%e,%e\n"%(frame_count,shrimp.id,cx,cy))
                angle=shrimp.angle
                r1=2*math.sqrt(shrimp.lambda1)
                r2=2*math.sqrt(shrimp.lambda2)
                cv2.line(cropped_frame,(cx,cy),
                        (int(cx+r1*math.cos(angle)),int(cy+r1*math.sin(angle))),
                        color,1) 
                cv2.ellipse(cropped_frame, center=shrimp.center, axes=accuracy, angle=angle*180/math.pi, startAngle=0, endAngle=360,
                        color=color)
                tracks_window.update_shrimp(cropped, shrimp.id, color)

        fout.flush()
        tracking_image = tracks_window.image(height=cropped_frame.shape[0])
        if avi is None:
            fileout,ext=os.path.splitext(filename)
            avi = cv2.VideoWriter(fileout+"_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (cropped_frame.shape[0]+100,cropped_frame.shape[1]))
        if tracking_image is None:
            cv2.imshow('Tracking', cropped_frame)
        else:
            sbs = side_by_side(cropped_frame, tracking_image, separator_line_width=1)
            avi.write(cv2.resize(sbs,(cropped_frame.shape[0]+100,cropped_frame.shape[1])))
            cv2.imshow('Tracking', sbs)

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
    fout.close()
    print("Shrimp tracking completed")
    tracker.write()
    cv2.destroyAllWindows()
    cap.release()
    if not avi is None:
        avi.release()
