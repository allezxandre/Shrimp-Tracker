import cv2
import numpy as np
import multiprocessing as mp


def find_circles_in_frame(frame, resize, minDist):
    if resize is not None:
        frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

    # convert to greyscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find circles
    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 2, minDist=minDist, param1=255, param2=300)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circles = circles[circles[:, 2].argsort()]
        return circles[-1]
    else:
        return None

class CircleCrop:

    @staticmethod
    def crop_circle(frame, circle):
        x, y, r = circle
        return frame[(y - r):(y + r), (x - r):(x + r)]

    @staticmethod
    def value_around_circle(frame: np.ndarray, value=0, circle=None):
        if circle is None:
            cx, cy, radius = frame.shape[0] // 2, frame.shape[1] // 2, frame.shape[1] // 2 - 2
        else:
            cx, cy, radius = circle
        y, x = np.ogrid[0:frame.shape[0], 0:frame.shape[1]]
        index = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        if value is None:
            value = np.mean(frame[index])
        new_frame = np.full(frame.shape[:3], value, dtype=frame.dtype)
        new_frame[index] = frame[index]
        return new_frame

    @staticmethod
    def hough_circle(input='../Resources/clip6.mp4'):
        # Read image
        cap = cv2.VideoCapture(input)

        centers = np.empty((0, 3), int)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

            # convert to greyscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 2, 350, param1=255, param2=300)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circles = circles[circles[:, 2].argsort()]
                (x, y, r) = circles[-1]
                centers = np.append(centers, np.array([[x, y, r]]), axis=0)

            # center = np.median(centers, [0]).astype("int")
            circle = centers[centers[:, 2].argmin()]
            centers = np.append(centers, np.array([[x, y, r]]), axis=0)

            frame = CircleCrop.crop_circle(frame, circle)
            frame = CircleCrop.value_around_circle(frame, None)

            cv2.imshow(winname="Final result", mat=frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    @staticmethod
    def find_circle(input='../Resources/clip6.mp4', minDist=350, resize=None):
        """
        Goes through all the frames of the input video to find
        the biggest circles. Returns the median and the minimum of these
        circles.
        """
        cap = cv2.VideoCapture(input)

        ret, frame = cap.read()
        frames = []
        while ret:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        # frames created

        pool = mp.Pool()
        centers = np.empty((0, 3), int)
        #circles_per_frame = pool.starmap(find_circles_in_frame, ((frame, resize, minDist) for frame in frames))
        circles_per_frame = [find_circles_in_frame(frame, resize, minDist) for frame in frames]

        for (x, y, r) in circles_per_frame:
            # Add circle to list
            centers = np.append(centers, np.array([[x, y, r]]), axis=0)

        return np.median(centers, [0]).astype("int"), centers[centers[:, 2].argmin()]

    if __name__ == '__main__':
        hough_circle()
