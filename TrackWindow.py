import cv2
import numpy as np


def resize(image, width, height):
    new_img = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    new_img[:image.shape[0], :image.shape[1], :] = image
    return new_img


class TrackWindow:
    def __init__(self, max_width=50, max_height=20):
        self.track_windows = {}
        self.colors = {}
        self.max_height = max_height
        self.max_width = max_width

    def update_shrimp(self, image, id, color):
        self.track_windows[id] = image
        self.colors[id] = color

    def reset(self):
        self.track_windows = {}
        self.colors = {}

    def image(self, height=None):
        """
        Generates a stacked image of all the track images. If height is not None,
        the returned image will be cropped/expanded to match the provided height.
        """
        labels = list(self.track_windows.keys())
        images = self.track_windows.values()
        if len(images) == 0:
            return None
        self.max_width = np.max([np.max([i.shape[1] for i in images]), self.max_width])
        self.max_height = np.max([np.max([i.shape[0] for i in images]), self.max_height])
        images = [resize(i, self.max_width, self.max_height) for i in images]
        # Add labels
        for i in range(len(images)):
            cv2.putText(images[i], str(labels[i]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, self.colors[labels[i]], 1,
                        cv2.LINE_AA)
        stack = np.vstack(images)
        if height is not None:
            # Cut or add zero to respect height
            if height < stack.shape[0]:
                # Cut
                stack = stack[:height, :, :]
            elif height > stack.shape[0]:
                # Add rows
                stack = resize(stack, stack.shape[1], height)

        return stack
