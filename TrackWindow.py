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
        Ig=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        shrimptype='0'
        if False:
            m = cv2.moments(255-Ig, False)
            # print("%d,%.3e"%(shrimp.id,m['nu21']))
            # print("%d,%d,%d,%s"%(shrimp.id,Ig.shape[0],Ig.shape[1],",".join(["%.2e"%(x) for x in [
            #     m['nu20'],m['nu11'],m['nu02'],
            #     m['nu30'],m['nu21'],m['nu12'],m['nu03']]])))
            if m['nu21'] > 1e-7:
                shrimptype='+'
            elif m['nu21'] < -1e-7:
                shrimptype='-'
        else:
            th,It=cv2.threshold(Ig,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            It = cv2.morphologyEx(It, cv2.MORPH_OPEN, np.ones((3,3)))
            h,w=It.shape
            It2 = np.ones((h+2,w+2),dtype=np.uint8)*255
            It2[1:1+h,1:1+w] = It
            _, cc, _ = cv2.findContours(It2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(cc)>1:
                m = [cv2.moments(c, True) for c in cc]
                sm = sorted(zip(cc,m),key=lambda x: x[1]['m00'],reverse=True)
                if sm[1][1]['nu21'] > 8e-3:
                    shrimptype='+'
                elif sm[1][1]['nu21'] < -8e-3:
                    shrimptype='-'
                #Itc=cv2.cvtColor(It2,cv2.COLOR_GRAY2RGB)
                #cv2.drawContours(Itc,[sm[1][0]],0,color,1)
                # print("%d,%.6f"%(shrimp.id,sm[1][1]['nu21']))
                # print("%d,%s"%(shrimp.id,",".join(["%.5f"%x for x in [
                #     sm[1][1]['nu20'],sm[1][1]['nu11'],sm[1][1]['nu02'],
                #     sm[1][1]['nu30'],sm[1][1]['nu21'],sm[1][1]['nu12'],sm[1][1]['nu03']]])))
        if shrimptype=='-':
            image = cv2.flip(image, 0)
        cv2.putText(image, shrimptype, (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
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
