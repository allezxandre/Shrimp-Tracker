import cv2
import numpy as np
import math


def pradalier(image):

    I = cv2.imread(image);

    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    circles = np.zeros((1, 10, 3));
    if cvmajor == 3:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, \
                                   2, 1000, circles, 40, 100, approx_radius - 20, approx_radius + 20)
    else:
        circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, \
                                   2, 1000, circles, 40, 100, approx_radius - 20, approx_radius + 20)

    if False:
        C = cv2.Canny(gray, 120, 60)
        Cg = C.copy()
        for i in range(circles.shape[1]):
            cv2.circle(Cg, (circles[0, i, 0], circles[0, i, 1]), circles[0, i, 2], 255)
        cv2.imwrite("cannyc.jpg", Cg)

    try:
        if circles.shape[1] < 1:
            print("No circle detected")
            return
    except:
        print("No circle detected")
        return

    C = (circles[0, 0, 0], circles[0, 0, 1])
    r = circles[0, 0, 2] - 25
    w = int(math.ceil(2 * r))
    ri = w / 2

    mask = np.ones((w, w), dtype=np.uint8) * 255
    cv2.circle(mask, (w / 2, w / 2), int(r), 0, -1)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    gray_pad = np.ones((gray.shape[0] + w, gray.shape[1] + w), dtype=np.uint8) * 255
    gray_pad[ri:ri + gray.shape[0], ri:ri + gray.shape[1]] = gray
    ulx = int(ri + C[0] - w / 2);
    uly = int(ri + C[1] - w / 2);

    # print gray.shape
    # print gray_pad.shape
    print
    circles[0, 0]
    cv2.circle(I, C, int(r), (0, 0, 255), 5)

    roi = gray_pad[uly:uly + w, ulx:ulx + w] | mask
    # threshold,roit = cv2.threshold(roi,0,255,\
    #        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roit = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 10)
    roit = cv2.morphologyEx(roit, cv2.MORPH_OPEN, kernel5)
    roit = cv2.morphologyEx(roit, cv2.MORPH_ERODE, kernel3)

    roic = roit.copy()
    test = np.zeros(roit.shape, dtype=np.uint8)
    if cvmajor == 3:
        # OpenCV 3
        _, contours, hierarchy = cv2.findContours(roic, cv2.RETR_EXTERNAL, \
                                                  cv2.CHAIN_APPROX_NONE);
    else:
        # OpenCV 2.4
        contours, hierarchy = cv2.findContours(roic, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_NONE);
    contour_shift = []
    for c in contours:
        a = c.copy()
        a[:, 0, 0] += ulx - ri
        a[:, 0, 1] += uly - ri
        contour_shift.append(a)
    cv2.drawContours(I, contour_shift, -1, (0, 255, 0), -1)
    cv2.drawContours(roit, contours, -1, 255, -1)
    roic = roit.copy()
    if cvmajor == 3:
        # OpenCV 3
        _, contours, hierarchy = cv2.findContours(roic, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    else:
        # OpenCV 2.4
        contours, hierarchy = cv2.findContours(roic, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    # Done after scale estimation now
    # fout=open(image[0:-4]+".csv","w")
    self.data = image[0:-4] + ".csv"
    firstline = True

    keys = ["m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03", "mu20", "mu11", "mu02", "mu30", "mu21", "mu12",
            "mu03", "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]

    contours.sort(key=cv2.contourArea)
    contours.reverse()

    statistics = []
    headers = []
    for cnt in contours:
        M = cv2.moments(cnt, True)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        cov_xx = M['mu20'] / M['m00']
        cov_xy = M['mu11'] / M['m00']
        cov_yy = M['mu02'] / M['m00']
        T = cov_xx + cov_yy;
        D = cov_xx * cov_yy - cov_xy * cov_xy;
        delta = T * T / 4 - D;
        assert (delta > -1e-8);
        if abs(delta) <= 1e-8:
            delta = 0;
        lambda1 = T / 2 + math.sqrt(delta);
        lambda2 = T / 2 - math.sqrt(delta);
        if lambda1 > 1e-12 and lambda2 > 1e-12:
            length = 4 * math.sqrt(lambda1);
            thickness = 4 * math.sqrt(lambda2);
            angle = 0.5 * math.atan2(2 * cov_xy, (cov_xx - cov_yy));
        else:
            # default value for pathological cases
            length = 0;
            thickness = 0;
            angle = 0;

        P = cv2.arcLength(cnt, True)
        values = [cx, cy, M['m00'], P, length, thickness, angle * 180 / math.pi] \
                 + [M[k] for k in keys]
        statistics.append(values)

