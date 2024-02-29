import cv2
import numpy as np


def nothing():
    pass


img = cv2.imread("colomn.jpeg")
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.createTrackbar('H1', 'output', 0, 255, nothing)
cv2.createTrackbar('H2', 'output', 0, 255, nothing)
cv2.createTrackbar('S1', 'output', 0, 255, nothing)
cv2.createTrackbar('S2', 'output', 0, 255, nothing)
cv2.createTrackbar('V1', 'output', 0, 255, nothing)
cv2.createTrackbar('V2', 'output', 0, 255, nothing)
img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
while True:
    h1 = cv2.getTrackbarPos('H1', 'output')
    h2 = cv2.getTrackbarPos('H2', 'output')
    s1 = cv2.getTrackbarPos('S1', 'output')
    s2 = cv2.getTrackbarPos('S2', 'output')
    v1 = cv2.getTrackbarPos('V1', 'output')
    v2 = cv2.getTrackbarPos('V2', 'output')
    hHigh = max(h1, h2)
    hLow = min(h1, h2)
    sHigh = max(s1, s2)
    sLow = min(s1, s2)
    vHigh = max(v1, v2)
    vLow = min(v1, v2)
    lower_green = np.array([hLow, sLow, vLow])
    upper_green = np.array([hHigh, sHigh, vHigh])
    if hLow == hHigh == sHigh == sLow == vLow == vHigh == 0:
        hLow = 38
        hHigh = 55
        sLow = 58
        sHigh = 255
        vLow = 169
        vHigh = 255
    mask = cv2.inRange(img, lower_green, upper_green)
    cv2.imshow("output", mask)

    cv2.waitKey(1)
