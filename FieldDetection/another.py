import cv2
import numpy as np


def callback():
    pass


imgsrc = cv2.imread("foot.png")
hLow = 38
hHigh = 55
sLow = 58
sHigh = 255
vLow = 169
vHigh = 255

# 高斯滤波，平滑图像
img = cv2.GaussianBlur(imgsrc, (3, 3), sigmaX=0, sigmaY=0)

# 锐化
kernel_sharpen = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], np.float32)
img = cv2.filter2D(img, -1, kernel=kernel_sharpen)


# 去除观众席背景
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([hLow, sLow, vLow])
upper_green = np.array([hHigh, sHigh, vHigh])
mask = cv2.inRange(img, lower_green, upper_green)
img = cv2.bitwise_and(img, img, mask=mask)
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 自适应阈值，二值化图像
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imshow("adaptive", img)

# 黑白转换
img = cv2.bitwise_not(img)
# cv2.imshow("white", img)

# 闭运算
kernel2 = np.ones((5, 5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
cv2.imshow("biyunsuan1", img)

draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 找椭圆
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
num = 0
for i in range(0, len(contours)):
    num = (num + 20) % 100
    cv2.drawContours(draw, contours, i, (i, 50 + i, 255 - i), 3)


cv2.imshow("t", draw)

# Hough Lines params
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
# minimum number of votes (intersections in Hough grid cell)
threshold = 50
min_line_length = 500  # minimum number of pixels making up a line
max_line_gap = 40  # maximum gap in pixels between connectable line segments

# Run Hough on edge detected image
lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

output = np.copy(imgsrc) * 0  # creating a blank to draw lines on
print(len(lines))
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255))

cv2.imshow("output", output)

cv2.waitKey(0)
