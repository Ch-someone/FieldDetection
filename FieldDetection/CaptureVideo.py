import cv2
import ui
import numpy as np
import time

def nothing(self):
    pass


def videoProcessing():
    video = cv2.VideoCapture()
    video.open('video1.mp4')
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    nowPos = 0
    pointer = 0
    ui.createUI(frames)

    while video.isOpened():
        startTime = time.time()
        cv2.setTrackbarPos('time', 'HSV', nowPos)

        key = cv2.waitKey(1) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break
        ret, imgsrc = video.read()
        readtime = time.time()
        if imgsrc is False:
            break

        # 高斯滤波，平滑图像
        img = cv2.GaussianBlur(imgsrc, (3, 3), sigmaX=0, sigmaY=0)

        # 锐化
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]], np.float32)
        img = cv2.filter2D(img, -1, kernel=kernel_sharpen)

        # 转HSV色彩空间
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h1 = cv2.getTrackbarPos('H1', 'HSV')
        h2 = cv2.getTrackbarPos('H2', 'HSV')
        hHigh = max(h1, h2)
        hLow = min(h1, h2)

        s1 = cv2.getTrackbarPos('S1', 'HSV')
        s2 = cv2.getTrackbarPos('S2', 'HSV')
        sHigh = max(s1, s2)
        sLow = min(s1, s2)

        v1 = cv2.getTrackbarPos('V1', 'HSV')
        v2 = cv2.getTrackbarPos('V2', 'HSV')
        vHigh = max(v1, v2)
        vLow = min(v1, v2)

        if hLow == hHigh == sHigh == sLow == vLow == vHigh == 0:
            hLow = 38
            hHigh = 55
            sLow = 58
            sHigh = 255
            vLow = 169
            vHigh = 255

        lower_green = np.array([hLow, sLow, vLow])
        upper_green = np.array([hHigh, sHigh, vHigh])
        mask = cv2.inRange(img, lower_green, upper_green)
        cv2.imshow("mask", mask)
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

        # Hough Lines params
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 30
        min_line_length = 300  # minimum number of pixels making up a line
        max_line_gap = 25  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        output = np.copy(imgsrc) * 0  # creating a blank to draw lines on

        fieldLine = np.ndarray
        if lines is not None:
            for i in range(0, len(lines)):
                for j in range(i + 1, len(lines)):
                    line1 = lines[i]
                    line2 = lines[j]
        processTime = time.time()
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255))
                print(line)
        cv2.imshow("output", output)

        pointer = cv2.getTrackbarPos('time', 'HSV')
        if pointer != nowPos:
            nowPos = pointer
            video.set(cv2.CAP_PROP_POS_FRAMES, pointer)
        nowPos += 1
        endtime = time.time()
        print("readtime: ", readtime - startTime)
        print("processTime: ", processTime - readtime)
        print("totalTime: ", endtime - startTime)
    cv2.waitKey(0)
