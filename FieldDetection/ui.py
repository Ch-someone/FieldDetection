import cv2

def nothing(self):
    pass

def createUI(frames):
    cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H1', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('H2', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('S1', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('S2', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('V1', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('V2', 'HSV', 0, 255, nothing)
    cv2.createTrackbar('time', 'HSV', 0, frames, nothing)