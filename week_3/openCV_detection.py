import cv2
import numpy as np

filename_test = ('test_1.jpg')
filename_detect_1 = ('detect_1.png')
filename_detect_2 = ('detect_2.png')

img_test = cv2.imread(filename_test, cv2.IMREAD_GRAYSCALE)
if img_test is None:
    raise FileNotFoundError(f"Couldn't read {filename_test}; check the path")

img_detect_1 = cv2.imread(filename_detect_1, cv2.IMREAD_GRAYSCALE)
if img_detect_1 is None:
    raise FileNotFoundError(f"Couldn't read {filename_detect_1}; check the path")

img_detect_2 = cv2.imread(filename_detect_2, cv2.IMREAD_GRAYSCALE)
if img_detect_2 is None:
    raise FileNotFoundError(f"Couldn't read {filename_detect_2}; check the path")

params = cv2.SimpleBlobDetector_Params()

params.filterByColor = False

params.minThreshold = 10
params.maxThreshold = 200

params.filterByArea = True
params.minArea = 100

params.filterByConvexity = True
params.minConvexity = .80

params.filterByInertia = True
params.minInertiaRatio = .01

params.filterByCircularity = True
params.minCircularity = 0.1
detector = cv2.SimpleBlobDetector_create(params)

keypoints_test = detector.detect(img_test)
keypoints_detect_1 = detector.detect(img_detect_1)
keypoints_detect_2 = detector.detect(img_detect_2)

output_test = cv2.drawKeypoints(img_test, keypoints_test, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

output_detect_1 = cv2.drawKeypoints(img_detect_1, keypoints_detect_1, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

output_detect_2 = cv2.drawKeypoints(img_detect_2, keypoints_detect_2, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("blobs_detected.png", output_test)
if not cv2.imwrite("blobs_detected.png", output_test):
    print("Save test failed")

cv2.imwrite("blobs_detected_1.png", output_detect_1)
if not cv2.imwrite("blobs_detected_1.png", output_detect_1):
    print("Save detected 1 failed")

cv2.imwrite("blobs_detected_2.png", output_detect_2)
if not cv2.imwrite("blobs_detected_2.png", output_detect_2):
    print("Save detected 2 failed")
