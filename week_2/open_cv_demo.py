import cv2
import numpy as np

filename = (beautiful_women.png)

img = cv2.imread(filename)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for i in range (10):
	low = i * 18
	high = low + 17
	mask = cv2.inRange(hsv, (low, 50, 50), (high, 255, 255))
	extracted = cv2.bitwise_and(img, img, mask=mask)
	
	cv2.imwrite(f'hue_{low}_{high}.png', extracted)
	print("Hue range" + str(low)+ ", " + str(high))

