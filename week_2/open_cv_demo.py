import cv2
import numpy as np


filename = ('IMG_1518')

img = cv2.imread(filename + '.JPG')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

for i in range (10):
	low = i * 18
	high = low + 17
	mask = cv2.inRange(hsv, (low, 50, 50), (high, 255, 255))
	combined_mask = cv2.bitwise_or(combined_mask, mask)
	extracted = cv2.bitwise_and(img, img, mask=mask)
	cv2.imwrite(f'{filename}_hue_{low}_{high}.png', extracted)
	print("Hue range" + str(low)+ ", " + str(high))

leftover_mask = cv2.bitwise_not(combined_mask)
leftover = cv2.bitwise_and(img, img, mask=leftover_mask)
cv2.imwrite(f'{filename}_leftover_low_sat_or_dark.png', leftover)