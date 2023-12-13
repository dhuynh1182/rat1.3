import cv2
import numpy as np
from matplotlib import pyplot as plt








path = "motherboard_image.jpeg"

img_real = cv2.imread(path, cv2.IMREAD_COLOR)
img_real = cv2.rotate(img_real, cv2.ROTATE_90_CLOCKWISE)

img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (47, 47), 4)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55,7)
img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)




edges = cv2.Canny(img_gray, 50, 300)
edges = cv2.dilate(edges,None, iterations = 10)

contours, cat = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(img_real)

cv2.drawContours(image=mask, contours=[max(contours, key = cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness= cv2.FILLED)

masked_img =cv2.bitwise_and(mask, img_real)

plt.imshow(masked_img)

plt.show()






