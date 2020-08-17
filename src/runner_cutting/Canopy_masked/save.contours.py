import cv2
import numpy as np

img = cv2.imread('start_masked.JPG')
img = cv2.resize(img, (1080, 720))
img_orig = img.copy()


#img = cv2.blur(img, (11,11)) 


img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 10, 255, 0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

hull = []
print("hulling: ", len(contours))
area_min = 20
area_max = 100000
for contour in contours:
   if area_min <= cv2.contourArea(contour) <= area_max: 
      hull.append(cv2.convexHull(contour, False))
      print("found_contour")
drawing = np.zeros((thresh.shape[0],thresh.shape[1],3), np.uint8)
print("drawing")

for i in range(len(contours)):
   #color_contours = (0, 255, 0)
   color = (255, 0, 0)
   #cv2.drawContours(drawing, contours, i, color_contours, 8, 8, hierarchy)
   cv2.drawContours(drawing, hull, i, color, 8, 8)
   if not (i % 100):
      print(i)
img_out =  drawing + (img_orig/2)
cv2.imshow('img',img_out)
cv2.imshow('img1',drawing)

cv2.waitKey(0)
cv2.destroyAllWindows()
