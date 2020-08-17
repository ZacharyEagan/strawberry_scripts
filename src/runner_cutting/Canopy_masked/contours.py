import cv2
import numpy as np




def boundry(img, min_rad=100, max_rad=10000):
   #expects an image which has been pre-filtered to isolate canopy
   #img = cv2.resize(img, (2160, 1440))
   img_orig = img.copy()
   

   area_min = 3 * min_rad * min_rad
   area_max = 3 * max_rad * max_rad
   print(area_min, area_max)

   #closing holes that might have been left by previous functions
   kernel = np.ones((5,5),np.uint8)
   img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
   img = cv2.blur(img, (5,5)) 

   #convert to grayscale
   img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   #reduce blur at edges by filtering dimmest pixels 
   _, thresh = cv2.threshold(img_gray, 10, 255, 0)
   #find contours in the image (use only last 2 outputs)
   contours, _ = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]


   hull = []
   color = (255, 0, 0)
   drawing = np.zeros((thresh.shape[0],thresh.shape[1],3), np.uint8)

   for contour in contours:
      if area_min <= cv2.contourArea(contour) <= area_max: 
         h = cv2.convexHull(contour, False)
         hull.append(h)

   for i in range(len(hull)):
      cv2.drawContours(drawing, hull, i, color, 8, 8)

   img_out = drawing + (img_orig/2)
   return  drawing, img_out
   

if __name__=='__main__':
   img = cv2.imread('start_masked.JPG')
   drawing, img_contours = boundry(img)
   cv2.imshow("contours", cv2.resize(img_contours,(720,640)))
   cv2.waitKey(0)
