import cv2
import numpy as np




def runner(img, min_len=2, max_gap=10000):
   #expects an image which has been pre-filtered to isolate canopy
   #img = cv2.resize(img, (2160, 1440))
   img_orig = img.copy()
   


   #closing holes that might have been left by previous functions
   kernel = np.ones((5,5),np.uint8)
   img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
   img = cv2.blur(img, (5,5)) 

   #convert to grayscale
   img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

   #reduce blur at edges by filtering dimmest pixels 
   _, img_gray = cv2.threshold(img_gray, 50, 255, 0)


   edges = cv2.Canny(img_gray,20,150,apertureSize = 7) 
   cv2.imshow("edges", cv2.resize(img_gray,(720,640)))
   lines = cv2.HoughLinesP(img_gray,1,np.pi/180, 100 , 100, 100)
   drawing = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
   if lines is not None:
      for line in lines:
         x1, y1, x2, y2 = line[0]
         cv2.line(drawing,(x1,y1),(x2,y2),(0,255,0),8)
   img = drawing + (img/2) + drawing

   return drawing, img
   

if __name__=='__main__':
   img = cv2.imread('start_masked.JPG')
   drawing, img_lines = runner(img)
   cv2.imshow("runner", cv2.resize(img_lines,(720,640)))
   cv2.waitKey(0)
