import cv2
import numpy as np

import sys



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
   #cv2.imshow("edges", cv2.resize(img_gray,(720,640)))
   lines = cv2.HoughLinesP(img_gray,1,np.pi/180, 100 , 100, 100)
   drawing = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
   if lines is not None:
      for line in lines:
         x1, y1, x2, y2 = line[0]
         cv2.line(drawing,(x1,y1),(x2,y2),(0,255,0),8)
   img = drawing + (img/2) + drawing

   return drawing, img

def cuts(boundries, runners):
   can_gray = cv2.cvtColor(boundries,cv2.COLOR_BGR2GRAY)
   run_gray = cv2.cvtColor(runners,cv2.COLOR_BGR2GRAY)

   kernel = kernel = np.ones((5,5),np.uint8)
   can_gray = cv2.dilate(can_gray,kernel,iterations = 5)
   run_gray = cv2.dilate(run_gray,kernel,iterations = 5)

   ret,can_gray = cv2.threshold(can_gray,10,255,cv2.THRESH_BINARY)
   ret,run_gray = cv2.threshold(run_gray,10,255,cv2.THRESH_BINARY)

   cuts = cv2.bitwise_and(can_gray, run_gray)
   drawing = np.zeros((cuts.shape[0],cuts.shape[1],1), np.uint8)
   color_out = cv2.merge([drawing,drawing,cuts])
   cv2.imshow("cuts",color_out)
   return color_out

def merge(top, bot):
   top_gray = cv2.cvtColor(top,cv2.COLOR_BGR2GRAY)
   ret,top_mask = cv2.threshold(top_gray,10,255,cv2.THRESH_BINARY)
   top_mask_not = cv2.bitwise_not(top_mask)
   print(top_mask.shape)
   print(bot.shape)
   bot_masked = cv2.bitwise_and(bot, bot, mask = top_mask_not)
   top_masked = cv2.bitwise_and(top, top, mask = top_mask)
   out = top_masked + bot_masked
   return out

def runner_cutter(name):
   img_can = cv2.imread('Canopy_masked/'+name+'_masked.JPG')
   drawing_can, img_can = boundry(img_can)

   img_run = cv2.imread('Runner_masked/'+name+'_masked.JPG')
   drawing_run, img_run = runner(img_run)

   drawings = cv2.resize(drawing_run + drawing_can, (720,480))
   cut = cuts(drawing_run, drawing_can)
   
   img = cv2.imread('8_13/'+name+'.JPG')
   out_img = merge(drawing_can, img)
   out_img = merge(drawing_run, out_img)
   out_img = merge(cut, out_img)

   cv2.imwrite("out/drawings/"+name+".jpg", drawings)
   cv2.imwrite("out/cuts/"+name+".jpg", cut)
   cv2.imwrite("out/output/"+name+".jpg", out_img)


   drawings = cv2.resize(drawings, (720,480))
   cut = cv2.resize(cut, (720,480))
   out_img = cv2.resize(out_img, (720,480))

   cv2.imshow('drawings', drawings)
   cv2.imshow('cuts', cut)  
   cv2.imshow('output',out_img)

   cv2.waitKey(20)   
   



if __name__=='__main__':
   for name in sys.stdin:
      name = name[:name.find('.')]
      try:
         runner_cutter(name)
      except Exception as ex:
         print(name, ex) 
      







