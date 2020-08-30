import cv2
import numpy as np
import os
import sys
from timeit import default_timer as timer

from skimage.filters import frangi, sato
from skimage.morphology import skeletonize, medial_axis

def small(img):
   return cv2.resize(img, (1280,720))

def remove_shadows(img):

   rgb_planes = cv2.split(img)

   result_planes = []
   result_norm_planes = []
   for plane in rgb_planes:
      #plane2 = cv2.blur(plane,(5,5))
      dilated_img = cv2.dilatewww(plane, np.ones((21,21), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 23)
      #bg_img = cv2.medianBlur(plane, 3)
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

   result_norm = cv2.merge(result_norm_planes)
   return result_norm





def boundry(img, min_rad=100, max_rad=10000):
   #expects an image which has been pre-filtered to isolate canopy
   #img = cv2.resize(img, (2160, 1440))
   img_orig = img.copy()


   area_min = 3 * min_rad * min_rad
   area_max = 3 * max_rad * max_rad
   #print(area_min, area_max)

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

def filter_lines(lines, width, threshold):
    new_lines = {}
    for i in range(len(lines)):
        line1 = lines[i]
        new_lines[i] = 0
        for line2 in lines:
            #dx,dy for parrallel lines is 2x the distance between the lines
            #dx,dy for diverging lines will be dominated by the furthest apart points
            
            dx1 = line1[0][0] - line2[0][0]
            dx2 = line1[0][0] - line2[0][2]
            if dx1 < width or dx2 < width:
                dy1 = line1[0][1] - line2[0][1]
                dy2 = line2[0][1] - line2[0][3]
                if dy1 < width or dy2 < width:
                    new_lines[i] += 1
    filtered_lines = []
    for i in range(len(new_lines)):
        if new_lines[i] > threshold:
            filtered_lines.append(lines[i])
    return filtered_lines
            

def find_lines(img,rho):
   #edges = cv2.Canny(img, 100,200)
   lines = cv2.HoughLinesP(img,rho = rho, theta = 1*np.pi/180, threshold = 90, minLineLength=90, maxLineGap = 15 )
   drawing = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
   drawing_filtered = drawing.copy()
   
   for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:     
         pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
         cv2.polylines(drawing, [pts], True, (255,255,255), thickness=4)
   #print(len(lines))
   
   lines = filter_lines(lines,80,20)
   for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:     
         pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
         cv2.polylines(drawing_filtered, [pts], True, (255,255,255), thickness=4)
   #print(len(lines))
      
   return lines, drawing, drawing_filtered

def runner3(img):
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   #combineing the lighting less-respective components helps remove shadows. 
   sat_1 = cv2.normalize(hsv[:,:,0]-hsv[:,:,1],None, alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)

   kernel = np.ones((5,5),np.uint8)
   sat_2= cv2.dilate(sat_1,kernel,iterations = 2)
   cv2.imshow('sat dilated',small(sat_2))
   
   kernel = np.ones((7,7),np.uint8)
   s_opened = cv2.morphologyEx(sat_2, cv2.MORPH_OPEN, kernel, iterations = 9)
   cv2.imshow('sat opened', s_opened)
   sat = sat_2 - s_opened
   sat &= sat_1
   
   #kernel = np.ones((5,5),np.uint8)
   #sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, kernel, iterations = 1)
   kernel = np.ones((3,3),np.uint8)
   sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, kernel, iterations = 1)
   
   #sat = cv2.blur(sat,(13,13))
   sat = cv2.inRange(sat,100,250)
   cv2.imshow('sat',small(sat))
  
   _,lines,lines_filtered = find_lines(sat,6)
   drawing = np.zeros((lines.shape[0],lines.shape[1],1), np.uint8)
   drawing = cv2.merge([lines_filtered,lines,drawing])
   cv2.imshow('d',small(drawing))
   return drawing, lines
   

def runner2(img):
   #img = remove_shadows(img)
   img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   f = frangi(img_gray,black_ridges=True, sigmas=range(1,2,1))
   tubes = cv2.normalize(f, None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
   tubes = find_lines(tubes)[1]
   kernel = np.ones((5,5),np.uint8)
   mask = cv2.morphologyEx(tubes, cv2.MORPH_CLOSE, kernel)
   ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
   img_masked = cv2.bitwise_and(img, img, mask = mask)
   return mask, img_masked, tubes 


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
   #print(top_mask.shape)
   #print(bot.shape)
   bot_masked = cv2.bitwise_and(bot, bot, mask = top_mask_not)
   top_masked = cv2.bitwise_and(top, top, mask = top_mask)
   out = top_masked + bot_masked
   return out

def count_cuts(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(img, contours, -1, (255,0,255), 8)
    
    return contours, len(contours), img
                                     


def runner_cutter(file, name):
   print(file)
   print(name)
   img = cv2.imread(file)
   
   img_can = cv2.imread('Canopy_masked/'+name+'_masked.JPG')
   drawing_can, img_can = boundry(img_can)

   img_run = cv2.imread('Runner_masked/'+name+'_masked.JPG')
   #drawing_run, img_run = runner3(img)
   drawing_run, img_run = runner3(img_run)

   drawings = cv2.resize(drawing_run + drawing_can, (720,480))
   cut = cuts(drawing_run, drawing_can)
   keypoints, num_cuts, cut = count_cuts(cut)
   
   out_img = merge(drawing_can, img)
   out_img = merge(drawing_run, out_img)
   out_img = merge(cut, out_img)

   cv2.imwrite("out/drawings/"+name+".jpg", drawings)
   cv2.imwrite("out/cuts/"+name+".jpg", cut)
   cv2.imwrite("out/output/"+name+".jpg", out_img)

    
   print('cut count', num_cuts)

   drawings = cv2.resize(drawings, (720,480))
   cut = cv2.resize(cut, (720,480))
   out_img = small(out_img)
   
   cv2.imshow('drawings', drawings)
   cv2.imshow('cuts', cut)  
   cv2.imshow('output',out_img)

      
   
def shutdown():
   cv2.waitKey(1)
   cv2.destroyAllWindows()
   cv2.waitKey(1)
   cv2.waitKey(1)
   cv2.waitKey(1)
   cv2.waitKey(1)

def file_manager(path):
   file_names = [x.replace('.JPG','') for x in os.listdir(path)]
   files = [path + '/' + x for x in os.listdir(path)]
   
   for i in range(len(files)):
      file = files[i]
      name = file_names[i]
      if ('.jpg' in file or '.png' in file or '.JPG' in file):
         #try:
         start = timer()
         runner_cutter(file, name)
         end = timer()
         print('elapsed = ' + str( end - start) + ' s')
         cv2.waitKey(0)
         #except Exception as ex:
          #  print(file, ex)
   shutdown()


if __name__=='__main__':
   if len(sys.argv) > 1:
      file_manager(sys.argv[1])
   else:
      print('Usage: python runner_cutter.py <path>')
      







