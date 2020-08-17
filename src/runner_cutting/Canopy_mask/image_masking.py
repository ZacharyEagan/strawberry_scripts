import sys
import cv2
import numpy as np

import PIL
from GPSPhoto import gpsphoto
import struct
import time


class filter:
   def __init__(self, name):
      cv2.namedWindow('image')
      self.circ_dist = cv2.createTrackbar('circ_dist','image',20,2000,self.callback)
      self.circ_low = cv2.createTrackbar('circ_low','image',2,2000,self.callback)
      self.circ_high = cv2.createTrackbar('circ_high','image',10,2000,self.callback)

      self.icirc_dist = cv2.getTrackbarPos('circ_dist', 'image')      
      self.icirc_low = cv2.getTrackbarPos('circ_low', 'image')      
      self.icirc_high = cv2.getTrackbarPos('circ_high', 'image')      

      self.new_img(name)

      
   def callback(self,data):
      self.icirc_dist = cv2.getTrackbarPos('circ_dist', 'image')      
      self.icirc_low = cv2.getTrackbarPos('circ_low', 'image')      
      self.icirc_high = cv2.getTrackbarPos('circ_high', 'image')      
      self.update_mask()

   def new_img(self, name):
      self.img = cv2.imread(name)
      self.name = name.replace('.JPG','')
      self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
      self.update_mask()


   def blobs_cb(self):
      # Setup SimpleBlobDetector parameters.
      params = cv2.SimpleBlobDetector_Params()

      # Change thresholds
      params.minThreshold = self.icirc_low
      params.maxThreshold = self.icirc_high


      # Filter by Area.
      params.filterByArea = False
      params.minArea = self.icirc_dist

      # Filter by Circularity
      params.filterByCircularity = False
      params.minCircularity = 0.1

      # Filter by Convexity
      params.filterByConvexity = False
      params.minConvexity = 0.5

      # Filter by Inertia
      params.filterByInertia = False
      params.minInertiaRatio = 0.01
   
      detector = cv2.SimpleBlobDetector_create(params)
      im = self.img_res
      im = cv2.blur(im, (11,11))
      keypoints = detector.detect(im)
      im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      self.img_res_blob = im_with_keypoints.copy()

   def circles_cb(self):
      self.img_res_gray = cv2.cvtColor(self.img_res, cv2.COLOR_BGR2GRAY)
      self.img_res_gray = cv2.resize(self.img_res_gray, (720,480))
      circles = cv2.HoughCircles(
               self.img_res_gray, 
               cv2.HOUGH_GRADIENT, 
               4.0,
               self.icirc_dist,
               minRadius=self.icirc_low, 
               maxRadius=self.icirc_high,
               )
      #circles = cv2.HoughCircles(self.img_res_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
      self.circle_out = self.img_res_gray.copy()
      if circles is not None:
         circles = np.round(circles[0,:]).astype("int")
         for (x, y, r) in circles:
            cv2.circle(self.circle_out, (x, y), r, (0,255,0),4)
            cv2.rectangle(self.circle_out, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
      orig_out = cv2.resize(self.img_res_gray,(720,480))
      circ_out = cv2.resize(self.circle_out, (720,480))
      temp = np.hstack([orig_out, circ_out])
      cv2.imshow("circles", temp) 
 
   def update_mask(self):

      self.img_res = self.img.copy()
      self.blobs_cb()
      #self.circles_cb()
   
   def show(self, save):
      #cv2.imshow('mask', self.mask)
      #cv2.imshow('origional', self.img)
      self.img_small = cv2.resize(self.img_res_blob, (960, 540));
      cv2.imshow('image', self.img_small)


      k = cv2.waitKey(10) & 0xFF
      if k == ord('s') or save:
         self.save_mask()
         self.save_masked()
      if k == ord('q'):
         return True 

   def save_mask(self):
      cv2.imwrite("mask/"+self.name+"_mask.png", self.mask)
      
   def save_masked(self):
      cv2.imwrite("masked/"+self.name+"_masked.JPG", self.img_res)
      #data = gpsphoto.getGPSData(self.name+'.JPG')
      #for tag in data.keys():
      #   print "%s: %s" % (tag, data[tag])
      #info = gpsphoto.GPSInfo((data['Latitude'],data['Longitude']),alt=data['Altitude'])
      #photo = gpsphoto.GPSPhoto("masked/"+self.name+"_masked.")
      #photo.modGPSData(info, "masked/"+self.name+"_masked.JPG")
      
      
"""
   usage: move script into working image directory, make sub-directory called "mask" and "masked".
      ls *.JPG | python image_masking.py
      then use the sliders on the "image" window to filter the first image. when satisfied press 's' then press 'q'. program will use the same settings to mask all images in the directory. masked versions of the image (with gps data still attatched if available in the origional) will be placed in masked/ the mask images will be placed into mask. use these with agisoft.      
"""
              
if __name__=='__main__':
   f = filter("start_mask.png")  
   while not f.show(False):
      pass
   for line in sys.stdin:
      print(line.strip())
      f.new_img(line.strip())
      f.show(True)


