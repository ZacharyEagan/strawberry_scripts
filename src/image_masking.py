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
      self.tilowH = cv2.createTrackbar('lowH','image',40,179,self.callback)
      self.tihighH = cv2.createTrackbar('highH','image',120,179,self.callback)
      self.ilowH = cv2.getTrackbarPos('lowH', 'image')      
      self.ihighH = cv2.getTrackbarPos('highH', 'image') 


      self.tilows = cv2.createTrackbar('lows','image',50,255,self.callback)
      self.tihighs = cv2.createTrackbar('highs','image',110,255,self.callback)
      self.ilows = cv2.getTrackbarPos('lows', 'image')      
      self.ihighs = cv2.getTrackbarPos('highs', 'image') 


      self.tilowv = cv2.createTrackbar('lowv','image',40,255,self.callback)
      self.tihighv = cv2.createTrackbar('highv','image',255,255,self.callback)
      self.ilowv = cv2.getTrackbarPos('lowv', 'image')      
      self.ihighv = cv2.getTrackbarPos('highH', 'image') 

      self.tclose = cv2.createTrackbar('close','image',1,16,self.callback)
      self.close = cv2.getTrackbarPos('close', 'image')      

      self.new_img(name)

      
   def callback(self,data):
      self.ilowH = cv2.getTrackbarPos('lowH', 'image')      
      self.ihighH = cv2.getTrackbarPos('highH', 'image') 
      self.ilows = cv2.getTrackbarPos('lows', 'image')      
      self.ihighs = cv2.getTrackbarPos('highs', 'image') 
      self.ilowv = cv2.getTrackbarPos('lowv', 'image')      
      self.ihighv = cv2.getTrackbarPos('highv', 'image') 
      self.close = int((cv2.getTrackbarPos('close', 'image')/2 + 1))      
      self.update_mask()

   def new_img(self, name):
      self.img = cv2.imread(name)
      self.name = name.replace('.jpg','')
      self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
      self.update_mask()
 
   def update_mask(self):
      low = np.array([self.ilowH,self.ilows,self.ilowv])
      high = np.array([self.ihighH,self.ihighs,self.ihighv])
      self.mask = cv2.inRange(self.img_hsv, low,high)

      kernel = np.ones((self.close,self.close),np.uint8)
      self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

      self.img_res = cv2.bitwise_and(self.img, self.img, mask=self.mask)
   
   def show(self, save):
      cv2.imshow('mask', self.mask)
      cv2.imshow('origional', self.img)
      cv2.imshow('image', self.img_res)

      k = cv2.waitKey(10) & 0xFF
      if k == ord('s') or save:
         self.save_mask()
         self.save_masked()
      if k == ord('q'):
         return True 

   def save_mask(self):
      cv2.imwrite("mask/"+self.name+"_mask.png", self.mask)
      
   def save_masked(self):
      cv2.imwrite("masked/"+self.name+"_masked.jpg", self.img_res)
      data = gpsphoto.getGPSData(self.name+'.jpg')
      for tag in data.keys():
         print "%s: %s" % (tag, data[tag])
      info = gpsphoto.GPSInfo((data['Latitude'],data['Longitude']),alt=data['Altitude'])
      photo = gpsphoto.GPSPhoto("masked/"+self.name+"_masked.jpg")
      photo.modGPSData(info, "masked/"+self.name+"_masked.jpg")
      
      
"""
   usage: move script into working image directory, make sub-directory called "mask" and "masked".
      ls *.jpg | python image_masking.py
      then use the sliders on the "image" window to filter the first image. when satisfied press 's' then press 'q'. program will use the same settings to mask all images in the directory. masked versions of the image (with gps data still attatched if available in the origional) will be placed in masked/ the mask images will be placed into mask. use these with agisoft.      
"""
              
if __name__=='__main__':
   f = filter("exif_bag_2020-06-29-15-08-53_0401.jpg")  
   while not f.show(False):
      pass
   for line in sys.stdin:
      print(line.strip())
      f.new_img(line.strip())
      f.show(True)


