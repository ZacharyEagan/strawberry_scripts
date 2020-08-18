import sys
import cv2
import numpy as np




class filter:
   def __init__(self, name):
      cv2.namedWindow('controls')

      self.tilowH = cv2.createTrackbar('lowH','controls',40,179,self.callback)
      self.tihighH = cv2.createTrackbar('highH','controls',120,179,self.callback)
      self.ilowH = cv2.getTrackbarPos('lowH', 'controls')      
      self.ihighH = cv2.getTrackbarPos('highH', 'controls') 


      self.tilows = cv2.createTrackbar('lows','controls',50,255,self.callback)
      self.tihighs = cv2.createTrackbar('highs','controls',110,255,self.callback)
      self.ilows = cv2.getTrackbarPos('lows', 'controls')      
      self.ihighs = cv2.getTrackbarPos('highs', 'controls') 


      self.tilowv = cv2.createTrackbar('lowv','controls',40,255,self.callback)
      self.tihighv = cv2.createTrackbar('highv','controls',255,255,self.callback)
      self.ilowv = cv2.getTrackbarPos('lowv', 'controls')      
      self.ihighv = cv2.getTrackbarPos('highH', 'controls') 

      self.tclose = cv2.createTrackbar('close','controls',1,16,self.callback)
      self.close = cv2.getTrackbarPos('close', 'controls')   
      cv2.waitKey(10) #short pause for windows to get trackbars ready
      self.init_callback()
     
      self.new_img(name)
      
   def init_callback(self):  
      self.ilowH = cv2.getTrackbarPos('lowH', 'controls')      
      self.ihighH = cv2.getTrackbarPos('highH', 'controls') 
      self.ilows = cv2.getTrackbarPos('lows', 'controls')      
      self.ihighs = cv2.getTrackbarPos('highs', 'controls') 
      self.ilowv = cv2.getTrackbarPos('lowv', 'controls')      
      self.ihighv = cv2.getTrackbarPos('highv', 'controls') 
      self.close = int((cv2.getTrackbarPos('close', 'controls')/2 + 1)) 

      
   def callback(self,data):  
      self.ilowH = cv2.getTrackbarPos('lowH', 'controls')      
      self.ihighH = cv2.getTrackbarPos('highH', 'controls') 
      self.ilows = cv2.getTrackbarPos('lows', 'controls')      
      self.ihighs = cv2.getTrackbarPos('highs', 'controls') 
      self.ilowv = cv2.getTrackbarPos('lowv', 'controls')      
      self.ihighv = cv2.getTrackbarPos('highv', 'controls') 
      self.close = int((cv2.getTrackbarPos('close', 'controls')/2 + 1))      
      self.update_mask()

   def new_img(self, name):
      self.img = cv2.imread(name)
      self.name = name.replace('.JPG','')
      self.name = self.name.replace('.JPG','')
      self.name = self.name.replace('.png','')
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
      #cv2.imshow('mask', self.mask)
      #cv2.imshow('origional', self.img)
      self.img_small = cv2.resize(self.img_res, (960, 540));
      cv2.imshow('image', self.img_small)

      k = cv2.waitKey(10) & 0xFF
      if k == ord('s') or save:
         self.save()
      if k == ord('q'):
         return True 

   def save(self):
      cv2.imwrite(self.name+"_mask.png", self.mask)
      cv2.imwrite(self.name+"_masked.png", self.img_res)
      
   def shutdown(self):
      cv2.waitKey(1)
      cv2.destroyAllWindows()
      cv2.waitKey(1)
      cv2.waitKey(1)
      cv2.waitKey(1)
      cv2.waitKey(1)
      
"""
   usage: move script into working image directory, make sub-directory called "mask" and "masked".
      ls *.JPG | python image_masking.py
      then use the sliders on the "image" window to filter the first image. when satisfied press 's' then press 'q'. program will use the same settings to mask all images in the directory. masked versions of the image (with gps data still attatched if available in the origional) will be placed in masked/ the mask images will be placed into mask. use these with agisoft.      
"""
              
if __name__=='__main__':
   f = filter("start.JPG")  
   while not f.show(False):
      pass
   for line in sys.stdin:
      print(line.strip())
      f.new_img(line.strip())
      f.show(True)
   f.shutdown()


