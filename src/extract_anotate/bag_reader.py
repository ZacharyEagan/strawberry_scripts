import rospy
import sys
from std_msgs.msg import String

from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge, CvBridgeError
import cv2

import PIL
from GPSPhoto import gpsphoto
import struct
import time

def writeTofile(filename, gps):
    info = gpsphoto.GPSInfo((gps[0],gps[1]),alt=int(gps[2]))
    photo = gpsphoto.GPSPhoto(filename)
    photo.modGPSData(info, 'exif_'+filename)
    print("Stored blob data into: ", 'exif_'+filename, "\n")


class gps_tagger:
   def __init__(self, name):
      self.name = name
      self.count = 0
      self.gps = [0.00,0.00, 0.00]
      rospy.init_node('node_name')
      rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
      rospy.Subscriber("/fix", NavSatFix, self.gps_callback)
      self.bridge = CvBridge()

   def image_callback(self, msg):
      print("Received an image!")
      try:
         # Convert your ROS Image message to OpenCV2
         cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
      except CvBridgeError, e:
         print(e)
      else:
         # Save your OpenCV2 image as a png 
         name = "bag_"+self.name+str(self.count)+".jpg"
         cv2.imwrite(name, cv2_img)
         writeTofile(name, self.gps)
         self.count += 1

   def gps_callback(self, gps):
      print("gps!")
      self.gps = [gps.latitude, gps.longitude, gps.altitude]
      print(self.gps)
      
      

if __name__=='__main__':
   tagger = gps_tagger(sys.argv[1])
   rospy.spin()
