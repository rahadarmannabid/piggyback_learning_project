#python yolov5/detect.py --source https://www.youtube.com/watch?v=NcaGFp76BTY --weights yolov5s.pt --conf 0.25


import cv2
import numpy as np

thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)


classNames= []
classFile = '/home/ran/Desktop/EduEntaninment/coco.names'
with open(classFile,'rt') as f:
   classNames = f.read().rstrip('n').split('n')

#print(classNames)
configPath = '/home/ran/Desktop/EduEntaninment/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/ran/Desktop/EduEntaninment/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/home/ran/Desktop/EduEntaninment/SampleVideo.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #---------------------

    #--------------
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()