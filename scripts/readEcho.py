import cv2
import numpy as np
import os


def get_file():
  path = './EchoNet-Dynamic/EchoNet-Dynamic/Videos/'
  files = os.listdir(path)

  for file in files:
    name_file = file
    path_file = os.path.join(path, name_file)
    print(name_file, path_file)


get_file()
 
""" # Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("./EchoNet-Dynamic/EchoNet-Dynamic/Videos/0X1A0A263B22CCD966.avi")


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    if cap.get(1) == 50.0:
        image = cv2.circle(frame, (100,100), radius=1, color=(0, 255,0), thickness=-1)

        cv2.imshow('punt',image)
    
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows() """