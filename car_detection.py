#!/usr/bin/env python
# coding: utf-8

# In[21]:


import cv2

# Load the video
cap = cv2.VideoCapture(r"C:\Users\hii\Downloads\854671-hd_1920_1080_25fps.mp4")

# Load the object you want to detect (e.g., a car)
car_cascade = cv2.CascadeClassifier(r"C:\Users\hii\Downloads\cas4.xml")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects in the frame
  
    car = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
    
  
    

    # Draw bounding boxes around the detected objects
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
   



    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[25]:


import cv2
import numpy as np

def diffUpDown(img):
    height, width, depth = img.shape
    half = height // 2
    top = img[0:half, 0:width]
    bottom = img[half:half+half, 0:width]
    top = cv2.flip(top, 1)
    bottom = cv2.resize(bottom, (32, 64))
    top = cv2.resize(top, (32, 64))
    return mse(top, bottom)

def diffLeftRight(img):
    height, width, depth = img.shape
    half = width // 2
    left = img[0:height, 0:half]
    right = img[0:height, half:half + half - 1]
    right = cv2.flip(right, 1)
    left = cv2.resize(left, (32, 64))
    right = cv2.resize(right, (32, 64))
    return mse(left, right)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def isNewRoi(rx, ry, rw, rh, rectangles):
    for r in rectangles:
        if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
            return False
    return True

def detectRegionsOfInterest(frame, cascade):
    cars = cascade.detectMultiScale(frame, 1.2, 1)
    newRegions = []
    minY = int(frame.shape[0] * 0.3)
        
    for (x, y, w, h) in cars:
        roiImage = frame[y:y+h, x:x+w]
        if y > minY:
            diffX = diffLeftRight(roiImage)
            diffY = round(diffUpDown(roiImage))
            if diffX > 1600 and diffX < 3000 and diffY > 12000:
                rx, ry, rw, rh = x, y, w, h
                newRegions.append([rx, ry, rw, rh])
    
    return newRegions
    
def detectCars(filename):
    rectangles = []
    cascade = cv2.CascadeClassifier(r"C:\Users\hii\Downloads\haarcascade_car.xml")
    vc = cv2.VideoCapture(filename)
    
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        
        newRegions = detectRegionsOfInterest(frame, cascade)
        for region in newRegions:
            if isNewRoi(region[0], region[1], region[2], region[3], rectangles):
                rectangles.append(region)
        
        for r in rectangles:
            cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255), 3)
        
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vc.release()
    cv2.destroyAllWindows()

detectCars(r"C:\Users\hii\Downloads\854671-hd_1920_1080_25fps.mp4")


# In[2]:


import cv2
import ctypes
print(cv2.__version__)

cascade_src = r"C:\Users\hii\Downloads\haarcascade_car.xml"
video_src = r"C:\Users\hii\Downloads\video.avi"
#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
# Get the screen resolution
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', screen_width, screen_height)


while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    cv2.imshow('Object Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[10]:


import cv2
import ctypes

#cascade_src = r"C:\Users\hii\Downloads\haarcascade_car.xml"
#video_src = r"C:\Users\hii\Downloads\video.avi"


vid = cv2.VideoCapture(r"C:\Users\hii\Downloads\5927708-hd_1080_1920_30fps.mp4")
car_cascade = cv2.CascadeClassifier(r"C:\Users\hii\Downloads\haarcascade_car.xml")
# Get the screen resolution
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

vid = cv2.VideoCapture(r"C:\Users\hii\Downloads\5927708-hd_1080_1920_30fps.mp4")
car_cascade = cv2.CascadeClassifier(r"C:\Users\hii\Downloads\haarcascade_car.xml")

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', screen_width, screen_height)


while True:
    ret, img = vid.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    cv2.imshow('Object Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# In[ ]:




