import numpy as np
import cv2
from pdb import set_trace as t

mouth_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.13.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')

img = cv2.imread('/Users/boraerden/Google Drive/classes/4 Senior/Seni 1/cs221/cs_221_project/vidTIMIT/fadg0/video/sa1/001.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)

biggestMouthArea = 0
biggestMouth = 0,0,0,0
for (x,y,w,h) in mouths:
	if w*h > biggestMouthArea:
		biggestMouthArea = w*h
		biggestMouth = x,y,w,h
t()
x,y,w,h = biggestMouth
cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
roi_gray = gray[y:y+h, x:x+w]
roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

