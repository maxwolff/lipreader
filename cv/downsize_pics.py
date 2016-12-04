import cv2, os
from pdb import set_trace as t

old_pics_folder = "/Users/boraerden/Desktop/vidTIMIT/"
new_pics_folder = "/Users/boraerden/Desktop/downsize_vidTIMIT/"

picture_names = os.listdir(old_pics_folder)

os.chdir(new_pics_folder)

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.13.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

face_width = 200
face_height = 250

for index, picture_name in enumerate(picture_names):
	print str(index) + '/' + str(len(picture_names))
	fullpath = old_pics_folder + picture_name
	picture = cv2.imread(fullpath)
	gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	assert len(faces) > 0

	biggest_face_area = 0
	biggest_face = 0,0,0,0
	for (x,y,w,h) in faces:
		if w*h > biggest_face_area:
			biggest_face_area = w*h
			biggest_face = x,y,w,h



	x,y,w,h = biggest_face
	x_crop_start = (2*x+w-face_width)/2
	x_crop_end = x_crop_start + face_width
	y_crop_start = (2*y+h-face_height)/2
	y_crop_end = y_crop_start + face_height
	cropped_gray = gray[y_crop_start:y_crop_end, x_crop_start:x_crop_end]

	cv2.imwrite(picture_name, cropped_gray)



