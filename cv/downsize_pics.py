import cv2, os
from pdb import set_trace as t

old_pics_folder = "/Users/boraerden/Desktop/vidTIMIT/"
new_pics_folder = "/Users/boraerden/Desktop/downsize_vidTIMIT/"

picture_names = os.listdir(old_pics_folder)


for picture_name in picture_names:
	cv2.imshow(old_pics_folder + picture_name)
	t()