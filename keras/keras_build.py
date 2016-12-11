import os, sys
import re
import numpy as np
import cv2

def read_data(train_dir):
	num_classes = 3


	jpegDirs = os.listdir(train_dir)
	jpegDirs = sorted(jpegDirs)
	# batch_size = len(jpegDirs)/NUM_FRAMES
	batch_size = 45
	height, width, _ = cv2.imread(train_dir + jpegDirs[1]).shape
	# input_shape = (1, NUM_FRAMES, height, width)
	input_shape = (width, height, 1)
	# curSpecs = ['', '', '']
	# batch_index =-1
	batch_index = 0
	# frame_index = 0
	# videos = np.empty((batch_size, NUM_FRAMES, height, width, 1))
	videos = np.empty((batch_size, width, height, 1))
	# phonemes = []
	# for jpegDir in jpegDirs:
	# 	if jpegDir[-3:] != 'jpg': continue
	# 	specs = re.split('\_', jpegDir)
	# 	if specs[2] not in phonemes: phonemes.append(specs[2])
	# phonemes = sorted(phonemes)
	# numPhonemes = len(phonemes)
	classes = ['dog', 'car', 'flower']
	labels = np.zeros((batch_size, num_classes))
	for jpegDir in jpegDirs:
		#speaker_sentence_phoneme_frameNumber
		if jpegDir[-3:] != 'jpg': continue
		jpeg = cv2.imread(train_dir + jpegDir)
		specs = re.split('\_', jpegDir)
		# if specs[0] != curSpecs[0] or specs[1] != curSpecs[1] or specs[2] != curSpecs[2]:
		# 	frame_index = 0
		# 	batch_index += 1
		# 	label = phonemes.index(specs[2])
		# 	labels[batch_index] = label
		for i in range(0,height):
		 	for j in range(0,width):
		 		videos[batch_index, j, i, 0] = jpeg[i, j, 0]
		labels[batch_index, classes.index(specs[0])] = 1
		batch_index += 1
		# curSpecs = specs
	print 'matrix: reloaded'
	return videos, labels, input_shape

	#labels is supposed be batch_sizexnum_classes and its supposed to be boolean