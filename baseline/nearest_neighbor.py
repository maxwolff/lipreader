import os, sys
import re
import cv2
from pdb import set_trace as t
import numpy as np

masterFolder = './vidTIMIT/'
inputFolder = './input/'

jpegDirs = os.listdir(masterFolder)
jpeg = cv2.imread(masterFolder + jpegDirs[1])
height, width, numChannels = jpeg.shape
i = 0;
phonemes = []
for jpegDir in jpegDirs:
	if jpegDir[-3:] != 'jpg': continue
	specs = re.split('\_', jpegDir)
	if specs[2] not in phonemes: phonemes.append(specs[2])
phonemes = sorted(phonemes)

eigenPhonemes = []
for phoneme in phonemes:
	eigenPhoneme = np.zeros((height, width, numChannels))
	counter = 0
	for jpegDir in jpegDirs:
		if jpegDir[-3:] != 'jpg': continue
		if counter == 50: break
		specs = re.split('\_', jpegDir)
		if specs[2] == phoneme:
			eigenPhoneme += cv2.imread(masterFolder + jpegDir)
			counter += 1
	eigenPhoneme /= 50
	eigenPhonemes.append(eigenPhoneme)
print 'eigenPhonemes loaded'


frameFolderDirs = os.listdir(inputFolder)
video = 1
correct = 0
for frameFolderDir in frameFolderDirs:
	if frameFolderDir == '.DS_Store': continue
	frameDirs = os.listdir(inputFolder + frameFolderDir)
	distances = [0]*len(phonemes)
	for frameDir in frameDirs:
		if frameDir[-3:] != 'jpg': continue
		frame = cv2.imread(inputFolder + frameFolderDir + '/'+ frameDir)
		if frame == None: t()
		for i in range(0, len(phonemes)):
			if eigenPhonemes[i] == None: t()
			distances[i] += np.linalg.norm(frame - eigenPhonemes[i])
	prediction = distances.index(min(distances))
	print 'video ' + str(video) + ' is ' + str(phonemes[prediction])
	label = re.split('\_', frameDirs[1])[2]
	if phonemes[prediction] == label: correct += 1
	video += 1
print 'accuracy = ' + str(correct) + '/' + str(len(frameFolderDirs) - 1)





