
import keras
from keras.optimizers import SGD
import numpy as np
import os, sys
import cv2
import re
from pdb import set_trace as t
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
import time

learning_rate = 0.015
decay = 0.003
momentum = 0.75


test_dir = './speaker1_test/'
phonemes = ['aa', 'ae', 'ah', 'ao', 'ax', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
print len(phonemes)


def get_frames_per_phoneme(actually_jpegs):
	ti = time.time()
	result_list = []
	pruned_jpgs_tuples = []

	while len(actually_jpegs) > 0:
		specs = re.split('\_', actually_jpegs[0])
		pattern = ''
		for each in specs:
			if '.jpg' in each:
				continue
			pattern = pattern + str(each) + '_' 
		pattern_fits = [x for x in actually_jpegs if pattern in x]
		indices = []
		for fit in pattern_fits:
			specs = re.split('\_', fit)
			indices.append(int(specs[-1][:-4]))
		if len(indices) != indices[-1]:
			actually_jpegs = [x for x in actually_jpegs if x not in pattern_fits]
		else:
			pruned_jpgs_tuples.append(pattern_fits)
			actually_jpegs = [x for x in actually_jpegs if x not in pattern_fits]

		result_list.append(indices[-1])
	print 'getting frames per phoneme and cleaning incompletes took ' + str(time.time()-ti) + ' seconds'
	return max(set(result_list), key=result_list.count), pruned_jpgs_tuples

def load_data(train_dir, pruned_jpgs_tuples, frames_per_phoneme, phonemes):
	ti = time.time()
	height, width, _ = cv2.imread(train_dir + pruned_jpgs_tuples[0][0]).shape

	#get phonemes present in frames
	batch_size = len(pruned_jpgs_tuples)
	numPhonemes = len(phonemes)
	data = np.empty((batch_size, frames_per_phoneme, height, width, 1))
	labels = np.empty((batch_size))
	for batch_index, tuple_i in enumerate(pruned_jpgs_tuples):
		for t_index, frame in enumerate(tuple_i):
			jpeg = cv2.imread(train_dir + frame)
			for i in range(height):
				for j in range(width):
					data[batch_index, t_index, i, j, 0] = jpeg[i,j,0]
			if t_index == 0:	
				specs = re.split('\_', frame)
				labels[batch_index] = phonemes.index(specs[2])
	print 'matrix: reloaded, took ' + str(time.time()- ti) + ' seconds'
	return data, labels, numPhonemes, batch_size




jpegDirs = os.listdir(test_dir)
jpegDirs = sorted(jpegDirs)
actually_jpegs = [jpegDir for jpegDir in jpegDirs if 'jpg' in  jpegDir]
frames_per_phoneme, pruned_jpgs_tuples =  get_frames_per_phoneme(actually_jpegs)
data, labels, _, _ = load_data(test_dir, pruned_jpgs_tuples, frames_per_phoneme, phonemes)
# convert labels to categorical labels
cat_labels = np_utils.to_categorical(labels, len(phonemes))


# load model saved by keras3d.py
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model')
 
# evaluate loaded model on test data
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)


loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
num_params = loaded_model.count_params()

test_accuracy = loaded_model.evaluate(data, cat_labels, verbose=2)
print test_accuracy

predictions = loaded_model.predict(data)
for pred in predictions:
	print np.argmax(pred)

top_3_accuracy = 0.0
top_5_accuracy = 0.0
for i, pred in enumerate(predictions):
	indices = np.argsort(pred)
	if labels[i] in indices[-3:]:
		top_3_accuracy += 1
	if labels[i] in indices[-5:]:
		top_5_accuracy += 1
top_3_accuracy /= len(predictions)
top_5_accuracy /= len(predictions)
print top_3_accuracy, top_5_accuracy


confusion_matrix = np.zeros((len(phonemes), len(phonemes)))
for i, label in enumerate(labels):
	indices = np.argsort(predictions[i])
	for j in range(2,5):
		confusion_matrix[label, indices[-j]] += 1
for i, row in enumerate(confusion_matrix):
	print phonemes[i], row
