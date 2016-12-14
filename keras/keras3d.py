# import keras, pickle
# from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
# from keras.datasets import cifar10
# from keras.utils import np_utils
# from keras.optimizers import SGD

import numpy as np
import os, sys
import cv2
import re
from pdb import set_trace as t
import time
#hyperparameters
samples_per_epoch = 500
num_epochs = 1000
kernel_shape = (3, 3)
num_filters = 32


train_dir = './phoneme_mouth_frames_3/'
#### comment if not on bora's
bora_mac_path = '/Users/boraerden/Desktop/221 project stuff/phoneme_mouth_frames_3/'
train_dir = bora_mac_path

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

def load_data(train_dir, pruned_jpgs_tuples, frames_per_phoneme):
	ti = time.time()
	height, width, _ = cv2.imread(train_dir + pruned_jpgs_tuples[0][0]).shape

	#get phonemes present in frames
	phonemes = []
	batch_size = len(pruned_jpgs_tuples)
	for tuple_i in pruned_jpgs_tuples:
	        specs = re.split('\_', tuple_i[0])
	        if specs[2] not in phonemes: 
	        	phonemes.append(specs[2])
	phonemes = sorted(phonemes)
	numPhonemes = len(phonemes)

	data = np.empty((batch_size, height, width, frames_per_phoneme, 1))
	labels = np.empty((batch_size))
	for batch_index, tuple_i in enumerate(pruned_jpgs_tuples):
		for t_index, frame in enumerate(tuple_i):
			jpeg = cv2.imread(train_dir + frame)
        	for i in range(height):
        		for j in range(width):
        				data[batch_index, i, j, t_index, 0] = jpeg[i,j,0]
	        if t_index == 0:	
	        	specs = re.split('\_', frame)
	        	t()
		        labels[batch_index] = phonemes.index(specs[2])
	print 'matrix: reloaded, took ' + str(time.time()- ti) + ' seconds'
	return data, labels, numPhonemes, batch_size

def partition_batch(batch_size, subbatch_size, data, labels):
	subbatches = []
	subbatch_labels = []
	n_subbatches = 0
	for i in range(0, batch_size, subbatch_size):
		if (i+subbatch_size) in range(batch_size):
			subbatches.append(data[i:i+subbatch_size])
			subbatch_labels.append(labels[i:i+subbatch_size])
			n_subbatches = n_subbatches + 1
	subbatches = np.array(subbatches)
	subbatch_labels = np.array(subbatch_labels)
	return subbatches, subbatch_labels, n_subbatches


############################################################################
############################################################################
############################################################################
############################################################################


jpegDirs = os.listdir(train_dir)
actually_jpegs = [jpegDir for jpegDir in jpegDirs if 'jpg' in  jpegDir]
frames_per_phoneme, pruned_jpgs_tuples =  get_frames_per_phoneme(actually_jpegs)
data, labels, num_classes, batch_size = load_data(train_dir, pruned_jpgs_tuples, frames_per_phoneme)
# Convert class vectors to binary class matrices.
t()
labels = np_utils.to_categorical(labels, num_classes)
t()
#make smaller batches for train_on_batch
subbatch_size = 200
subbatches, subbatch_labels, n_subbatches = partition_batch(batch_size, subbatch_size, data, labels)


model = keras.models.Sequential()
kdim1, kdim2 = kernel_shape
#print num_filters, kdim1, kdim2, data_input_shape
model.add(Convolution2D(num_filters, kdim1, kdim2, input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# def accuracy(y_true, y_pred):
# 	keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

for i in range(0, num_epochs):
	ti = time.time()
	metrics = []
	for b in range(n_subbatches-1):
		print model.train_on_batch(subbatches[b], subbatch_labels[b])
		print 'batch ' + str(b) + '/' + str(n_subbatches)
	metric = model.train_on_batch(subbatches[b+1], subbatch_labels[b+1])
	metrics.append('step = ' + str(i) + ':' + str(metric) + ' in ' + str(time.time()-ti))
	print 'with ' + str(subbatch_size) + ' in batch:'
	for step_metric in metrics:
		print '  ' + step_metric









