import keras, pickle
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import model_from_json
import os, sys
import cv2
import re
from pdb import set_trace as t
import time
#hyperparameters
num_epochs = 1000
kernel_shape = (5, 5)
num_filters = 32
subbatch_size = 200
learning_rate = 0.01
decay = 0.003
momentum = 0.75



train_dir = './phoneme_mouth_frames_3/'


def load_data(train_dir):
	jpegDirs = os.listdir(train_dir)
	batch_size = 0
	height, width, _ = cv2.imread(train_dir + jpegDirs[1]).shape
	phonemes = []
	for jpegDir in jpegDirs:
	        if jpegDir[-3:] != 'jpg': continue
	        specs = re.split('\_', jpegDir)
	        if specs[2] not in phonemes: phonemes.append(specs[2])
	        batch_size += 1
	phonemes = sorted(phonemes)
	numPhonemes = len(phonemes)
	data = np.empty((batch_size, height, width, 1))
	labels = np.empty((batch_size))
	batch_index = 0
	for jpegDir in jpegDirs:
	        #speaker_sentence_phoneme_frameNumber
	        if jpegDir[-3:] != 'jpg': continue
	        jpeg = cv2.imread(train_dir + jpegDir)
	        specs = re.split('\_', jpegDir)
	        for i in range(0,height):
	                for j in range(0,width):
	                        data[batch_index, i, j, 0] = jpeg[i,j,0]
	        labels[batch_index] = phonemes.index(specs[2])
	        batch_index += 1
	print 'matrix: reloaded'
	return data, labels, numPhonemes, batch_size

def partition_batch(batch_size, subbatch_size, data, labels):
	subbatches = []
	subbatch_labels = []
	n_subbatches = 0
	order = np.random.permutation(batch_size)
	for i in range(0, batch_size, subbatch_size):
		if (i+subbatch_size) in range(batch_size):
			subbatches.append(np.stack(data[order[j]] for j in range(i, i + subbatch_size)))
			subbatch_labels.append(np.stack(labels[order[j]] for j in range(i, i + subbatch_size)))
			n_subbatches = n_subbatches + 1
	subbatches = np.array(subbatches)
	subbatch_labels = np.array(subbatch_labels)
	return subbatches, subbatch_labels, n_subbatches

data, labels, num_classes, batch_size = load_data(train_dir)
# Convert class vectors to binary class matrices.
labels = np_utils.to_categorical(labels, num_classes)

#make smaller batches for train_on_batch
subbatches, subbatch_labels, n_subbatches = partition_batch(batch_size, subbatch_size, data, labels)


model = keras.models.Sequential()
kdim1, kdim2 = kernel_shape
#print num_filters, kdim1, kdim2, data_input_shape
model.add(Convolution2D(num_filters, kdim1, kdim2, input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, kdim1, kdim2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, kdim1, kdim2, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kdim1, kdim2))
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

#nesterov momentum update with 1/t decay
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
for i in range(0, num_epochs):
	t = time.time()
	for b in range(0, n_subbatches):
		metrics = model.train_on_batch(subbatches[b], subbatch_labels[b])
		if b%10 == 0:
			print 'batch ' + str(b) + '/' + str(n_subbatches) + str(metrics)
	if i%5 == 0: 
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(model_json)
		model.save_weights("model.h5")
		print 'model saved'
	print 'step = ' + str(i) + ':' + str(metrics) + 'in' + str(time.time()-t)


