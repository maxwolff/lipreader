import keras, pickle
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
import os, sys
import cv2
import re

#hyperparameters
samples_per_epoch = 500
num_epochs = 1000
kernel_shape = (3, 3)
num_filters = 32
batch_size = 500


train_dir = './downsize_vidTIMIT_train_1000/'


def load_data(train_dir):
	jpegDirs = os.listdir(train_dir)
	# batch_size = 0
	height, width, _ = cv2.imread(train_dir + jpegDirs[1]).shape
	phonemes = []
	batch_index = 0
	for jpegDir in jpegDirs:
	        if jpegDir[-3:] != 'jpg': continue
	        # batch_size += 1
	        if batch_index >= batch_size: break
	        specs = re.split('\_', jpegDir)
	        if specs[2] not in phonemes: phonemes.append(specs[2])
	        batch_index += 1
	phonemes = sorted(phonemes)
	numPhonemes = len(phonemes)
	data = np.empty((batch_size, height, width, 1))
	labels = np.empty((batch_size))
	batch_index = 0
	for jpegDir in jpegDirs:
	        #speaker_sentence_phoneme_frameNumber
	        if jpegDir[-3:] != 'jpg': continue
	        if batch_index >= batch_size: break
	        jpeg = cv2.imread(train_dir + jpegDir)
	        specs = re.split('\_', jpegDir)
	        for i in range(0,height):
	                for j in range(0,width):
	                        data[batch_index, i, j, 0] = jpeg[i,j,0]
	        labels[batch_index] = phonemes.index(specs[2])
	        batch_index += 1
	print 'matrix: reloaded'
	return data, labels, numPhonemes, batch_size

data, labels, num_classes, batch_size = load_data(train_dir)

print('data shape:', data.shape)
print(data.shape[0], 'train samples')

# Convert class vectors to binary class matrices.
labels = np_utils.to_categorical(labels, num_classes)


model = keras.models.Sequential()
kdim1, kdim2 = kernel_shape
#print num_filters, kdim1, kdim2, data_input_shape
model.add(Convolution2D(num_filters, kdim1, kdim2, input_shape = data.shape[1:]))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# def accuracy(y_true, y_pred):
# 	keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# batch1 = data[1:batch_size/4, :]
# labels1 = labels[1:batch_size/4, :]
# batch2 = data[batch_size/4:batch_size/2, :]
# labels2 = labels[batch_size/4:batch_size/2, :]
# batch3 = data[batch_size/2:3*batch_size/4, :]
# labels3 = labels[batch_size/2:3*batch_size/4, :]
# batch4 = data[3*batch_size/4:, :]
# labels4 = labels[3*batch_size/4:, :]
# print ('batch1 shape:', batch1.shape)

# for i in range(0, num_epochs):
# 	model.train_on_batch(batch1, labels1)
# 	model.train_on_batch(batch2, labels2)
# 	model.train_on_batch(batch3, labels3)
# 	metrics = model.train_on_batch(batch4, labels4)
# 	print 'step = ' + str(i) + ':' + str(metrics)
model.fit(data, labels, batch_size = samples_per_epoch, nb_epoch = num_epochs, verbose = 2)
