import tensorflow as tf
import os, sys
import cv2
import re
from pdb import set_trace as t
import numpy as np

padding = 'SAME'

#hyperParameters
INIT_DEV = 0.05
NUM_STEPS_PER_DECAY = 5
INIT_LEARNING_RATE = 0.3
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.999
NUM_FRAMES = 10


#import videos
def inputs(train_dir):
	jpegDirs = os.listdir(train_dir)
	jpegDirs = sorted(jpegDirs)
	batch_size = len(jpegDirs)/NUM_FRAMES
	height, width, _ = cv2.imread(train_dir + jpegDirs[1]).shape
	curSpecs = ['', '', '']
	batch_index = -1
	frame_index = 0
	videos = np.empty((batch_size, NUM_FRAMES, height, width, 1))
	phonemes = []
	for jpegDir in jpegDirs:
		if jpegDir[-3:] != 'jpg': continue
		specs = re.split('\_', jpegDir)
		if specs[2] not in phonemes: phonemes.append(specs[2])
	phonemes = sorted(phonemes)
	numPhonemes = len(phonemes)
	labels = np.empty((batch_size))
	for jpegDir in jpegDirs:
		#speaker_sentence_phoneme_frameNumber
		if jpegDir[-3:] != 'jpg': continue
		jpeg = cv2.imread(train_dir + jpegDir)
		specs = re.split('\_', jpegDir)
		if specs[0] != curSpecs[0] or specs[1] != curSpecs[1] or specs[2] != curSpecs[2]:
			frame_index = 0
			batch_index += 1
			label = phonemes.index(specs[2])
			labels[batch_index] = label
		for i in range(0,height):
			for j in range(0,width):
				videos[batch_index, frame_index, i, j, 0] = jpeg[i, j, 0]
		frame_index += 1
		curSpecs = specs
	videosVar = tf.Variable(tf.zeros([batch_size, NUM_FRAMES, height, width, 1]), trainable=False, name = 'videos')
	t()
	assignOp = videosVar.assign(videos)
	return batch_size, videosVar, labels, assignOp, numPhonemes


#videos = batch_size*numFrames*height*width*numChannels)
def inference(videos, batch_size, numPhonemes):
	#conv1
	filterShape = [3, 5, 5, 1, 1]
	filterTensor = tf.Variable(tf.truncated_normal(filterShape, stddev = INIT_DEV), name = 'filter1')
	strides = [1, 1, 2, 2, 1]
	conv1 = tf.nn.conv3d(videos, filterTensor, strides, padding, name = 'conv1')
	
	
	#rectified linear activation
	relu1 = tf.nn.relu(conv1, name = 'relu1')

	#pool1
	pool1 = tf.nn.max_pool3d(relu1, [1,1,2,2,1], strides, padding, name = 'pool1')

	#norm1

	#local1 (rectified linear activation)

	filterTensor = tf.Variable(tf.truncated_normal(filterShape, stddev = INIT_DEV), name = 'filter2')
	conv2 = tf.nn.conv3d(pool1, filterTensor, strides, padding, name = 'conv2')
	relu2 = tf.nn.relu(conv2, name = 'relu2')
	pool2 = tf.nn.max_pool3d(relu2, [1,1,2,2,1], strides, padding, name = 'pool2')

	reshape = tf.reshape(pool2, [batch_size, -1])
	dim = reshape1.get_shape()[1].value
	weights1 = tf.Variable(tf.truncated_normal([dim, 200], stddev = INIT_DEV), name = 'weights1')
	biases1 = tf.Variable(tf.zeros([200]), name = 'biases1')



	#softmax classifier: outputs batch_size*numPhonemes matrix
	#randomly intialized weigts, biases are 0
	weights2 = tf.Variable(tf.truncated_normal([200,numPhonemes], stddev = INIT_DEV), name = 'weights2')
	biases2 = tf.Variable(tf.zeros([numPhonemes]), name = 'biases2')
	return tf.add(tf.matmul(reshape2, weight2s), biases2)

#labels is shape [batch_size], where each label is an index in [0, num_classes]
#returns a tensor of shape [batch_size]
#logits is batch_size*numPhonemes, where the numPhonemes entries for each video in the batch
#hold the score corresponding to the likelihood that the video shows that phoneme
#I probably shouldn't be naming the variables in exactly the same way the online example does
def loss(logits, labels, batch_size):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name = 'cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
	classifications = tf.nn.in_top_k(logits, labels, 1)
	return cross_entropy_mean, classifications


#I probably shouldn't be naming the variables in exactly the same way the online example does
def train(total_loss, step, batch_size):
	lr = tf.train.exponential_decay(INIT_LEARNING_RATE, step, NUM_STEPS_PER_DECAY, LEARNING_RATE_DECAY_FACTOR, staircase = True)
	optimizer = tf.train.GradientDescentOptimizer(lr)
	gradients = optimizer.compute_gradients(total_loss)
	train_op = optimizer.apply_gradients(gradients, global_step=step)
	return train_op