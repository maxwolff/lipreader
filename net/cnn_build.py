import tensorflow as tf
import os, sys
import cv2
import re

padding = 'SAME'
numPhonemes = 52

#hyperParameters
INIT_DEV = 0.05
NUM_EXAMPLES_PER_TRAIN_EPOCH = 50000
NUM_EPOCHS_PER_DECAY = 200
INIT_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.999


#import videos
def inputs(train_dir):
	jpegDirs = os.listdir(train_dir)
	batch_size = 0
	numFrames = 1
	numFramesBool = True
	width = 0
	height = 0
	numChannels = 0
	curSpecs = []
	videos = []
	index = -1
	for jpegDir in jpegDirs:
		#speaker_sentence_phoneme_frameNumber
		if jpegDir[-3:] != 'jpg': continue
		jpeg = cv2.imread(train_dir + jpegDir)
		specs = re.split('\_', jpegDir)
		if numFramesBool:
			if specs[0] == curSpecs[0] and specs[1] == curSpecs[1] and specs[2] == curSpecs[2]:
				numFrames += 1
			elif curSpecs != []:
				numFramesBool = False
		if width == 0 and height == 0 and numChannels == 0:
			height, width, numChannels = jpeg.shape
		if specs[0] != curSpecs[0] or specs[1] != curSpecs[1] or specs[2] != curSpecs[2]:
			batch_size += 1
			index += 1
			videos[index] = [jpeg]
		else:
			videos[index].append(jpeg)
		curSpecs = specs
	videosVar = tf.Variable(tf.zeros([batch_size, numFrames, height, width, 3]), trainable=False, name = 'videos')
	videosVar.assign(videos)
	return batch_size, videosVar, labels


#videos = batch_size*numFrames*height*width*numChannels)
def inference(videos, batch_size):
	#conv1
	filterShape = [3, 5, 5, 3, 3]
	filterTensor = tf.Variable(tf.truncated_normal(filterShape, stddev = INIT_DEV), name = 'filter')
	strides = [1, 1, 2, 2, 1]
	conv = tf.nn.conv3d(videos, filterTensor, strides, padding)
	
	#rectified linear activation
	relu = tf.nn.relu(conv)

	#pool1
	pool = tf.nn.max_pool3d(relu, [1,1,2,2,1], strides, padding)

	#norm1

	#local1 (rectified linear activation)


	#softmax classifier: outputs batch_size*numPhonemes matrix
	#randomly intialized weigts, biases are 0
	reshape = tf.reshape(pool, [batch_size, -1])
	dim = reshape.get_shape()[1].value
	weights = tf.Variable(tf.truncated_normal([dim,numPhonemes], stddev = INIT_DEV), name = 'weights')
	biases = tf.Variable(tf.zeros([numPhonemes]), name = 'biases')
	return tf.add(tf.matmul(reshape, weights), biases)

#labels is shape [batch_size], where each label is an index in [0, num_classes]
#returns a tensor of shape [batch_size]
#I probably shouldn't be naming the variables in exactly the same way the online example does
def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name = 'cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
	return cross_entropy_mean


#I probably shouldn't be naming the variables in exactly the same way the online example does
def train(total_loss, step, batch_size):
	num_batches_per_epoch = NUM_EXAMPLES_PER_TRAIN_EPOCH/batch_size
	decay_steps = num_batches_per_epoch*NUM_EPOCHS_PER_DECAY
	lr = tf.train.exponential_decay(INIT_LEARNING_RATE, step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase = True)
	optimizer = tf.train.GradientDescentOptimizer(lr)
	gradients = optimizer.compute_gradients(total_loss)
	train_op = optimizer.apply_gradients(gradients, global_step=step)
	return train_op

