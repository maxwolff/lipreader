import tensorflow as tf
import os, sys
import cv2
import re
from pdb import set_trace as t
import numpy as np

padding = 'SAME'

#hyperParameters
INIT_DEV = 0.05
NUM_STEPS_PER_DECAY = 50
INIT_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.999


#import videos
def inputs(train_dir):
        jpegDirs = os.listdir(train_dir)
        batch_size = 0
        height, width, _ = cv2.imread(train_dir + jpegDirs[1]).shape
        phonemes = []
        for jpegDir in jpegDirs:
                if jpegDir[-3:] != 'jpg': continue
                batch_size += 1
                specs = re.split('\_', jpegDir)
                if specs[2] not in phonemes: phonemes.append(specs[2])
        phonemes = sorted(phonemes)
        numPhonemes = len(phonemes)
        videos = np.empty((batch_size, height, width, 1))
        labels = np.empty((batch_size))
        batch_index = 0
        curSpecs = ['', '', '']
        for jpegDir in jpegDirs:
                #speaker_sentence_phoneme_frameNumber
                if jpegDir[-3:] != 'jpg': continue
                jpeg = cv2.imread(train_dir + jpegDir)
                specs = re.split('\_', jpegDir)
                for i in range(0,height):
                        for j in range(0,width):
                                videos[batch_index, i, j, 0] = jpeg[i,j,0]
                labels[batch_index] = phonemes.index(specs[2])
                batch_index += 1
        videosVar = tf.Variable(tf.zeros([batch_size, height, width, 1]), trainable=False, name = 'videos')
        print 'matrix: reloaded'
        assignOp = videosVar.assign(videos)
        return batch_size, videosVar, labels, assignOp, numPhonemes


#videos = batch_size*numFrames*height*width*numChannels)
def inference(videos, batch_size, numPhonemes):
        #conv1
        filterShape = [5, 5, 1, 1]
        filterTensor = tf.Variable(tf.truncated_normal(filterShape, stddev = INIT_DEV), name = 'filter')
        strides = [1, 2, 2, 1]
        conv = tf.nn.conv2d(videos, filterTensor, strides, padding)

        #rectified linear activation
        relu = tf.nn.relu(conv)

        #pool1
        pool = tf.nn.max_pool(relu, [1,2,2,1], strides, padding)

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