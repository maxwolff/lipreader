import pickle
import numpy as np
from pdb import set_trace as t

file = '/Users/boraerden/Desktop/221 project stuff/cifar-10/cifar-10-batches-py/data_batch_1'

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

batch = unpickle(file)
batch_size = len(batch['labels'])

#batch size, width, height, n_channels
width = 32
height = 32
keras_input_matrix = np.empty((batch_size, width, height, 3))
for i in range(batch_size):
	keras_input_matrix[i,:,:,0] = np.reshape(batch['data'][i][0:1024], (width, height))
	keras_input_matrix[i,:,:,1] = np.reshape(batch['data'][i][1024:2048], (width, height))
	keras_input_matrix[i,:,:,2] = np.reshape(batch['data'][i][2048:3072], (width, height))

bool_labels = np.zeros((len(batch['labels']),10))
for i in range(batch_size):
	bool_labels[i, batch['labels']] = 1


