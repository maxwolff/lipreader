import pickle

file = '/Users/boraerden/Desktop/221 project stuff/cifar-10/cifar-10-batches-py/data_batch_1'

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

dataset = unpickle(file)

print dataset['batch_label']