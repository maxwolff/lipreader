import keras
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import keras_build

#hyperparameters
samples_per_epoch = 25
num_epochs = 1000
num_classes = 3
kernel_shape = (3, 3)
train_dir = './easy/'
num_filters = 32
#num_frames = 3


data, labels, data_input_shape = keras_build.read_data(train_dir)



model = keras.models.Sequential()
kdim1, kdim2 = kernel_shape
print num_filters, kdim1, kdim2, data_input_shape
model.add(Convolution2D(num_filters, kdim1, kdim2, input_shape = data_input_shape, dim_ordering = 'tf'))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Convolution2D(num_filters, kdim1, kdim2))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# def accuracy(y_true, y_pred):
# 	keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(data, labels, batch_size = samples_per_epoch, nb_epoch = num_epochs, verbose = 2)