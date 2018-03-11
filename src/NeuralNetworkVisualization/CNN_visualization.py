import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10


import config

def convertWeightFormat(model):
    weights = model.get_weights()
    ret = list()
    for weight in weights:
        ret += list(np.reshape(weight,weight.size))
    return np.asarray(tuple(ret))

def projection(x, minima, d1, d2):
    x = np.reshape(x, minima.shape)
    minima = np.asarray(minima)
    b = x-minima
    A = np.concatenate([d1,d2],axis=1)
    At = np.transpose(A)
    AA = np.dot(At, A)
    AA_inverse = np.linalg.inv(AA)
    ret = np.dot(AA_inverse, np.dot(At, b))
    return ret



model = Sequential()
#We want to output 32 features maps. The kernel size is going to be
#3x3 and we specify our input shape to be 32x32 with 3 channels
#Padding=same means we want the same dimensional output as input
#activation specifies the activation function
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                 activation='relu'))
#20% of the nodes are set to 0
model.add(Dropout(0.2))
#now we add another convolution layer, again with a 3x3 kernel
#This time our padding=valid this means that the output dimension can
#take any form
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
#maxpool with a kernet of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
#In a convolution NN, we neet to flatten our data before we can
#input it into the ouput/dense layer
model.add(Flatten())
#Dense layer with 512 hidden units
model.add(Dense(512, activation='relu'))
#this time we set 30% of the nodes to 0 to minimize overfitting
model.add(Dropout(0.3))
#Finally the output dense layer with 10 hidden units corresponding to
#our 10 classe
model.add(Dense(10, activation='softmax'))
#Few simple configurations
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])


model.load_weights(config.module_dir + "CNN1/cifar10_final.hdf5")
test = convertWeightFormat(model)

minima1 = np.ones(10,1)
minima2 = np.ones(10,1)+np.ones(10,1)
dim=10
d1 = minima2-minima1
d2 = np.asmatrix(np.random.rand(dim, 1)-0.5)

d1 = np.transpose(np.asarray(d1))
d2 = np.transpose(np.asarray(d2))


d2 = d1 - (np.inner(d1, d1)[0,0] / np.inner(d2, d1)[0,0]) * d2
print("d1.d2",np.inner(d1,d2))
