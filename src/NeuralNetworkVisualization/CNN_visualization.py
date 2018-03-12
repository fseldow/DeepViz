import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10


import config

def convertWeightFormat(model):
    """convert a multi-layer model weights into a single array with shape(N,)"""
    weights = model.get_weights()
    ret = []
    for weight in weights:
        ret += list(np.reshape(weight,weight.size))
    return np.asarray(tuple(ret))

def convertList2LayoutWeight(parameter, model):
    """convert a array with shape (N,) to multi layer weights"""
    ret = []
    weights_format = model.get_weights()
    end=0
    start=0
    for format in weights_format:
        end += format.size
        p = parameter[start:end]
        temp = np.reshape(p, format.shape)
        ret.append(temp)
        start=end
    return ret

def projection(x, minima, d1, d2):
    """Project a point x onto space determined by d1, d2"""
    x = np.reshape(x, minima.shape)
    minima = np.asarray(minima)
    b = x-minima
    A = np.ones((len(d1),2))
    A[:,0]=d1
    A[:,1]=d2
    #A = np.concatenate([d1,d2],axis=1)
    At = np.transpose(A)
    AA = np.dot(At, A)
    AA_inverse = np.linalg.inv(AA)
    ret = np.dot(AA_inverse, np.dot(At, b))
    return ret

def convertOthrographic(d1, d2):
    """adjust d2 to make d1 othrographic to d2
        usage: d2= convertOthrographic(d1,d2)
    """
    d1 = np.transpose(np.asarray(d1))
    d2 = np.transpose(np.asarray(d2))
    temp = np.inner(d1, d1)

    d2 = d1 - (np.inner(d1, d1) / np.inner(d2, d1)) * d2

    return d2



###########################################################################################
#Now we can go ahead and create our Convolution model
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
##################################################################################################

model.load_weights(config.module_dir + "CNN1/cifar10_final.hdf5")
minima1 = convertWeightFormat(model)


model.load_weights(config.module_dir + "CNN2/cifar10_final.hdf5")
minima2 = convertWeightFormat(model)

dim = len(minima1)
d1 = minima2-minima1
d2 = np.random.rand(dim)-0.5
d2 = convertOthrographic(d1, d2)

"""
trace1 = []
trace2 = []
for i in range(25):
    model.load_weights(config.module_dir + "CNN1/cifar10_"+str(i)+".hdf5")
    trace1.append(convertWeightFormat(model))

    model.load_weights(config.module_dir + "CNN2/cifar10_" + str(i) + ".hdf5")
    trace2.append(convertWeightFormat(model))

trace_map1 = []
trace_map2 = []
for i in range(25):
    trace_map1.append(projection(trace1[i], minima1, d1, d2))
    trace_map2.append(projection(trace2[i], minima1, d1, d2))
"""

x_range = range(-100,100)
y_range = range(-100,100)
Ground = np.asarray([[i,j] for i in x_range for j in y_range])
#Lets start by loading the Cifar10 data

(X, y), (X_test, y_test) = cifar10.load_data()
y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)
fnn=[]
for pos in Ground:
    p = minima1 + pos[0] * d1 + pos[1] * d2
    weight = convertList2LayoutWeight(p, model)
    model.set_weights(weight)
    #model.load_weights(config.module_dir + "CNN2/cifar10_final.hdf5")
    print(model.evaluate(X, y)[1] )
    fnn.append(1)
print(Ground)



