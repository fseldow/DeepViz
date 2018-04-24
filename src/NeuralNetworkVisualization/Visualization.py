import numpy as np
import time
import math

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10
from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D

import config
from src.Utils.utils import savePlane

def normalization(d, model):
    weights = convertList2LayoutWeight(d, model)
    for weight in weights:
        weight /= np.linalg.norm(weight)
    model.set_weights(weights)
    return convertWeightFormat(model)

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
    A[:,0]=np.squeeze(d1)
    A[:,1]=np.squeeze(d2)
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

##########################################################################
num_classes = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:50]
y_train = y_train[0:50]
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
###########################################################################################
# reconstruct network
model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(5, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
##################################################################################################

model.load_weights(config.module_dir + "MNIST/weights_%d.hdf5"%(config.Epoch_visual-1))
minima1 = convertWeightFormat(model)

# create W - W0 through the gradient descent path
dim = len(minima1)
w = np.empty((dim, config.Epoch_visual - 1))
for epoch in range(config.Epoch_visual - 1):
    model.load_weights(config.module_dir + 'MNIST/weights_%d.hdf5'%epoch)
    weight = convertWeightFormat(model)
    w[:, epoch] = weight - minima1

# calculate the most three represented direction
"""
W = np.dot(w, np.transpose(w))
eig_v, eig_d = np.linalg.eig(W)
index_sort = sorted(range(len(eig_v)), key=lambda k: eig_v[k], reverse = True)

d1 = np.real(eig_d[:, index_sort[0]])
d21 = np.real(eig_d[:, index_sort[1]])
d22 = np.real(eig_d[:, index_sort[2]])

np.save('MNIST', [d1,d21,d22])
"""
d = np.load('MNIST.npy')
d1 = np.squeeze(d[0,:])
d21 = np.squeeze(d[1,:])
d22 = np.squeeze(d[2,:])

d1 = normalization(d1, model)
d21 = normalization(d21, model)
d22 = normalization(d22, model)
"""
d1 = 
d2 = np.random.rand(dim)-0.5
d2 = convertOthrographic(d1, d2)
"""
lim_alpha = [-50, 50]
lim_beta = [-50, 50]
precise = 100

alpha_range = np.linspace(lim_alpha[0], lim_alpha[1], precise)
beta_range = np.linspace(lim_beta[0], lim_beta[1], precise)


Ground = np.asarray([[i,j] for i in alpha_range for j in beta_range])

nFrame = 40

min_cost = 3
max_cost = 7

for f in range(nFrame):

    start_time = time.time()
    degree = 2.0 * math.pi / nFrame * f
    print('degree %.2f'%degree)

    d2 = d21 * math.cos(degree) + d22 * math.sin(degree)

    trace = np.empty((config.Epoch_visual, dim))
    for i in range(config.Epoch_visual):
        model.load_weights(config.module_dir + "MNIST/weights_%d.hdf5"%i)
        trace[i,:] = convertWeightFormat(model)

    path = np.empty((config.Epoch_visual,2))
    for i in range(config.Epoch_visual):
        x_path = np.transpose(trace[i,:])
        path[i,:] = np.transpose(projection(x_path, minima1, d1, d2))

    fnn = [None] * precise * precise
    count = 0
    for pos in Ground:
        p = minima1 + pos[0] * d1 + pos[1] * d2
        weight = convertList2LayoutWeight(p, model)
        model.set_weights(weight)
        #model.load_weights(config.module_dir + "SimpleNN/weights_final.hdf5")
        ret = model.evaluate(x_train, y_train, verbose = 0)
        #print(time.time()-start_time)
        #print(ret)
        fnn[count] = math.log(ret[0] + 1)
        count += 1
    print('time cost %.2f'% (time.time() - start_time))
    #print(fnn)

    if f==0:
        max_cost = math.ceil(max(fnn)) + 1
        min_cost = math.floor(min(fnn)) - 1
        level = (max_cost - min_cost) / 20

    fig = plt.figure(1)
    triang=tr.Triangulation(np.asarray(Ground[:,0]),np.asarray(Ground[:,1]))
    surf = plt.tricontourf(triang,np.squeeze(fnn),np.arange(min_cost, max_cost, level))#draw contour colors
    plt.colorbar()#draw colorbar
    surf2 = plt.tricontour(triang,np.squeeze(fnn),np.arange(min_cost, max_cost, level))#draw contour lines
    line = plt.plot(np.asarray(path[:,0]),np.asarray(path[:,1]),c='r')
    plt.plot(np.asarray(path[-1,0]),np.asarray(path[-1,1]),marker='x',c='y')
    #plt.set_xlim([min(min(Z1)),max(max(Z1))])
    #plt.set_ylim([min(min(Z2)),max(max(Z2))])
    plt.title("rotate "+"%.2f" % degree)#set title
    fig.savefig(config.data_dir + 'result/rotation_MNIST/frame%d.png'%f)
    fig.clear()
    #plt.show()

    # store data:
    savePlane(config.data_dir + 'result/rotation_MNIST/', alpha= alpha_range, beta=beta_range, loss=fnn, name='surface_rotation_%.2f.csv' % degree)