import numpy as np
import time
import math

from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import matplotlib.tri as tr
from mpl_toolkits.mplot3d import axes3d, Axes3D


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


X_train = np.load(config.module_dir + "SimpleNN/X_train.npy")
y = np.load(config.module_dir + "SimpleNN/fnn.npy")
###########################################################################################
# reconstruct network
model = Sequential()
model.add(Dense(30, input_shape = (X_train.shape[1],), activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam')
##################################################################################################

model.load_weights(config.module_dir + "SimpleNN/weights_final.hdf5")
minima1 = convertWeightFormat(model)


model.load_weights(config.module_dir + "SimpleNN/weights_final.hdf5")
minima2 = convertWeightFormat(model)

# create W - W0 through the gradient descent path
dim = len(minima1)
w = np.empty((dim, config.Epoch_visual - 1))
for epoch in range(config.Epoch_visual - 1):
    model.load_weights(config.module_dir + 'SimpleNN/weights_%d.hdf5'%epoch)
    weight = convertWeightFormat(model)
    w[:, epoch] = weight - minima1

# calculate the most three represented direction
W = np.dot(w, np.transpose(w))
eig_v, eig_d = np.linalg.eig(W)
index_sort = sorted(range(len(eig_v)), key=lambda k: eig_v[k], reverse = True)

d1 = eig_d[:, index_sort[0]]
d21 = eig_d[:, index_sort[1]]
d22 = eig_d[:, index_sort[2]]

"""
d1 = np.random.rand(dim)-0.5
d2 = np.random.rand(dim)-0.5
d2 = convertOthrographic(d1, d2)
"""
lim_alpha = [-2, 2]
lim_beta = [-2, 2]
precise = 50

alpha_range = np.linspace(lim_alpha[0], lim_alpha[1], precise)
beta_range = np.linspace(lim_beta[0], lim_beta[1], precise)


Ground = np.asarray([[i,j] for i in alpha_range for j in beta_range])

nFrame = 40
for f in range(nFrame):

    start_time = time.time()
    degree = 2.0 * math.pi / nFrame * f
    print('degree %.2f'%degree)

    d2 = d21 * math.cos(degree) + d22 * math.sin(degree)

    fnn = [None] * precise * precise
    count = 0
    for pos in Ground:
        p = minima1 + pos[0] * d1 + pos[1] * d2
        weight = convertList2LayoutWeight(p, model)
        model.set_weights(weight)
        #model.load_weights(config.module_dir + "SimpleNN/weights_final.hdf5")
        ret = model.evaluate(X_train, y, verbose = 0)
        #print(time.time()-start_time)
        #print(ret)
        fnn[count] = math.log(ret + 1)
        count += 1
    #print('time cost %.2f'% (time.time() - start_time))
    #print(fnn)

    fig = plt.figure(1)
    triang=tr.Triangulation(np.asarray(Ground[:,0]),np.asarray(Ground[:,1]))
    surf = plt.tricontourf(triang,np.squeeze(fnn),np.arange(3, 7, 0.5))#draw contour colors
    plt.colorbar()#draw colorbar
    surf2 = plt.tricontour(triang,np.squeeze(fnn),np.arange(3, 7, 0.5))#draw contour lines
    #line = plt.plot(np.asarray(path[:,0]),np.asarray(path[:,1]),c='r')
    #plt.plot(np.asarray(path[-1,0]),np.asarray(path[-1,1]),marker='x')
    #plt.set_xlim([min(min(Z1)),max(max(Z1))])
    #plt.set_ylim([min(min(Z2)),max(max(Z2))])
    plt.title("rotate "+"%.2f" % degree)#set title
    fig.savefig(config.data_dir + 'result/rotation/frame%d.png'%f)
    fig.clear()
    #plt.show()