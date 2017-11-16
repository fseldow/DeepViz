#
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from keras.models import Model,optimizers,Sequential
from keras.layers import Dense, Input,BatchNormalization,Dropout,ActivityRegularization,LeakyReLU
from keras.callbacks import EarlyStopping
from buildData import constructData
from buildData import constructDataWithConstraints
from buildData import constructDataMAE
from mpl_toolkits.mplot3d import Axes3D
import os
import config

dim=config.dim

# Encoder
lim1=10#limit the range of values
N=20000 #number of points in the train set
N_test=1000# number of points in the test set
epochs=50
#randomly generate points between lim and -lim


[X_train,fnn]=constructDataWithConstraints(lim1,N,dim=dim)
#[X_train,fnn]=constructDataMAE(10,N)



#build model
input = Input(shape=(X_train.shape[1],))
hidden1=LeakyReLU(alpha=-1000)(input)
hidden2=Dense(600,activation='relu')(hidden1)
hidden3=Dense(500,activation='relu')(hidden2)
hidden4=Dense(2,activation='relu')(hidden3)
hidden5=BatchNormalization(scale=True)(hidden4)
hidden6=Dense(200,activation='relu')(hidden5)
hidden7=Dense(150,activation='relu')(hidden6)
hidden7_1=Dense(100,activation='relu')(hidden7)
hidden8=Dense(50,activation='relu')(hidden7_1)
output=Dense(1, activation='linear')(hidden8)

encoder = Model(input, hidden5)
model = Model(input,output)


decoder_input=Input(shape=(2,))
decoder_layer1=model.layers[-5](decoder_input)
decoder_layer2=model.layers[-4](decoder_layer1)
decoder_layer3=model.layers[-3](decoder_layer2)
decoder_layer4=model.layers[-2](decoder_layer3)
decoder_layer5=model.layers[-1](decoder_layer4)
decoder=Model(decoder_input,decoder_layer5)


# compile the model
rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error',
              optimizer='adam')
# train the model with the train and validation data
callback = [
    EarlyStopping(monitor='loss', patience=4, min_delta=0.01 ,verbose=0)
]
model.fit(X_train, fnn, epochs=epochs, verbose=1,callbacks=callback)

model.save('my_model_3.h5')
encoder.save('my_encoder_3.h5')
decoder.save('my_decoder_3.h5')
#calculate minimum point

'''
#plot
fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(np.asarray(Z[:,0]),np.asarray(Z[:,1]),np.squeeze(Y_test),cmap='coolwarm',
                       linewidth=0.2, antialiased=False)
ax.scatter(z1_center,z2_center,0,c='r')
plt.colorbar(surf)
plt.show()
'''
'''
[X_test,fnn_test]=constructData(10,N_test)
low_dim = Encoder.predict(X_test) # low dimensional output
fnn_result=model.predict(X_test)

low1=np.asarray(low_dim[:,0])
low2=np.asarray(low_dim[:,1])
triang=tr.Triangulation(low1,low2)



plt.figure(figsize=(9,4)) #generate figure with a size of 9x4 inches
plt.subplot(131)#subplot 1 row 2 columns the first item
plt.tricontourf(triang,np.squeeze(fnn_test))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(fnn_test))#draw contour lines
plt.title("True Objective")#set title

plt.subplot(132)#subplot 1 row 2 columns the first item
plt.tricontourf(triang,np.squeeze(fnn_result))#draw contour colors
plt.colorbar()#draw colorbar
plt.tricontour(triang,np.squeeze(fnn_result))#draw contour lines
plt.title("Test Objective")#set title
plt.show()
'''