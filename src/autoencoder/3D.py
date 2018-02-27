#
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tr
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from src.dataGeneration.buildData import constructData

# Encoder
lim1=10#limit the range of values
N=10000 #number of points in the train set
N_test=1000# number of points in the test set
epochs=400
#randomly generate points between lim and -lim
[X_train,fnn]=constructData(10,N)


input = Input(shape=(X_train.shape[1],))
#hidden0=BatchNormalization()(input)
hidden1 = Dense(5, activation='relu')(input)
hidden2 = Dense(2, activation='relu')(hidden1)
hidden2_2=Dense(10,activation='relu',input_shape=(2,))(hidden2)
output = Dense(1, activation='relu')(hidden2_2)

Encoder = Model(input, hidden2)
model = Model(input,output)




# compile the model
model.compile(loss='mean_squared_error',
              optimizer='Adam')
# train the model with the train and validation data

callback = [
    EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)
]
model.fit(X_train, fnn, epochs=epochs, verbose=1,callbacks=callback,validation_split=0.2)



[X_test,fnn_test]=constructData(10,N_test)
low_dim = Encoder.predict(X_test) # low dimensional output
fnn_result=model.predict(X_test)

low1=np.asarray(low_dim[:,0])
low2=np.asarray(low_dim[:,1])
triang=tr.Triangulation(low1,low2)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_wireframe(low1, low2, np.squeeze(fnn_test), linewidth=0.2, antialiased=True)
ax.scatter(1,1,0,c='r')
fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_wireframe(low1, low2, np.squeeze(fnn_result), linewidth=0.2, antialiased=True)
plt.show()
