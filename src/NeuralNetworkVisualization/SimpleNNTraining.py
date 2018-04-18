import config
import numpy as np
from src.Utils.buildData import constructData
from keras.models import Model,optimizers
from keras.models import Sequential
from keras.layers import Dense, Input,BatchNormalization

epochs = config.Epoch_visual

dim = config.dim # dimension of training dataset
lim = 10 # limit the range of values
N = 1000 # number of points in the train set
[X_train, fnn] = constructData(lim, N, dim=dim) # construct dataset
np.save(config.module_dir + "SimpleNN/X_train", X_train)
np.save(config.module_dir + "SimpleNN/fnn", fnn)

# building network
model = Sequential()
model.add(Dense(30, input_shape = (X_train.shape[1],), activation = 'relu'))
model.add(Dense(30, activation = 'selu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam')

for epoch in range(epochs):
    print('step %d'% epoch)
    model.fit(X_train, fnn, epochs=1)
    model.save_weights(config.module_dir + 'SimpleNN/weights_%d.hdf5' %epoch)

model.save_weights(config.module_dir + "SimpleNN/" + 'weights_final' + ".hdf5")
