import numpy as np
from keras.models import Model,load_model,Sequential
from keras.layers import Dense, Input,BatchNormalization,Dropout,ActivityRegularization,LeakyReLU
from keras.callbacks import EarlyStopping
import os
import config
import buildData as bd
N=16000
lim=100
epochs=400

encoder=load_model('my_encoder_3.h5')
[Y,none_used]=bd.constructDataWithConstraints(lim,N,dim=config.dim)
encoder_train=encoder.predict(Y)
decoder=Sequential()
decoder.add(Dense(400, activation='relu', input_shape=(encoder_train.shape[1],)))
decoder.add(Dense(200, activation='relu'))
decoder.add(Dense(200, activation='relu'))
decoder.add(Dense(400, activation='relu'))
decoder.add(Dense(50, activation='relu'))
decoder.add(Dense(Y.shape[1],activation='linear'))

decoder.compile(optimizer='Adam',loss='mean_squared_error')
callback = [
    EarlyStopping(monitor='val_loss', patience=2, min_delta=0.01 ,verbose=0)
]
decoder.fit(encoder_train,Y,epochs=epochs,verbose=2)
decoder.save('real_decoder_3.h5')