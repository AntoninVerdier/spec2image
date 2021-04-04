
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import keras
import librosa
import numpy as np
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras import Sequential
from scipy.io import wavfile


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sample, samplerate = librosa.load('../Samples/000005.wav')
n_samples = int(len(sample)//(samplerate*0.04))

samples = np.array_split(np.array(sample), list(range(0, len(sample), 750))[1:])[:-1]
samples = np.array(samples).reshape(-1, int(samplerate*0.04))

X_train, X_test = train_test_split(samples, test_size=0.2)

X_train = X_train.reshape(-1, int(samplerate*0.04), 1)
X_test = X_test.reshape(-1, int(samplerate*0.04), 1)

# data's shape is (n_sample, n_timepoints, n_feature) or (600, 882, 1) given that sound is mono (1 channel)

# This is our input image
model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(X_train.shape[1:])))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file='lstm_autoencoder.png')

model.fit(X_train, X_train,
                epochs=10,
                batch_size=8)


output = model.predict(X_train)
output = (output- np.min(output))/np.max(output)

wavfile.write('sound_reconstruction.wav', samplerate, output.reshape(600*882))
