from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Embedding
from keras.optimizers import SGD, Nadam
from keras.callbacks import EarlyStopping
import keras.regularizers as regularizers
from Matrix import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from ExampleData0 import *
from keras import backend as K


class MultiClassifier:
          def __init__(self):
                    self.epochs = 10000
                    self.lr = 0.001
                    self.batch = 1000
                    self.metrics = ['acc', 'mse', 'mae']
                    self.loss = 'categorical_crossentropy'


def Model(multi, x_train, y_train, bins, reg):
          model = Sequential()
          model.add(Dense(30, activation='relu', input_shape=(1,)))
          model.add(Dropout(0.5))
          model.add(Dense(30, activation='relu'))
          model.add(Dropout(0.5))
          model.add(Dense(bins, activation='softmax', kernel_regularizer=regularizers.l2(reg)))
          #sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
          nadam = Nadam(lr = multi.lr)
          stop = EarlyStopping(patience=5)
          model.compile(loss=multi.loss, optimizer=nadam, metrics=multi.metrics)
          model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch, callbacks=[stop])
          return model

