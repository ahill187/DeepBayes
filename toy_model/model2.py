#------------------------------
# Author: Ainsleigh Hill
# Date created: 180705
# Date modified: 180705
# Description: Testing keras
#------------------------------

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Embedding
from keras.optimizers import SGD
import os
from ExampleData import *
from Matrix import *
import numpy as np
import matplotlib.pyplot as plt

plot_directory = os.getcwd() + "/Plots/"

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

training = CreateTrainingData()
testing = CreateTestingData()
x_train = training[0]
y_train = training[1]
x_test = testing[0]
y_test = testing[1]

# Keras Model -------------------------------------------------------------------------------------

class MultiClassifier:
    def __init__(self):
        self.epochs = 300
        self.lr = 0.001
        self.batch = 1000
        self.metrics = ['acc']
        self.loss = 'categorical_crossentropy'

def PlotHistogram(bins, xbins, ybins, weights_x, weights_y, weights_predict, k, wd, a):
    plt.hist(xbins, bins, weights=weights_x, label="Detector Data", alpha=0.3, edgecolor='grey')
    plt.hist(ybins, bins, weights=weights_y, label="True Data", alpha=0.3, edgecolor='grey')
    plt.hist(ybins, bins, weights=weights_predict, label="Predicted from Keras", alpha=0.3, edgecolor='grey')
    plt.legend(loc='upper right')
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.close()


def Model(multi, x_train, y_train, bins, ngauss=1000):

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=ngauss))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(bins, activation='softmax'))
    sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=multi.loss, optimizer=sgd, metrics=multi.metrics)
    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch)
    return model

bins = 50
multi = MultiClassifier()
model = Model(multi, x_train, y_train, bins)
score = model.evaluate(np.expand_dims(x_test, axis=0), y_test, batch_size=128)
print("Loss = {0}, Accuracy = {1}" .format(score[0], score[1]))

prediction = model.predict(x_train)
weights_predict = prediction[m] * 1000
xbins = Bins(bins, -2, 2)
ybins = Bins(bins, -2, 2)
PlotHistogram(bins, xbins, ybins, weights_x, weights_y, weights_predict, k, working_directory, "Training")
prediction = model.predict(np.expand_dims(x_test, axis=0))
weights_predict = prediction[0] * 1000
weights = np.asarray([w / ndata for w in weights_predict])

PlotHistogram(bins, xbins, ybins, weights_x_test, weights_y_test, weights_predict, k, working_directory, "Testing")






