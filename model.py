#------------------------------
# Author: Ainsleigh Hill
# Date created: 180606
# Date modified: 180606
# Description: Testing keras
#------------------------------
from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Embedding
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from Matrix import *
import numpy as np
import matplotlib.pyplot as plt
import os
from ExampleData0 import *

plot_directory = os.getcwd() + "/Plots_0/"

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# Training Data -----------------------------------------------------------
bins = 70
sd = round(float(np.random.uniform(0, 1)), 8)
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, sd=sd, sd_smear=0.3)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, sd=0.3, sd_smear=0.3)
plt.close('all')
# Keras Model -------------------------------------------------------------------------------------

class MultiClassifier:
    def __init__(self):
        self.epochs = 10000
        self.lr = 0.001
        self.batch = 1000
        self.metrics = ['acc', 'mse', 'mae']
        self.loss = 'categorical_crossentropy'

def PlotHistogram(bins, xbins, ybins, weights_x, weights_y, weights_predict, k, wd, a, t):
    plt.hist(xbins, bins, weights=weights_x, label=t[0], alpha=0.5, edgecolor='grey', color='cyan')
    plt.hist(ybins, bins, weights=weights_y, label=t[1], alpha=0.5, edgecolor='grey', color='grey')
    plt.hist(ybins, bins, weights=weights_predict, label=t[2], alpha=0.5, edgecolor='grey', color='yellow')
    plt.legend(loc='upper right', fontsize='x-small')
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.close()

def Model(multi, x_train, y_train, bins):

    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(1,)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(bins, activation='softmax'))
    sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
    stop = EarlyStopping(patience=5)
    model.compile(loss=multi.loss, optimizer=sgd, metrics=multi.metrics)
    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch, callbacks=[stop])
    return model

def BayesIteration(multi, train, test, bins):

    model = Model(multi, train.x, train.y, bins)
    k = 0
    score = []

    while True:

        prediction = model.predict(train.x)
        weights_predict_train = [sum(Column(prediction, i)) for i in range(bins)]
        t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, train.xbins, train.ybins, train.x_weights, train.y_weights, weights_predict_train, k, plot_directory, "Training", t)
        plt.close('all')
        prediction = model.predict(test.x)
        weights_predict = np.asarray([sum(Column(prediction, i)) for i in range(bins)])
        y_weights = []
        for y in train.y_weights:
            if y==0:
                y_weights.append(0.0000000001)
            else:
                y_weights.append(y)
        weights = np.asarray([weights_predict[i]/y_weights[i] for i in range(bins)])

        score.append(model.evaluate(test.x, test.y, batch_size=128))

        print("Loss = {0}, Accuracy = {1}" .format(score[k][0], score[k][1]))
        t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, test.xbins, test.ybins, test.x_weights, test.y_weights, weights_predict, k, plot_directory, "Testing",t)
        plt.close('all')
        t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, test.ybins, test.ybins, train.y_weights, test.y_weights, weights_predict, k, plot_directory, "Combined", t)
        plt.close('all')

        if k==150:
            break
        else:
            model.fit(train.x, train.y, epochs=multi.epochs, batch_size=multi.batch, class_weight=weights)
            k = k+1

    return model

multi = MultiClassifier()
model = BayesIteration(multi, train, test, bins)
