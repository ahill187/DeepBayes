#------------------------------
# Author: Ainsleigh Hill
# Date created: 180606
# Date modified: 180606
# Description: Testing keras
#------------------------------

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input, LSTM, Embedding
from keras.optimizers import SGD
from Matrix import *
import numpy as np
import matplotlib.pyplot as plt

working_directory = "/Users/Work/plots/"
def BinClass(x_train, bins, xbins):

    y_train = []
    for x in x_train:
        xvec = [x for i in range(bins)]
        diff = VecSub(xvec, xbins)
        yclass = [index for index, item in enumerate(diff) if item > 0]
        if yclass==[]:
            yclass=0
        else:
            yclass = max(yclass)
        y_train.append(yclass)
    y_train = keras.utils.to_categorical(y_train, bins)

    return y_train

# Training Data -----------------------------------------------------------

ndata = 10000
mean = 0
sd = round(float(np.random.uniform(0,1)),8)
sd2 = round(float(np.random.uniform(0, 1)), 8)
bins = 30
y_train = np.random.normal(mean, sd, ndata)
x_train = np.random.normal(mean, sd2, ndata)
x_train = x_train + y_train

xrange = [min(x_train), max(x_train)]
yrange = [min(y_train), max(y_train)]
binwidthx = np.abs(xrange[1] - xrange[0]) / bins
xbins = [xrange[0] + i * binwidthx for i in range(bins)]
binwidthy = np.abs(yrange[1] - yrange[0]) / bins
ybins = [yrange[0] + i * binwidthy for i in range(bins)]

y_train = BinClass(y_train, bins, ybins)
x_weights = BinClass(x_train, bins, xbins)

weights_x = [sum(Column(x_weights, i)) for i in range(bins)]
weights_y = [sum(Column(y_train,i)) for i in range(bins)]

plt.hist(xbins, bins, weights=weights_x, label="Detector Data", alpha=0.5, edgecolor='grey')
plt.hist(ybins, bins, weights=weights_y, label="True Data", alpha=0.5, edgecolor='grey')
plt.legend(loc='upper right')
plt.show()

# Testing Data -----------------------------------------------------------------------------------

ndata = 1000
sd = round(float(np.random.uniform(0,1)),8)
sd2 = round(float(np.random.uniform(0, 1)), 8)
y_test = np.random.normal(mean, sd, ndata)
x_test = np.random.normal(mean, sd2, ndata)
x_test = x_test + y_test
y_test = BinClass(y_test, bins, ybins)
x_test_weights = BinClass(x_test, bins, xbins)

weights_x_test = [sum(Column(x_test_weights, i)) for i in range(bins)]
weights_y_test = [sum(Column(y_test,i)) for i in range(bins)]

plt.hist(xbins, bins, weights=weights_x_test, label="Detector Data Test", alpha=0.5, edgecolor='grey')
plt.hist(ybins, bins, weights=weights_y_test, label="True Data Test", alpha=0.5, edgecolor='grey')
plt.legend(loc='upper right')
plt.show()

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
    #plt.show()
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.show()


def Model(multi, x_train, y_train, bins):
    model = Sequential()
    model.add(Dense(bins, activation='relu', input_shape=(1,)))
    model.add(Dropout(0.5))
    model.add(Dense(bins, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(bins, activation='softmax'))
    sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=multi.loss, optimizer=sgd, metrics=multi.metrics)

    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch)

    return model

def BayesIteration(multi, x_train, y_train, bins, x_test, y_test):

    model = Model(multi, x_train, y_train, bins, [], 0)
    k = 0
    score = []
    ndata=len(x_train)

    while True:
        prediction = model.predict(x_train)
        weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
        weights = [w/ndata for w in weights_predict]
        predictiont = model.predict(x_test)
        weights_predict_test = [sum(Column(predictiont, i)) for i in range(bins)]

        score.append(model.evaluate(x_test, y_test, batch_size=128))

        print("Loss = {0}, Accuracy = {1}" .format(score[k][0], score[k][1]))
        PlotHistogram(bins, xbins, ybins, weights_x_test, weights_y_test, weights_predict_test, k, working_directory, "Testing")
        PlotHistogram(bins, xbins, ybins, weights_x, weights_y, weights_predict, k, working_directory, "Training")

        if score[k][0] < 0.002 and score[k][1]>0.8 or k==5:
            break
        else:
            model = Model(multi, x_train, y_train, bins, weights, 1)
            k = k+1

    return score

multi = MultiClassifier()
model = BayesIteration(multi, x_train, y_train, bins, x_test, y_test)

from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))






# New Data -----------------------------------------------------------------------

ndata = 1000
mean = 4.5
sd = 0.2
y_predict_true = np.random.normal(mean, sd, ndata)
y_predict_true.sort()
x_predict = np.random.normal(mean, sd2, ndata)
x_predict = x_predict + y_predict_true
x_predict.sort()
yrange = [min(y_predict_true), max(y_predict_true)]
xrange = [min(x_predict), max(x_predict)]
y_predict = model.predict(x_predict)
x_predict = BinClass(x_predict, bins)
y_predict_true = BinClass(y_predict_true, bins)

binwidth = np.abs(yrange[1] - yrange[0]) / bins
ybins = [yrange[0] + i * binwidth for i in range(bins)]

binwidth = np.abs(xrange[1] - xrange[0]) / bins
xbins = [xrange[0] + i * binwidth for i in range(bins)]

# Making Plots -----------------------------------------------------------------------

weights_predict = [sum(Column(y_predict, i)) for i in range(bins)]
weights_true = [sum(Column(y_predict_true,i)) for i in range(bins)]
weights_smear = [sum(Column(x_predict,i)) for i in range(bins)]

plt.hist(ybins, bins, weights=weights_predict, label="Predicted from Keras", alpha=0.3, edgecolor='grey')
plt.hist(ybins, bins, weights=weights_true, label="True Data", alpha=0.3, edgecolor='grey')
plt.hist(xbins, bins, weights=weights_smear, label="Detector Data", alpha=0.3, edgecolor='grey')
plt.legend(loc='upper right')
plt.show()
