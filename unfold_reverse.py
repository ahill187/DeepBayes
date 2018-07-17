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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from ExampleData0 import *

version = sys.version_info.major
if version==2:
    folder = raw_input("Plotting Directory: ")
    epochs = int(raw_input("Epochs Training: "))
    epochs2 = int(raw_input("Epochs Iteration: "))
else:
    folder = input("Plotting Directory: ")
    epochs = int(input("Epochs: "))
    epochs2 = int(input("Epochs Iteration: "))

parent = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
plot_directory = parent + "/" + folder + "/"

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

def GenerateData(ndata, weights, bins):
    y = np.zeros((ndata+30, bins))
    j=0
    k=1
    i=0
    for m in range(ndata+30):
        y[i, j] = 1.0
        if round(weights[j])==0:
            y[i,j]=0
            i=i-1
        if k==round(weights[j]) or round(weights[j])==0:
            k = 1
            j = j+1
            if j==bins:
                break
            i=i+1
        else:
            k = k+1
            i=i+1
    for a in range(len(y)):
        if DotProduct(y[a], y[a])==0:
            break
    y = y[0:a]
    return y

def SmearingMatrix(model, bins, binsx, weights, ndata):
    mat = np.zeros((binsx, bins))
    ysort = GenerateData(ndata, weights, bins)
    prediction = model.predict(ysort)
    k=0
    for i in range(bins):
        a = int(round(weights[i]))
        if a==0:
            continue
        else:
            a = k + a
            pxzi = VecScalar(sum(prediction[k:a]), 1/(a-k))
            mat[:,i] = pxzi
            k = a
    return mat


# Training Data -----------------------------------------------------------
bins = 20
binsx=15
#sd = round(float(np.random.uniform(0, 1)), 8)
sd = 0.6
ndata=20000
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=binsx, sd=sd, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=binsx, sd=0.3, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
plt.close('all')
# Keras Model -------------------------------------------------------------------------------------

class MultiClassifier:
    def __init__(self):
        self.epochs = 10000
        self.lr = 0.001
        self.batch = 1000
        self.metrics = ['acc', 'mse', 'mae']
        self.loss = 'categorical_crossentropy'

def PlotHistogram(bins, binsx, xbins, ybins, weights_x, weights_y, k, wd, a, t, weights_predict=[],num=3):
    plt.hist(xbins, binsx, weights=weights_x, label=t[0], alpha=0.5, edgecolor='grey', color='cyan')
    plt.hist(ybins, bins, weights=weights_y, label=t[1], alpha=0.5, edgecolor='grey', color='grey')
    if num==3:
        plt.hist(ybins, bins, weights=weights_predict, label=t[2], alpha=0.5, edgecolor='grey', color='yellow')
    else:
        pass
    plt.legend(loc='upper right', fontsize='x-small')
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.close()

def Model(multi, x_train, y_train, binsx, bins):
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(bins,)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(binsx, activation='softmax'))
    sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
    stop = EarlyStopping(patience=5)
    model.compile(loss=multi.loss, optimizer=sgd, metrics=multi.metrics)
    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch, callbacks=[stop])
    return model

def BayesIteration(multi, train, test, bins, iterations):


    x = train.x
    xbins = train.xbins
    x = BinClass(x, binsx, xbins)
    model = Model(multi, train.y, x, binsx, bins)
    score = []
    ndata = len(train.x)
    k=0


    prediction = model.predict(train.y)
    weights_predict_train = [sum(Column(prediction, i)) for i in range(binsx)]
    t = ["MC True", "MC Smeared", "Predicted from Keras"]
    PlotHistogram(binsx, bins, train.ybins, train.xbins, train.y_weights, train.x_weights, k, plot_directory, "Training", t, weights_predict_train)
    plt.close('all')
    prediction = model.predict(test.y)
    weights_predict = np.asarray([sum(Column(prediction, i)) for i in range(binsx)])

    t = ["Measured True", "Measured Smeared", "Predicted from Keras"]
    PlotHistogram(binsx, bins, test.ybins, test.xbins, test.y_weights, test.x_weights, k, plot_directory, "Testing",t, weights_predict)
    plt.close('all')

    smearing_matrix = SmearingMatrix(model, bins, binsx, train.y_weights, ndata)
    x=test.x
    x = BinClass(x, binsx, xbins)
    x = [sum(Column(x, i)) / ndata for i in range(binsx)]
    #eff = [sum(Column(smearing_matrix, i)) + 0.000000000001 for i in range(bins)]
    prior = VecScalar(train.y_weights, 1/ndata)

    for k in range(1,iterations):

        smearing_matrix = SmearingMatrix(model, bins, binsx, VecScalar(prior, ndata), ndata)
        post = np.zeros((bins, binsx))
        for i in range(binsx):
            pzi = VecMult(smearing_matrix[i], prior)
            margin = DotProduct(smearing_matrix[i], prior)+0.000000000001
            pzi = VecScalar(pzi, 1 / margin)
            post[:, i] = pzi

        prior = MatVec(post, x)
        #prior = VecDivide(prior, eff)
        prior = VecScalar(prior, 1 / sum(prior))
        t = ["Measured Smeared", "True Distribution", "Bayes Unfolded"]
        PlotHistogram(bins, binsx, xbins, test.ybins, VecScalar(x, ndata), test.y_weights, k, plot_directory, "Training", t,
                      weights_predict=VecScalar(prior, ndata), num=3)
        plt.close('all')


multi = MultiClassifier()
multi.epochs = epochs
model = BayesIteration(multi, train, test, bins, 25)





