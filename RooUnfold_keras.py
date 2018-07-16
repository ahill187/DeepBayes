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




# Training Data -----------------------------------------------------------
bins = 10
binsx=15
#sd = round(float(np.random.uniform(0, 1)), 8)
sd = 0.6
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=binsx, sd=sd, sd_smear=0.1, plot_directory=plot_directory, ndata=5000)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=binsx, sd=0.3, sd_smear=0.1, plot_directory=plot_directory, ndata=5000)
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

def BayesIteration(multi, train, test, bins, iterations):

    model = Model(multi, train.x, train.y, bins)
    score = []
    ndata = len(train.x)
    k=0


    prediction = model.predict(train.x)
    weights_predict_train = [sum(Column(prediction, i)) for i in range(bins)]
    t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
    PlotHistogram(bins, binsx, train.xbins, train.ybins, train.x_weights, train.y_weights, k, plot_directory, "Training", t, weights_predict_train)
    plt.close('all')
    prediction = model.predict(test.x)
    weights_predict = np.asarray([sum(Column(prediction, i)) for i in range(bins)])
    weights_predict = VecScalar(weights_predict, 1/sum(weights_predict))
    y_weights = []
    for y in train.y_weights:
        if y==0:
            y_weights.append(0.0000000001)
        else:
            y_weights.append(y)
    class_weights = np.asarray([weights_predict[i]/y_weights[i] for i in range(bins)])

    score.append(model.evaluate(test.x, test.y, batch_size=128))

    print("Loss = {0}, Accuracy = {1}" .format(score[k][0], score[k][1]))
    t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
    PlotHistogram(bins, binsx, test.xbins, test.ybins, test.x_weights, test.y_weights, k, plot_directory, "Testing",t, VecScalar(weights_predict, ndata))
    plt.close('all')
    t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
    PlotHistogram(bins, binsx, test.ybins, test.ybins, train.y_weights, test.y_weights, k, plot_directory, "Combined", t, VecScalar(weights_predict, ndata))
    plt.close('all')

    sample_weights = SampleWeights(class_weights, train)

    x = train.x
    xbins = train.xbins
    x = BinClass(x, binsx, xbins)

    pxz = np.zeros((binsx, bins))
    for i in range(bins):
        zi = Column(train.y, i)
        xclass = [index for index, item in enumerate(zi) if item == 1]
        if xclass==[]:
            continue
        else:
            pxzi = sum(x[xclass])
            pxzi = VecScalar(pxzi, 1/sum(zi))
            pxz[:, i] = pxzi

    smearing_matrix = pxz
    prior = weights_predict
    eff = [sum(Column(smearing_matrix, i))+0.000000000001 for i in range(bins)]
    x=test.x
    x = BinClass(x, binsx, xbins)
    x = [sum(Column(x, i)) / ndata for i in range(binsx)]

    for k in range(1,iterations):

        post = np.zeros((bins, binsx))
        for i in range(binsx):
            pzi = VecMult(smearing_matrix[i], prior)
            margin = DotProduct(smearing_matrix[i], prior)+0.000000000001
            pzi = VecScalar(pzi, 1 / margin)
            post[:, i] = pzi

        prior = MatVec(post, x)
        prior = VecDivide(prior, eff)
        prior = VecScalar(prior, 1 / sum(prior))
        t = ["Measured Smeared", "True Distribution", "Bayes Unfolded"]
        PlotHistogram(bins, binsx, xbins, test.ybins, VecScalar(x, ndata), test.y_weights, k, plot_directory, "Training", t,
                      weights_predict=VecScalar(prior, ndata), num=3)

    return model

multi = MultiClassifier()
multi.epochs = epochs
model = BayesIteration(multi, train, test, bins, 25)
