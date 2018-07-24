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
from keras import backend as K

version = sys.version_info.major
if version==2:
    folder = raw_input("Plotting Directory: ")
    epochs = int(raw_input("Epochs Training: "))
    epochs2 = int(raw_input("Epochs Iteration: "))
    sd = float(raw_input("sd Training: "))
    sd2 = float(raw_input("sd Testing: "))
else:
    folder = input("Plotting Directory: ")
    epochs = int(input("Epochs: "))
    epochs2 = int(input("Epochs Iteration: "))
    sd = float(input("sd Training: "))
    sd2 = float(input("sd Testing: "))

parent = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
plot_directory = parent + "/" + folder + "/"

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)




# Training Data -----------------------------------------------------------
bins = 20
ndata = 1000
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=bins, sd=sd, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=bins,sd=sd2, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
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
    plt.hist(xbins, bins, weights=weights_x, label=t[0], alpha=0.5, edgecolor='grey', color='cyan')
    plt.hist(ybins, bins, weights=weights_y, label=t[1], alpha=0.5, edgecolor='grey', color='grey')
    if num==3:
        plt.hist(ybins, bins, weights=weights_predict, label=t[2], alpha=0.5, edgecolor='grey', color='yellow')
    else:
        pass
    plt.legend(loc='upper right', fontsize='x-small')
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.close()

def prior(weight_matrix):
    reg = np.zeros(K.int_shape(weight_matrix))
    reg[0] = class_weights
    return K.variable(reg)

def Model(multi, x_train, y_train, bins):
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(1,)))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(bins, activation='softmax', kernel_regularizer=prior))
    sgd = SGD(lr=multi.lr, decay=1e-6, momentum=0.9, nesterov=True)
    stop = EarlyStopping(patience=5)
    model.compile(loss=multi.loss, optimizer=sgd, metrics=multi.metrics)
    model.fit(x_train, y_train, epochs=multi.epochs, batch_size=multi.batch, callbacks=[stop])
    return model


def BayesIteration(multi, train, test, bins):

    global class_weights
    class_weights = np.zeros(bins)
    model = Model(multi, train.x, train.y, bins)
    k = 0
    score = []
    ndata = len(train.x)

    while True:

        # Testing algorithm on training data for verification
        prediction = model.predict(train.x)
        weights_predict_train = [sum(Column(prediction, i)) for i in range(bins)]
        t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, bins, train.xbins, train.ybins, train.x_weights, train.y_weights, k, plot_directory, "Training", t, weights_predict_train)
        plt.close('all')
        prediction = model.predict(test.x)
        weights_predict = np.asarray([sum(Column(prediction, i)) for i in range(bins)])
        #weights_predict = VecScalar(weights_predict, 1/sum(weights_predict))
        y_weights = []
        for y in train.y_weights:
            if y==0:
                y_weights.append(0.0000000001)
            else:
                y_weights.append(y)
        class_weights = np.asarray([np.log(weights_predict[i]/y_weights[i]) for i in range(bins)])
        a = range(0,bins)
        class_dict = dict([(a[i], class_weights[i]) for i in range(bins)])

        score.append(model.evaluate(test.x, test.y, batch_size=128))

        print("Loss = {0}, Accuracy = {1}" .format(score[k][0], score[k][1]))
        t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, bins, test.xbins, test.ybins, test.x_weights, test.y_weights, k, plot_directory, "Testing",t, weights_predict)
        plt.close('all')
        t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
        PlotHistogram(bins, bins, test.ybins, test.ybins, train.y_weights, test.y_weights, k, plot_directory, "Combined", t, weights_predict)
        plt.close('all')

        sample_weights = SampleWeights(class_weights, train)

        if k==150:
            break
        else:
            #model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, sample_weight=sample_weights)
            #model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, class_weight=class_dict)
            model = Model(multi, train.x, train.y, bins)

            k = k+1

    return model

multi = MultiClassifier()
multi.epochs = epochs
model = BayesIteration(multi, train, test, bins)
