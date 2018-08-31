#------------------------------
# Author: Ainsleigh Hill
# Date created: 180606
# Date modified: 180606
# Description: Testing keras
#------------------------------
from __future__ import division
from builtins import input
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
from models import *
from plots import *
from keras import backend as K

# Default Settings
epochs = 15000
epochs2 = 1000
sd = 0.6
sd2 = 0.4
iterations = 30

# User Defined Settings

defaults = input("Use Defaults? (Y or N) ")
if defaults == "Y":
          pass
elif defaults == "N":       
          epochs = int(input("Epochs: "))
          epochs2 = int(input("Epochs Iteration: "))
          sd = float(input("sd Training: "))
          sd2 = float(input("sd Testing: "))
          iterations = int(input("Iterations: "))
folder = input("Plotting Directory: ")
parent = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
plot_directory = parent + "/" + folder + "/"

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# Training Data -----------------------------------------------------------
bins = 20
ndata = 10000
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=bins, sd=sd, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=bins,sd=sd2, sd_smear=0.1, plot_directory=plot_directory, ndata=ndata)
plt.close('all')

# Keras Model -------------------------------------------------------------------------------------
def BayesIteration(multi, train, test, bins, iterations):

          global class_weights
          global y_weights
          y_weights = train.y_weights
          class_weights = np.zeros(bins)
          model = Model(multi, train.x, train.y, bins, 0.5)
          k = 0
          score = []
          ndata = len(train.x)

          y_weights_class= []
          for y in train.y_weights:
                    if y == 0:
                              y_weights_class.append(0.0000000001)
                    else:
                              y_weights_class.append(y)

          while True:

                    # Plot Training Data
                    prediction = model.predict(train.x)
                    t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.x_weights, train.y_weights, train.xbins, train.ybins, k, plot_directory, t, [0, bins], bins, "Training")

                    # Plot Testing Data
                    prediction = model.predict(test.x)
                    t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, test.x_weights, test.y_weights, test.xbins, test.ybins, k, plot_directory, t, [0, bins], bins, "Testing")

                    # Plot Combined Data
                    t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.y_weights, test.y_weights, train.ybins, train.ybins, k, plot_directory, t, [0, bins], bins, "Combined_Binned")

                    # Get Score
                    score.append(model.evaluate(test.x, test.y, batch_size=128))
                    print("Loss = {0}, Accuracy = {1}".format(score[k][1], score[k][2]))

                    # Calculate Sample Weights
                    weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
                    class_weights = np.asarray([weights_predict[i] / y_weights_class[i] for i in range(bins)])
                    a = range(0, bins)
                    class_dict = dict([(a[i], class_weights[i]) for i in range(bins)])
                    sample_weights = SampleWeights(class_weights, train)

                    # Generate Score
                    print(score)
                    print(k)
                    plt.plot(np.arange(start=0, stop=len(score), step=1),Column(score, 0))
                    title = 'Score.png'
                    plt.savefig(plot_directory + title)
                    plt.close('all')

                    if k == iterations:
                              break
                    else:
                              model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, sample_weight=sample_weights)
                              #model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, class_weight=class_dict)
                              #model = Model(multi, train.x, train.y, bins, True)

                    k = k+1

          return model

multi = MultiClassifier()
multi.epochs = epochs
model = BayesIteration(multi, train, test, bins, iterations=iterations)
