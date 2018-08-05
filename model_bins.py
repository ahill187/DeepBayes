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
import keras.regularizers as regularizers
from Matrix import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from ExampleData0 import *
from plots import *
from models import *
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
ndata = 10000
sd_smear=0.1
reg = 0
lr = 0.00001
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=bins, sd=sd, sd_smear=sd_smear, plot_directory=plot_directory, ndata=ndata, quantiles=1)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=bins,sd=sd2, sd_smear=sd_smear, plot_directory=plot_directory, ndata=ndata, quantiles=2, xbins=train.xbins, ybins=train.ybins)
plt.close('all')

# Create README File--------------------------------------------------------------------------

f = open(plot_directory+'README.txt', 'w+')
text = ["Date: ", "SD Training: ", "SD Testing: ", "SD Smear: ", "Epochs Training: ", "Epochs Testing: ", "Regularization: ", "Optimizer: ", "Layers: ", "Learning Rate: "]
date = datetime.datetime.now()
layers = [1, 30, 30, 20]
info = [str(date.year)+"_"+str(date.month)+"_"+str(date.day), str(sd), str(sd2), str(sd_smear), str(epochs), str(epochs2), str(reg), "Nadam", str(layers), str(lr)]
for i in range(len(text)):
          f.write(text[i]+info[i]+"\n")
f.close()
# Keras Model -------------------------------------------------------------------------------------

def BayesIteration(multi, train, test, bins, reg, iterations):

          model = Model(multi, train.x, train.y, bins, reg)
          k = 0
          score = []
          ndata = len(train.x)
          newbins = Bins(bins, -2, 2, uneven=True)
          y_weights_class = []
          for y in train.y_weights:
                    if y == 0:
                              y_weights_class.append(0.0000000001)
                    else:
                              y_weights_class.append(y)

          while True:

                    # Plot Training Data
                    prediction = model.predict(train.x)
                    t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.x_weights, train.y_weights, train.xbins, train.ybins, k, plot_directory, t, [1,19], bins, "Training_Binned")
                    PlotQuantile(prediction, train.x_weights, train.y_weights, train.xbins, train.ybins, k, plot_directory, t, newbins, bins, "Training_Hist")

                    # Plot Testing Data
                    prediction = model.predict(test.x)
                    t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, test.x_weights, test.y_weights, test.xbins, test.ybins, k, plot_directory, t, [1,19], bins, "Testing_Binned")
                    PlotQuantile(prediction, test.x_weights, test.y_weights, test.xbins, test.ybins, k, plot_directory, t, newbins, bins, "Testing_Hist")

                    #Plot Combined Data
                    t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.y_weights, test.y_weights, train.ybins, train.ybins, k, plot_directory, t, [1,19], bins, "Combined_Binned")
                    PlotQuantile(prediction, train.y_weights, test.y_weights, train.ybins, train.ybins, k, plot_directory, t, newbins, bins, "Combined_Hist")

                    # Get Score
                    score.append(model.evaluate(test.x, test.y, batch_size=128))
                    print("Loss = {0}, Accuracy = {1}".format(score[k][1], score[k][2]))

                    # Calculate Sample Weights
                    weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
                    binerror = BinError(train, model.predict(train.x), bins)
                    error = VecMult(weights_predict, binerror)
                    weights_predict = VecAdd(weights_predict, error)
                    class_weights = np.asarray([weights_predict[i]/y_weights_class[i] for i in range(bins)])

                    a = range(0,bins)
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
multi.lr = lr
model = BayesIteration(multi, train, test, bins, reg, iterations=30)
