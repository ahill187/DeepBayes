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
# matplotlib.use('Agg')
matplotlib.use('TKAgg')
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
train = CreateExampleData(title="ExampleTrainingData.pdf", bins=bins, binsx=bins, sd=sd, sd_smear=0.3, plot_directory=plot_directory, ndata=ndata, quantiles=1)
plt.close('all')

# Testing Data -----------------------------------------------------------------------------------

test = CreateExampleData(title="ExampleTestingData.pdf", bins=bins, binsx=bins,sd=sd2, sd_smear=0.3, plot_directory=plot_directory, ndata=ndata, quantiles=2, xbins=train.xbins, ybins=train.ybins)
plt.close('all')



# Parameters
reg = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
lr = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10,50]
epochs = "1000 - 10000"

# Create README File--------------------------------------------------------------------------

f = open(plot_directory+'README.txt', 'w+')
text = ["Date: ", "SD Training: ", "SD Testing: ", "SD Smear: ", "Epochs: ", "Regularization: ", "Learning Rate: ",  "Optimizer: "]
date = datetime.datetime.now()
info = [str(date.year)+"_"+str(date.month)+"_"+str(date.day), str(sd), str(sd2), str(sd_smear), str(epochs), str(reg[1]), str(lr[0]), "Nadam"]
for i in range(len(text)):
          f.write(text[i]+info[i]+"\n")
f.close()

# Model
def TestIteration(multi, train, test, bins, parameter, reg0, iterations=150):

          model = Model(multi, train.x, train.y, bins, reg0)
          epochs = multi.epochs
          epochs2 = epochs
          param = epochs
          score = []
          ndata = len(train.x)
          newbins = Bins(bins, -2, 2, uneven=True)
          y_weights_class = []
          for y in train.y_weights:
                    if y == 0:
                              y_weights_class.append(0.0000000001)
                    else:
                              y_weights_class.append(y)

          for k in range(iterations):

                    # Plot Training Data
                    prediction = model.predict(train.x)
                    t = ["MC Smeared", "MC True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.x_weights, train.y_weights, train.xbins, train.ybins, param, plot_directory, t, [1,19], bins, "Training_Binned")
                    PlotQuantile(prediction, train.x_weights, train.y_weights, train.xbins, train.ybins, param, plot_directory, t, newbins, bins, "Training_Hist")

                    # Plot Testing Data
                    prediction = model.predict(test.x)
                    t = ["Measured Smeared", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, test.x_weights, test.y_weights, test.xbins, test.ybins, param, plot_directory, t, [1,19], bins, "Testing_Binned")
                    PlotQuantile(prediction, test.x_weights, test.y_weights, test.xbins, test.ybins, param, plot_directory, t, newbins, bins, "Testing_Hist")

                    # Plot Combined Data
                    t = ["MC True Distribution", "Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, train.y_weights, test.y_weights, train.ybins, train.ybins, param, plot_directory, t, [1,19], bins, "Combined_Binned")
                    PlotQuantile(prediction, train.y_weights, test.y_weights, train.ybins, train.ybins, param, plot_directory, t, newbins, bins, "Combined_Hist")

                    # Get Score
                    score.append(model.evaluate(train.x, train.y, batch_size=128))
                    print("Loss = {0}, Accuracy = {1}".format(score[k][1], score[k][2]))

                    # Calculate Sample Weights
                    if k == 0:
                              weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
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

                    if parameter == "epoch":
                              epochs2 = epochs2 + epochs
                              model.fit(train.x, train.y, epochs=epochs, batch_size=multi.batch)
                              param = epochs2
                    elif parameter == "regularization":
                              model = Model(multi, train.x, train.y, bins, reg[k])
                              param = reg[k]
                              # model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, class_weight=class_dict)
                    elif parameter == "learning_rate":
                              multi.lr = lr[k]
                              model = Model(multi, train.x, train.y, bins, reg0)
                              param = lr[k]
                    elif parameter == "bayes":
                              epochs2 = 1000
                              model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, sample_weight=sample_weights)
                              param = param + epochs2
          return score

multi = MultiClassifier()
multi.epochs = epochs
multi.lr = 0.00001
score= TestIteration(multi, train, test, bins, parameter="bayes", reg0=0, iterations=30)

