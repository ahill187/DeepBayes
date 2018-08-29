from __future__ import division
from Matrix import *
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import sys
import os
import datetime
from deep_bayes.get_data import getTrees, getModel, getTestingData, getTrainingData, compileModel, fitModel, reweightData, MODELLIST, fitModelFast
from deep_bayes.data_helper_functions import BinError, SampleWeights
from deep_bayes.plots import PlotData
from deep_bayes.classes import MultiClassifier

folder = "/home/ahill/output_directory/plots/"
epochs=20
epochs2=20
plot_directory = folder

if not os.path.exists(plot_directory):
          os.makedirs(plot_directory)

# Create README File--------------------------------------------------------------------------

f = open(plot_directory+'README.txt', 'w+')
text = ["Date: ", "Epochs Training: ", "Epochs Testing: ", "Regularization: ", "Optimizer: ", "Layers: ", "Learning Rate: "]
date = datetime.datetime.now()
layers = [1, 30, 30, 20]
lr=0.00001
reg=0
info = [str(date.year)+"_"+str(date.month)+"_"+str(date.day), str(epochs), str(epochs2), str(reg), "Nadam", str(layers), str(lr)]
for i in range(len(text)):
          f.write(text[i]+info[i]+"\n")
f.close()

# Training Data ---------------------------------------------------------------------
bins = 20
fast = True
n_train = 10000
multi = MultiClassifier()
train = getTrees(bins)
model_num = int(train.keras_model_method)
MODELLIST[model_num]['nepochs'] = epochs
nbatches = 10
train = getModel(train)
train = compileModel(train, multi)
import cProfile
if fast == True:
          data = getTrainingData(train, n_train)
          model = fitModelFast(train, data)
else:
          model_hist = fitModel(train, multi)
cProfile.runctx('fitModel(train, multi, nbatches)', globals(), locals(), sort='tottime')
model_hist = fitModel(train, multi, nbatches, fast=True)
model = model_hist[0]
history = model_hist[1]

# Keras Model -------------------------------------------------------------------------------------

def BayesIteration(model, train, bins, epochs2, iterations):

          ndata=1000
          data = getTrainingData(train, ndata)
          reweight = reweightData(data.y_weights, ndata)
          test = getTestingData(train, reweight, ndata)
          y_weights = data.y_weights
          class_weights = np.zeros(bins)

          k = 0
          score = []

          y_weights_class= []
          for y in data.y_weights:
                    if y == 0:
                              y_weights_class.append(0.0000000001)
                    else:
                              y_weights_class.append(y)

          while True:

                    # Plot Training Data
                    prediction = model.predict(np.asarray(data.x))
                    t = ["MC True Distribution", "Predicted from Keras"]
                    PlotData(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Training")

                    # Plot Testing Data
                    prediction = model.predict(np.asarray(test.x))
                    t = ["Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, test.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Testing")

                    # Plot Combined Data
                    t = ["MC True Distribution", "Predicted from Keras", "Measured True Distribution"]
                    PlotData(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Combined_Binned", weights_3=test.y_weights, num=3)

                    # Get Score
                    score.append(model.evaluate(np.asarray(data.x), np.asarray(data.y), batch_size=128))
                    print("Loss = {0}, Accuracy = {1}".format(score[k][1], score[k][2]))

                    # Calculate Sample Weights
                    weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
                    binerror = BinError(data.y, model.predict(np.asarray(data.x)), bins)
                    error = VecMult(weights_predict, binerror)
                    weights_predict = VecAdd(weights_predict, error)
                    class_weights = np.asarray([weights_predict[i]/y_weights_class[i] for i in range(bins)])

                    a = range(0,bins)
                    class_dict = dict([(a[i], class_weights[i]) for i in range(bins)])
                    sample_weights = SampleWeights(class_weights, data.x, data.y)

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
                              #model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, sample_weight=sample_weights)
                              model.fit_generator(train.train_data.generator(),
                                                             steps_per_epoch=train.train_data.getNBatchesPerEpoch(),
                                                             epochs=epochs2,
                                                             #callbacks=train.callbacks.callbacks,
                                                             #validation_data=train.val_data.generator(),
                                                             #validation_steps=train.val_data.getNBatchesPerEpoch(),
                                                             max_q_size=5, class_weight=class_dict)
                              #model.fit(train.x, train.y, epochs=epochs2, batch_size=multi.batch, class_weight=class_dict)
                              #model = Model(multi, train.x, train.y, bins, True)

                    k = k+1

          return model

multi = MultiClassifier()
multi.epochs = epochs
model = BayesIteration(model, train, bins, epochs2, iterations=30)

# Testing

import cProfile
cProfile.runctx('fitModel(train, multi)', globals(), locals(), sort='cumtime')

class Complex:
          def __init__(self, realpart, imagpart):
                    self.r = realpart
                    self.i = imagpart
          def setRealPart(self, realpart):
                    self.r = realpart