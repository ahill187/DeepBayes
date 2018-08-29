#------------------------------
# Author: Ainsleigh Hill
# Date created: 180606
# Date modified: 180606
# Description:
#------------------------------
from __future__ import division
from __future__ import print_function
import numpy as np
from builtins import input
from ast import literal_eval as boolean
from Matrix import *
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import sys
import datetime
import time
from argparse import ArgumentParser
from get_data import getTrees, getModel, getTestingData, getTrainingData, compileModel, fitModel, reweightData, fitModelFast, fitModelFastReweight
from data_helper_functions import BinError, SampleWeights, Bins, Damping, ErrorAdjust
from plots import PlotData, PlotQuantile
from classes import MultiClassifier
from settings import MODELLIST, DEFAULTLIST, ADJUST, BIN, TESTLIST

# Input from the Console -----------

default = input('Default (Yes or No)? ')
if default == "No" or default == 'N':
          folder = input("Plot Directory: ")
          epochs = int(input("Epochs: "))
          epochs2 = int(input("Epochs Bayes: "))
          n_train = int(input("Number of Training Points: "))
          fast = boolean(input("Use fit instead of fit.generator (True or False)"))
else:
          folder = DEFAULTLIST['folder']
          epochs = DEFAULTLIST['epochs']
          epochs2 = DEFAULTLIST['epochs2']
          fast = DEFAULTLIST['fast']
          n_train = DEFAULTLIST['n_train']
ndata = DEFAULTLIST['ndata']
bins = DEFAULTLIST['bins']
custom = DEFAULTLIST['custom']
test_weights = DEFAULTLIST['test_weights']
damping_constant = DEFAULTLIST['damping_constant']
iterations=DEFAULTLIST['iterations']
adjust = ADJUST
bin_weights = BIN

# Getting input from bash console

parser = ArgumentParser('Run the training')
parser.add_argument('inputDataCollection', default="")
parser.add_argument('outputDir', default="")
parser.add_argument('plots', default="")
parser.add_argument('--modelMethod', help='Method to be used to instantiate model in derived training class',
                              metavar='OPT', default=None)
parser.add_argument("--gpu", help="select specific GPU", type=int, metavar="OPT", default=-1)
parser.add_argument("--gpufraction", help="select memory fraction for GPU", type=float, metavar="OPT",
                              default=-1)
args = parser.parse_args()

plot_directory = args.plots+"/"+folder

if not os.path.exists(plot_directory):
          os.makedirs(plot_directory)
print("Putting plots in: %s" % plot_directory)
print(fast)
time.sleep(2)

# Training Data ---------------------------------------------------------------------
multi = MultiClassifier()
train = getTrees(bins, args=args)
model_num = int(train.keras_model_method)

MODELLIST[model_num]['nepochs'] = epochs
train = getModel(train, MODELLIST, custom, bin_weights)
train = compileModel(train, multi, MODELLIST)
if fast:
          MODELLIST[model_num]['nepochs'] = 1
          model = fitModel(train, MODELLIST)
          data = getTrainingData(train, n_train, custom, bin_weights)
          MODELLIST[model_num]['nepochs'] = epochs
          model = fitModelFast(train, data, MODELLIST)
else:
          model = fitModel(train, MODELLIST, custom, bin_weights, ndata=n_train)

# Create README File--------------------------------------------------------------------------

f = open(plot_directory+'README.txt', 'w+')
text = ["Date: ", "Epochs Training: ", "Epochs Bayes: ", "Regularization: ", "Optimizer: ", "Layers: ", "Learning Rate: ", "Model: ", "Bins: ", "Test Weights: "]
date = datetime.datetime.now()
layers = MODELLIST[model_num]['arch']
lr = MODELLIST[model_num]['learningrate']
reg = MODELLIST[model_num]['reg']
info = [str(date.year)+"_"+str(date.month)+"_"+str(date.day), str(epochs), str(epochs2), str(reg), "Nadam", str(layers), str(lr), str(model_num), str(bins), str(test_weights)]
for i in range(len(text)):
          f.write(text[i]+info[i]+"\n")
f.close()

# Keras Model -------------------------------------------------------------------------------------

def BayesIteration(model, n_train, ndata, bins, train, fast, adjust, test_weights_list, data="", iterations=30, damping_constant=1):

          if not fast:
              data = getTrainingData(train, ndata)
          y_weights_class= []
          for y in data.y_weights:
                    if y == 0:
                              y_weights_class.append(0.0000000001)
                    else:
                              y_weights_class.append(y)
          if n_train != ndata:
                    y_weights_class=[y*(ndata/n_train) for y in y_weights_class]
          reweight = reweightData(y_weights_class, ndata, test_weights_list[test_weights])
          test = getTestingData(train, reweight, ndata)
          newbins = Bins(bins, train.train_data.ybins[0], train.train_data.ybins[bins], uneven=False)
          k = 0
          score = []

          while True:

                    # Plot Training Data
                    prediction = model.predict(np.asarray(data.x))
                    t = ["MC True Distribution", "Predicted from Keras"]
                    PlotData(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Training", n_train)
                    PlotQuantile(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, newbins, bins, "Training_Hist", n_train)

                    # Plot Testing Data
                    prediction = model.predict(np.asarray(test.x))
                    t = ["Measured True Distribution", "Predicted from Keras"]
                    PlotData(prediction, test.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Testing", ndata)
                    PlotQuantile(prediction, test.y_weights, train.train_data.ybins, k, plot_directory, t, newbins, bins, "Testing_Hist", ndata)

                    # Plot Combined Data
                    t = ["MC True Distribution", "Predicted from Keras", "Measured True Distribution"]
                    PlotData(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, [0, bins], bins, "Combined_Binned", n_train, weights_3=test.y_weights, num=3)
                    PlotQuantile(prediction, data.y_weights, train.train_data.ybins, k, plot_directory, t, newbins, bins, "Combined_Hist", n_train, weights_3=test.y_weights, num=3)

                    # Get Score
                    score.append(model.evaluate(np.asarray(data.x), np.asarray(data.y), batch_size=128))
                    print("Loss = {0}, Accuracy = {1}".format(score[k][1], score[k][2]))

                    # Calculate Sample Weights
                    weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
                    binerror = BinError(data.y, model.predict(np.asarray(data.x)), bins)
                    # Bin Error
                    if adjust['binerror']:
                              error = VecMult(weights_predict, binerror)
                              weights_predict = VecAdd(weights_predict, error)
                              class_weights = np.asarray([weights_predict[i]/y_weights_class[i] for i in range(bins)])
                    elif adjust['binerror2']:
                              class_weights = np.asarray([weights_predict[i] / y_weights_class[i] for i in range(bins)])
                              class_weights = ErrorAdjust(binerror, class_weights, c=damping_constant)
                    # Damping
                    elif adjust['damping']:
                              class_weights = np.asarray([weights_predict[i]/y_weights_class[i] for i in range(bins)])
                              class_weights = Damping(binerror, class_weights, bins, c=damping_constant)
                    # Reweight data
                    elif adjust['retrain']:
                              class_weights=weights_predict
                              data = getTestingData(train, weights_predict, ndata)
                    else:
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
                              if fast:
                                        #fitModelFastReweight(data, model, epochs2, multi.batch, sample_weights)
                                        model.fit(x=np.asarray(data.x), y=np.asarray(data.y), epochs=epochs2,
                                                  batch_size=multi.batch, sample_weight=sample_weights)
                                        # model.fit(x=np.asarray(data.x), y=np.asarray(data.y), epochs=epochs2,
                                        #           batch_size=multi.batch)
                              else:
                                        model.fit_generator(train.train_data.generator(),
                                                             steps_per_epoch=train.train_data.getNBatchesPerEpoch(),
                                                             epochs=epochs2,
                                                             #callbacks=train.callbacks.callbacks,
                                                             #validation_data=train.val_data.generator(),
                                                             #validation_steps=train.val_data.getNBatchesPerEpoch(),
                                                             max_q_size=5, class_weight=class_dict)

                    k = k+1

          return model

multi = MultiClassifier()
multi.epochs = epochs
if fast:
          model = BayesIteration(model, n_train, ndata, bins, train, fast, adjust, TESTLIST, data, iterations, damping_constant)
else:
          model = BayesIteration(model, n_train, ndata, bins, train, fast, adjust, TESTLIST, iterations=iterations, damping_constant=damping_constant)

# Testing

#import cProfile
#cProfile.runctx('fitModel(train, multi)', globals(), locals(), sort='cumtime')
