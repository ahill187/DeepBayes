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
from keras import backend as K

def PlotHistogram(bins, binsx, xbins, ybins, weights_x, weights_y, k, wd, a, t, weights_predict=[],num=3):
        plt.hist(xbins[0:binsx], xbins, weights=weights_x, label=t[0], alpha=0.5, edgecolor='grey', color='cyan')
        plt.hist(ybins[0:bins], ybins, weights=weights_y, label=t[1], alpha=0.5, edgecolor='grey', color='grey')
        if num==3:
                plt.hist(ybins[0:bins], ybins, weights=weights_predict, label=t[2], alpha=0.5, edgecolor='grey', color='yellow')
        else:
                pass
        plt.legend(loc='upper right', fontsize='x-small')
        title = a + "_Iteration_" + str(k) + ".png"
        plt.savefig(wd + title)
        plt.close()

def PlotQuantile(prediction, x_weights, y_weights, xbins, ybins, k, plot_directory, t, newbins, bins, plot_title):
        weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
        ndata = len(prediction)
        weights_predict = QuantiletoWeights(weights_predict, ybins, newbins, ndata)
        x_weights = QuantiletoWeights(x_weights, xbins, newbins, ndata)
        y_weights = QuantiletoWeights(y_weights, ybins, newbins, ndata)
        PlotHistogram(bins, bins, newbins, newbins, x_weights, y_weights, k, plot_directory, plot_title, t, weights_predict)
        plt.close('all')

def PlotData(prediction, x_weights, y_weights, xbins, ybins, k, plot_directory, t, binstouse, bins, plot_title):
        weights_predict= [sum(Column(prediction, i)) for i in range(bins)]
        a = binstouse[0]
        b = binstouse[1]
        bins = b-a
        PlotHistogram(bins, bins, xbins[a:b+1], ybins[a:b+1], x_weights[a:b], y_weights[a:b], k, plot_directory, plot_title, t, weights_predict[a:b])
        plt.close('all')