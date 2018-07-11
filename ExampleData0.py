#----------------------------------------------------------------------
# Author: Ainsleigh Hill
# Contact: ainsleighhill@gmail.com
# Date created: 180705
# Date modified: 180705
# Description:
# Creating small data samples for testing the package.
#------------------------------
from __future__ import division
import keras
import os
from Matrix import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from matplotlib.backends.backend_pdf import PdfPages

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

def Bins(bins, min, max):
    binwidth = np.abs(max - min) / (bins)
    binsplit = [min + i * binwidth for i in range(bins)]
    return binsplit

class Data:
    def __init__(self, length, x, y, xbins, ybins, x_weights, y_weights):
        self.length = length
        self.y = y
        self.x = x
        self.xbins = xbins
        self.ybins = ybins
        self.x_weights = x_weights
        self.y_weights = y_weights

# Training Data -----------------------------------------------------------

def CreateExampleData(ndata=1000, ngauss=1000, bins=50, sd_smear=0.1, title="NA", sd=0.5, plots="/Plots_0/"):

    plot_directory = os.getcwd() + "/Plots_2/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    mean = 0

    y = np.random.normal(mean, sd, ndata)
    x = np.random.normal(mean, sd_smear, ndata)
    x = x + y

    xbins = Bins(bins, -4, 4)
    ybins = Bins(bins, -4, 4)

    y = BinClass(y, bins, ybins)
    x_weights = BinClass(x, bins, xbins)

    weights_x = [sum(Column(x_weights, i)) for i in range(bins)]
    weights_y = [sum(Column(y, i)) for i in range(bins)]

    plt.hist(xbins, bins, weights=weights_x, label="Detector Data", alpha=0.5, edgecolor='grey')
    plt.hist(ybins, bins, weights=weights_y, label="True Data", alpha=0.5, edgecolor='grey')
    plt.legend(loc='upper right')
    a = plot_directory + title
    plt.savefig(a)
    plt.close()


    data = Data(ngauss, x, y, xbins, ybins, weights_x, weights_y)
    return data
