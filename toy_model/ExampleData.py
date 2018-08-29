#----------------------------------------------------------------------
# Author: Ainsleigh Hill
# Contact: ainsleighhill@gmail.com
# Date created: 180705
# Date modified: 180705
# Description:
# Creating small data samples for testing the package.
#------------------------------

import keras
import os
from Matrix import *
import numpy as np
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
    binwidth = np.abs(max - min) / bins
    binsplit = [min + i * binwidth for i in range(bins)]
    return binsplit

# Training Data -----------------------------------------------------------

def CreateTrainingData(ndata=1000, ngauss=1000, bins=50, sd_smear=0.8):

    plot_directory = os.getcwd() + "/Plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    mean = 0
    sd_list = [round(float(np.random.uniform(0, 1)), 8) for i in range(ndata)]

    y_train = [np.random.normal(mean, std, ngauss) for std in sd_list]
    x_train = [np.random.normal(mean, sd_smear, ngauss) for i in range(ndata)]
    x_train = np.asarray([x_train[i] + y_train[i] for i in range(ndata)])

    xbins = Bins(bins, -2, 2)
    ybins = Bins(bins, -2, 2)
    y_train = [BinClass(y_train[i], bins, ybins) for i in range(ndata)]
    x_weights = [BinClass(x_train[i], bins, xbins) for i in range(ndata)]
    y_train = np.asarray([sum(y_train[i]) for i in range(ndata)])

    title = "ExampleTrainingData.pdf"
    pdf_pages = PdfPages(plot_directory+title)

    for m in range(10,20):
        # Create a figure instance (ie. a new page)
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)

        weights_x = [sum(Column(x_weights[m], i)) for i in range(bins)]
        weights_y = y_train[m]

        plt.hist(xbins, bins, weights=weights_x, label="Detector Data", alpha=0.5, edgecolor='grey')
        plt.hist(ybins, bins, weights=weights_y, label="True Data", alpha=0.5, edgecolor='grey')
        plt.legend(loc='upper right')

        pdf_pages.savefig(fig)
        plt.close()

    pdf_pages.close()

    return x_train, y_train


# Testing Data -----------------------------------------------------------------------------------

def CreateTestingData(ndata=1000, ngauss=1000, bins=50, sd_smear=0.8):

    sd = 0.4
    mean = 0
    xbins = Bins(bins, -2, 2)
    ybins = Bins(bins, -2, 2)
    y_test = np.random.normal(mean, sd, ndata)
    x_test = np.random.normal(mean, sd_smear, ndata)
    x_test = x_test + y_test
    y_test = BinClass(y_test, bins, ybins)
    x_test_weights = BinClass(x_test, bins, xbins)
    y_test = np.asarray([sum(Column(y_test,i)) for i in range(bins)])

    weights_x_test = [sum(Column(x_test_weights, i)) for i in range(bins)]
    weights_y_test = y_test

    plot_directory = os.getcwd() + "/Plots/"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    title = "ExampleTestingData.pdf"
    plt.hist(xbins, bins, weights=weights_x_test, label="Detector Data Test", alpha=0.5, edgecolor='grey')
    plt.hist(ybins, bins, weights=weights_y_test, label="True Data Test", alpha=0.5, edgecolor='grey')
    plt.legend(loc='upper right')
    plt.savefig(fname=plot_directory+title)
    plt.close()

    return x_test, y_test



