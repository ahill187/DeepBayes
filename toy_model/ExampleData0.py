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
#matplotlib.use('Agg')
matplotlib.use('TKAgg')
from math import *
import matplotlib.pyplot as plt

def BinClass(x_train, bins, xbins, ones=True):
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
          if ones==True:
                    y_train = keras.utils.to_categorical(y_train, bins)
          return y_train

def Bins(bins, min, max, uneven=False):
          if uneven==True:
                    binsplit = []
                    binsplit.append(min)
                    binwidth = np.abs(max-min-2)/(bins-2)
                    binsplit2 = [min + 1 + i*binwidth for i in range(bins-1)]
                    binsplit = binsplit + binsplit2 + [max]
          else:
                    binwidth = np.abs(max - min) / (bins)
                    binsplit = [min + i * binwidth for i in range(bins)]
                    binsplit[0] = min
                    binsplit.append(max)
          return binsplit

def QuantileBins(bins, min, max, x):
          quant = 100 / bins
          binsplit = [min]
          for i in range(1, bins):
                    percentile = np.percentile(x, i*quant)
                    binsplit.append(percentile)
          binsplit.append(max)
          return binsplit

def ProbABNorm(a, b, mean=0, sd=0.1):
          phi_a = 0.5*(1 + erf((a - mean)/(sd*sqrt(2))))
          phi_b = 0.5 * (1 + erf((b - mean) / (sd * sqrt(2))))
          prob = phi_b - phi_a
          return prob

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

def CreateExampleData(ndata=1000, ngauss=1000, bins=50, binsx=50, sd_smear=0.1, title="NA", sd=0.5, plot_directory="Plots", quantiles=0, xbins=[], ybins=[]):

          if not os.path.exists(plot_directory):
                    os.makedirs(plot_directory)

          mean = 0

          y = np.random.normal(mean, sd, ndata)
          x = np.random.normal(mean, sd_smear, ndata)
          x = x + y

          if quantiles==0:
                    xbins = Bins(binsx, -2, 2)
                    ybins = Bins(bins, -2, 2)
          elif quantiles==1:
                    ybins = QuantileBins(bins, -50, 50, y)
                    xbins = ybins
          elif quantiles==2:
                    pass

          y1 = BinClass(y, bins, ybins, ones=False)
          y1 = np.asarray(y1)
          x_weights = BinClass(x, binsx, xbins)
          y2 = BinClass(y, bins, ybins)

          weights_x = [sum(Column(x_weights, i)) for i in range(binsx)]
          weights_y = [sum(Column(y2, i)) for i in range(bins)]

          plt.hist(xbins[0:binsx], xbins, weights=weights_x, label="Detector Data", alpha=0.5, edgecolor='grey')
          plt.hist(ybins[0:bins], ybins, weights=weights_y, label="True Data", alpha=0.5, edgecolor='grey')
          plt.legend(loc='upper right')
          a = plot_directory + title
          plt.savefig(a)
          plt.close('all')


          data = Data(ngauss, x, y2, xbins, ybins, weights_x, weights_y)
          return data

def SampleWeights(class_weights, train):
           sample_weights = []
           for i in range(len(train.x)):
                    zclass = [index for index, item in enumerate(train.y[i]) if item==1]
                    sample_weights.append(class_weights[zclass[0]])
           sample_weights = np.asarray(sample_weights)
           return(sample_weights)

def Num(x, bins):
          n = []
          for i in range(bins):
                    num = len([index for index, item in enumerate(x) if item == i])
                    n.append(num)
          return n

def QuantiletoWeights(quantile_weights, xbins, newbins, ndata):

          cuts = []
          for i in range(len(newbins)-1):
                    lower = newbins[i]
                    upper = newbins[i+1]
                    upper2 = [index for index, item in enumerate(xbins) if item <=upper]
                    lower2 = [index for index, item in enumerate(xbins) if item >=lower]
                    intersect = list(set(lower2) & set(upper2))
                    intersect.sort()
                    if len(intersect)==0:
                              cuts.append([])
                    else:
                              cuts.append(intersect)

          weights = []
          for i in range(len(cuts)):
                    if len(cuts[i])==0:
                              upper = newbins[i+1]
                              lower = newbins[i]
                              diff = upper - lower
                              a = [index for index, item in enumerate(xbins) if item>=upper]
                              a = min(a)
                              diff = diff / (xbins[a] - xbins[a-1])
                              weights.append(diff*quantile_weights[a-1])
                    else:
                              cut = cuts[i]
                              weight = 0
                              for j in range(len(cut)):
                                        if j==0:
                                                  diff = xbins[cut[j]] - newbins[i]
                                                  diff = diff / (xbins[cut[j]]- xbins[cut[j]-1])
                                                  weight = weight + diff*quantile_weights[cut[j]-1]

                                        elif j<len(cut):
                                                  weight = weight + quantile_weights[cut[j]-1]
                                        if j==(len(cut)-1):
                                                  diff = newbins[i +1] - xbins[cut[j]]
                                                  diff = diff / (xbins[cut[j]+1] - xbins[cut[j]])
                                                  weight = weight + diff*quantile_weights[cut[j]]
                              weights.append(weight)
          weights = VecScalar(weights, ndata/sum(weights))
          return weights

def PredtoOneHot(prediction):

          onehotarray = []
          bins = len(prediction[0])
          for i in range(len(prediction)):
                    bin = [index for index, item in enumerate(prediction[i]) if item == max(prediction[i])][0]
                    onehot = []
                    for j in range(bins):
                              if j == bin:
                                        onehot.append(1)
                              else:
                                        onehot.append(0)
                    np.asarray(onehot)
                    onehotarray.append(onehot)
          return np.asarray(onehotarray)

def Efficiency(train, bins):

          x = BinClass(train.x, bins, train.xbins, ones=False)
          y = train.y
          efficiency = []
          for i in range(bins):
                    a = [index for index, item in enumerate(Column(y, i)) if item == 1]
                    x_class = np.zeros(bins)
                    for j in range(len(a)):
                              x_val = x[a[j]]
                              x_class[x_val] = x_class[x_val] + 1
                    num_z = len(a)
                    x_class = VecScalar(x_class, 1/num_z)
                    efficiency.append(sum(x_class))

def BinError(train, prediction, bins):

          error =[]
          for i in range(bins):
                    z = sum(Column(train.y, i))
                    h = sum(Column(prediction, i))
                    err = (z - h)/z
                    error.append(err)
          return error


