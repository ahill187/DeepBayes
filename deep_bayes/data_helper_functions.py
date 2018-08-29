from __future__ import division
from keras.utils import to_categorical
from Matrix import *
import numpy as np

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
                    y_train = to_categorical(y_train, bins)
          return y_train

def Bins(bins, min, max, uneven=False, custom=False, weights=[]):
          if uneven==True:
                    binsplit = []
                    binsplit.append(min)
                    binwidth = np.abs(max-min-2)/(bins-2)
                    binsplit2 = [min + 1 + i*binwidth for i in range(bins-1)]
                    binsplit = binsplit + binsplit2 + [max]
          else:
                    binwidth = np.abs(max-min)/(bins)
                    if custom:
                              binsplit = VecScalar(weights, binwidth)
                    else:
                              binsplit = [min + i * binwidth for i in range(bins)]
                    binsplit[0] = min
                    binsplit.append(max)
          return binsplit

def BinError(y, prediction, bins):

          error =[]
          for i in range(bins):
                    z = sum(Column(y, i)) + 0.0001
                    h = sum(Column(prediction, i)) +0.0001
                    err = (z - h)/z
                    error.append(err)
          return error


def SampleWeights(class_weights, x, y):
          sample_weights = []
          for i in range(len(x)):
                    zclass = [index for index, item in enumerate(y[i]) if item == 1]
                    sample_weights.append(class_weights[zclass[0]])
          sample_weights = np.asarray(sample_weights)
          return (sample_weights)

def Damping(error, weights, bins, c=0.9):
          damping = []
          if c >=0 and c < 1:
                    print("Increasing damping by factor of %d", c)
          elif c > 1:
                    print("Decreasing damping by factor of %d", c)
          for i in range(0, bins):
                    damp = weights[i] ** (error[i]*c)
                    damping.append(damp)
          return damping

def ErrorAdjust(binerror, class_weights, c=1):
          ones = np.ones(len(class_weights))
          diff = VecSub(ones, class_weights)
          binerror = VecScalar(binerror, c)
          binerror = [min(b, 1.0) for b in binerror]
          new_weights = VecAdd(class_weights, VecMult(diff, binerror))
          return new_weights


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
                    if len(cuts[i]) == 0:
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
                                        if j==(len(cut)-1) and cut[j]+1<len(xbins):
                                                  diff = newbins[i +1] - xbins[cut[j]]
                                                  diff = diff / (xbins[cut[j]+1] - xbins[cut[j]])
                                                  weight = weight + diff*quantile_weights[cut[j]]
                                        elif j==(len(cut)-1) and cut[j]+1>=len(xbins):
                                                  pass
                              weights.append(weight)
          weights = VecScalar(weights, ndata/sum(weights))
          return weights

