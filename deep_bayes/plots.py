from __future__ import division
from Matrix import *
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from data_helper_functions import QuantiletoWeights

def PlotHistogram(bins, ybins, weights_y, weights_predict, k, wd, a, t, weights_3, num=3):
          plt.hist(ybins[0:bins], ybins, weights=weights_y, label=t[0], color='blue', histtype='step', linewidth=1.5)
          plt.hist(ybins[0:bins], ybins, weights=weights_predict, label=t[1], color='orangered', histtype='step', linewidth=1.5)
          if num == 3:
                    plt.hist(ybins[0:bins], ybins, weights=weights_3, label=t[2], color='darkturquoise', histtype='step', linewidth=1.5)
          plt.xlabel('Recoil True p_T (GeV)')
          plt.ylabel('Events')
          plt.legend(loc='upper right', fontsize='x-small')
          title = a + "_Iteration_" + str(k) + ".png"
          plt.savefig(wd + title)
          plt.close()

def PlotQuantile(prediction, y_weights, ybins, k, plot_directory, t, newbins, bins, plot_title, n_train, weights_3=[], num=2):
          weights_predict = [sum(Column(prediction, i)) for i in range(bins)]
          ndata = len(prediction)
          weights_predict = QuantiletoWeights(weights_predict, ybins, newbins, ndata)
          y_weights = QuantiletoWeights(y_weights, ybins, newbins, n_train)
          if num ==3:
                    weights_3 = QuantiletoWeights(weights_3, ybins, newbins, ndata)
          if n_train != ndata:
                    y_weights = [y*(ndata/n_train) for y in y_weights]
          PlotHistogram(bins, newbins, y_weights, weights_predict, k, plot_directory, plot_title, t, weights_3, num=num)
          plt.close('all')

def PlotData(prediction, y_weights, ybins, k, plot_directory, t, binstouse, bins, plot_title, n_train, weights_3=[], num=2):
          weights_predict= [sum(Column(prediction, i)) for i in range(bins)]
          ndata = len(prediction)
          a = binstouse[0]
          b = binstouse[1]
          bins = b-a
          if n_train != ndata:
                    y_weights = [y*(ndata/n_train) for y in y_weights]
          PlotHistogram(bins, ybins, y_weights, weights_predict[a:b],k, plot_directory, plot_title, t,  weights_3, num=num)
          plt.close('all')
