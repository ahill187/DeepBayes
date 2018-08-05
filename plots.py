def PlotHistogram(bins, binsx, xbins, ybins, weights_x, weights_y, k, wd, a, t, weights_predict=[],num=3):
    plt.hist(xbins, xbins, weights=weights_x, label=t[0], alpha=0.5, edgecolor='grey', color='cyan')
    plt.hist(ybins, ybins, weights=weights_y, label=t[1], alpha=0.5, edgecolor='grey', color='grey')
    if num==3:
        plt.hist(ybins, ybins, weights=weights_predict, label=t[2], alpha=0.5, edgecolor='grey', color='yellow')
    else:
        pass
    plt.legend(loc='upper right', fontsize='x-small')
    title = a + "_Iteration_" + str(k) + ".png"
    plt.savefig(wd + title)
    plt.close()