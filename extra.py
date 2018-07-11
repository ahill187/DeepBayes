a = list(range(k))
a.append(20)
t = ["Accuracy (Categorical Cross Entropy)", "Mean Absolute Error","Mean Squared Error" ]
for i in range(1,len(score)):
    b = Column(score, i)
    plt.scatter(a,b)
    plt.xlabel("Iteration")
    plt.ylabel(t[i-1])
    plt.savefig(fname=plot_directory+t[i-1]+".pdf")
    plt.close('all')

plt.hist(test.ybins, bins, weights=np.asarray(train.y_weights)*weights, label="Reweighted", alpha=0.3, edgecolor='grey')
plt.hist(test.ybins, bins, weights=weights_predict, label="Predicted Test Data", alpha=0.3, edgecolor='grey')
plt.legend(loc='upper right')
title = "Reweight.png"
plt.savefig(plot_directory + title)
plt.close('all')