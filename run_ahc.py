import pandas

import matplotlib.pyplot as pyplot

dataset = pandas.read_csv("dataset_ahc.csv")

import scipy.cluster.hierarchy as shc
#numpy is numerical python
#scipy is scientific python

from sklearn.cluster import AgglomerativeClustering


dataset = pandas.read_csv("dataset_ahc.csv")


# print(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'])
pyplot.savefig("scatterplot_ahc.png")
pyplot.close()


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="ward"))
pyplot.savefig("dendrogram.png")
pyplot.close()

machine = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
results_ahc = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results_ahc)
pyplot.savefig("scatterplot_ahc_color.png")
pyplot.close()

