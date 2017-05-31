
# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
# some setting for this notebook to actually show the graphs inline, you probably won't need this
# %matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
print X.shape  # 150 samples with 2 dimensions

# plt.scatter(X[:,0], X[:,1])
# plt.show()


# generate the linkage matrix
Z = linkage(X, 'ward')