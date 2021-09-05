import numpy as np
from numpy.random import multivariate_normal
from numpy.random import binomial
import matplotlib.pyplot as plt


# dataset consists of M vectors sampled from N-dimensional gmm distribution
def create_dataset(M, mean1, mean2, var1, var2, p):
    idx1 = binomial(1, p, M)
    idx2 = np.ones(M) - idx1

    data1 = multivariate_normal(mean1, np.diag(np.sqrt(var1)), M)
    data2 = multivariate_normal(mean2, np.diag(np.sqrt(var2)), M)

    gmm_data = np.diag(idx1) @ data1 + np.diag(idx2) @ data2
    return gmm_data


# 1-D or 2-D distribution visualization
def draw_distribution(gmm_data):
    dim = gmm_data.shape
    N = len(dim) - 1
    if N == 1:
        step = 100
        fig, ax = plt.subplots(figsize=(30, 20))
        ax.hist(gmm_data, step)
    else:
        plt.scatter(gmm_data.T[0], gmm_data.T[1])
        plt.title("2dim distribution")

    plt.show()
