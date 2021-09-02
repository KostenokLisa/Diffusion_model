from numpy.random import normal
from math import sqrt
import matplotlib.pyplot as plt


# dataset consists of N floats sampled from gmm distribution
def create_dataset(N, mean1, mean2, var1, var2, p):

    data1 = normal(mean1, sqrt(var1), size=N)
    data2 = normal(mean2, sqrt(var2), size=N)

    gmm_data = p * data1 + (1 - p) * data2
    return gmm_data


def draw_distribution(gmm_data, N):
    step = 100

    fig, ax = plt.subplots(figsize=(30, 20))
    ax.hist(gmm_data, int(N / step))
    plt.show()
