import numpy as np
from numpy.random import multivariate_normal
from math import sqrt


# for N-dimensional input x sample and save all latent variables,
# T - number of steps in forward process, betas - hyperparameters
def sample_latent_var(betas, x):
    samples = [x]
    T = len(betas)
    N = len(x)
    x = np.array(x)

    for i in range(T):
        mean = sqrt(1 - betas[i]) * x
        scale = sqrt(betas[i]) * np.eye(N)
        x = multivariate_normal(mean, scale, size=1)[0]
        samples.append(x)

    return np.array(samples)
