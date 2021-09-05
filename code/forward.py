from numpy.random import normal
from math import sqrt


# for x sample and save all latent variables, T - number of steps in forward process, betas - hyperparameters
def sample_latent_var(betas, x):
    samples = [x]
    T = len(betas)

    for i in range(T):
        mean = sqrt(1 - betas[i]) * x
        scale = sqrt(betas[i])
        x = normal(mean, scale)
        samples.append(x)

    return samples
