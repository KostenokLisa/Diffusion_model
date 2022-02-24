import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
from forward import q_sample, q_posterior_mean_variance

N_STEPS = 100

# linear layer which works with intermediate steps of diffusion
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(2, 32, n_steps)
        self.lin2 = ConditionalLinear(32, 128, n_steps)
        self.lin3 = nn.Linear(128, 4)

    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)

# extracting mean and variance
def p_mean_variance(model, x, t):
    out = model(x, t)
    mean, log_var = torch.split(out, 2, dim=-1)
    return mean, log_var


def p_sample(model, x, t):
    mean, log_var = p_mean_variance(model, x, torch.tensor(t))
    noise = torch.randn_like(x)
    sample = mean + torch.exp(0.5 * log_var) * noise
    return (sample)


def p_sample_loop(model, shape):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(N_STEPS)):
        cur_x = p_sample(model, cur_x, i)
        x_seq.append(cur_x)
    return x_seq


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl


def entropy(val):
    return (0.5 * (1 + np.log(2. * np.pi))) + 0.5 * np.log(val)


def compute_loss(true_mean, true_var, model_mean, model_var, param_dict):
    # computing KL divergence between model prediction and true posterior
    KL = normal_kl(true_mean, true_var, model_mean, model_var).float()
    # computing conditional entropies x_t|x_0
    H_start = entropy(param_dict["betas"][0].float()).float()
    beta_full_trajectory = 1. - torch.exp(torch.sum(torch.log(param_dict["alphas"]))).float()
    H_end = entropy(beta_full_trajectory.float()).float()
    H_prior = entropy(torch.tensor([1.])).float()
    negL_bound = KL * N_STEPS + H_start - H_end + H_prior
    # applied under the assumption of isotropic Gaussian distribution of the data
    negL_gauss = entropy(torch.tensor([1.])).float()
    negL_diff = negL_bound - negL_gauss
    L_diff_bits = negL_diff / np.log(2.)
    L_diff_bits_avg = L_diff_bits.mean()
    return L_diff_bits_avg


def loss_likelihood_bound(model, x_0, param_dict):
    # selecting random timestep, calculating true posterior, making prediction and calculating loss value
    batch_size = x_0.shape[0]
    t = torch.randint(0, N_STEPS, size=(batch_size // 2 + 1,))
    t = torch.cat([t, N_STEPS - t - 1], dim=0)[:batch_size].long()
    x_t = q_sample(x_0, t, param_dict)
    true_mean, true_var = q_posterior_mean_variance(x_0, x_t, t, param_dict)
    model_mean, model_var = p_mean_variance(model, x_t, t)
    loss = compute_loss(true_mean, true_var, model_mean, model_var, param_dict)
    return loss



