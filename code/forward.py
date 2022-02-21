import torch

DEVICE = torch.device("cuda")

# creating dictionary of constants used while estimating prior and posterior distributions
def estimate_a_b_parameters(betas):
    param_dict = {}

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    post_mean_coef_1 = betas * torch.sqrt(alphas_prod_p / (1 - alphas_prod))
    post_mean_coef_2 = (1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod)
    post_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
    post_log_variance = torch.log(torch.cat((post_variance[1].view(1, 1), post_variance[1:].view(-1, 1)), 0)).view(-1)

    param_dict["betas"] = betas
    param_dict["alphas"] = alphas
    param_dict["alphas_prod"] = alphas_prod
    param_dict["alphas_bar_sqrt"] = alphas_bar_sqrt
    param_dict["one_minus_alphas_bar_log"] = one_minus_alphas_bar_log
    param_dict["one_minus_alphas_bar_sqrt"] = one_minus_alphas_bar_sqrt
    param_dict["post_mean_coef_1"] = post_mean_coef_1
    param_dict["post_mean_coef_2"] = post_mean_coef_2
    param_dict["post_variance"] = post_variance
    param_dict["post_log_variance"] = post_log_variance

    return param_dict

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def q_sample(x_0, t, param_dict, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(param_dict["alphas_bar_sqrt"], t, x_0)
    alphas_1_m_t = extract(param_dict["one_minus_alphas_bar_sqrt"], t, x_0)
    return alphas_t * x_0 + alphas_1_m_t * noise

def q_posterior_mean_variance(x_0, x_t, t, param_dict):
    coef_1 = extract(param_dict["posterior_mean_coef_1"], t, x_0)
    coef_2 = extract(param_dict["posterior_mean_coef_2"], t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(param_dict["posterior_log_variance_clipped"], t, x_0)
    return mean, var










