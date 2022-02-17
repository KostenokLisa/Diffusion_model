import torch

DEVICE = torch.device("cuda")


def naive_forward_process(data_loader, n_steps, betas):
    # gradually adding gaussian noise to tensors in batch with fixed betas parameter
    res_seq = []
    for inputs, labels in data_loader:
        inputs = inputs.to(DEVICE)
        #labels = labels.to(DEVICE)  labels can be used for conditional generation
        batch_start = inputs
        batch_seq = [batch_start]
        for n in range(n_steps):
            batch_seq.append((torch.sqrt(1 - betas[n]) * batch_seq[-1]) + (betas[n] * torch.rand_like(batch_start)))
        res_seq.append(batch_seq)
    return res_seq


def make_beta_schedule(schedule, n_timesteps, start, end):
    # variation of betas parameter depending on schedule mode
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        limit = 6  # set limits for sigmoid argument
        betas = torch.linspace(-limit, limit, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


def get_alphas(betas):
    # estimation of alpha parameter depending on betas variation
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    return torch.sqrt(alphas_prod), torch.sqrt(1 - alphas_prod)


def extract(input, t, x_batch):
    # getting alpha parameters in required number of dimensions
    shape = x_batch.shape
    N = len(shape)
    output = torch.gather(input, 0, t.to(DEVICE))
    new_shape = [t.shape[0]] + [1] * (N - 1)
    return output.reshape(*new_shape)


def q_sample(x_batch, t, betas, noise):
    # more efficient forward process for sampling at a fixed timestep t
    if noise is None:
        noise = torch.randn_like(x_batch).to(DEVICE)
    alphas_bar_sqrt, one_minus_alphas_bar_sqrt = get_alphas(betas)
    alphas_t = extract(alphas_bar_sqrt, t, x_batch)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_batch)

    alphas_1_m_t = alphas_1_m_t.to(DEVICE)
    alphas_t = alphas_t.to(DEVICE)
    return alphas_t * x_batch + alphas_1_m_t * noise


def calc_posterior(betas, alphas, alphas_prod_p, alphas_prod):
    # estimating posterior distribution parameters based on beta and alpha parameters
    post_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
    post_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
    post_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
    post_log_variance = torch.log(torch.cat((post_variance[1].view(1, 1), post_variance[1:].view(-1, 1)), 0))
    return post_mean_coef_1, post_mean_coef_2, post_log_variance.view(-1)


def q_posterior_mean_variance(betas, alphas, alphas_prod_p, alphas_prod, t, x_0, x_t):
    # estimating posterior distribution parameters at a fixed timestep t
    post_params = calc_posterior(betas, alphas, alphas_prod_p, alphas_prod)
    post_mean_coef_1 = post_params[0]
    post_mean_coef_2 = post_params[1]
    post_log_variance = post_params[2]

    coef_1 = extract(post_mean_coef_1, t, x_0)
    coef_2 = extract(post_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(post_log_variance, t, x_0)
    return mean, var