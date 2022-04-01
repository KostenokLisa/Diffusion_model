import os

import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader

from src.data import get_dataset_by_name
from src.models import DDGM
from src.train import training, evaluation
from src.utils import samples_real, samples_diffusion, samples_generated, plot_curve, parse_args, get_config

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)

    data_params = config["data_params"]
    train_params = config["train_params"]
    opt_params = config["opt_params"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = tt.Lambda(lambda x: 2. * (x / 17.) - 1.)

    train_data = get_dataset_by_name(name="MNIST_sklearn", mode='train', transforms=transforms)
    val_data = get_dataset_by_name(name="MNIST_sklearn", mode='val', transforms=transforms)
    test_data = get_dataset_by_name(name="MNIST_sklearn", mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=data_params["train_batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=data_params["val_batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=data_params["test_batch_size"], shuffle=False)

    D = data_params["input_dimension"]  # input dimension
    M = train_params["hidden_units"]  # the number of neurons in scale (s) and translation (t) nets
    T = train_params["T"]
    beta = train_params["beta"]
    lr = float(opt_params["lr"])  # learning rate
    num_epochs = train_params["num_epochs"]  # max. number of epochs
    max_patience = train_params[
        "early_stopping"]  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    name = 'ddmg' + '_' + str(T) + '_' + str(beta)
    result_dir = os.path.join('results', name)
    if not (os.path.exists(result_dir)):
        os.mkdir(result_dir)

    p_dnns = [nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * D)).to(DEVICE) for _ in range(T - 1)]

    decoder_net = nn.Sequential(nn.Linear(D, M * 2), nn.LeakyReLU(),
                                nn.Linear(M * 2, M * 2), nn.LeakyReLU(),
                                nn.Linear(M * 2, M * 2), nn.LeakyReLU(),
                                nn.Linear(M * 2, D), nn.Tanh()).to(DEVICE)

    # Eventually, we initialize the full model
    model = DDGM(p_dnns, decoder_net, beta=beta, T=T, D=D).to(DEVICE)

    # OPTIMIZER
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Training procedure
    nll_val = training(name=os.path.join(result_dir, name), max_patience=max_patience, num_epochs=num_epochs,
                       model=model,
                       optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader, device=DEVICE)

    test_loss = evaluation(name=os.path.join(result_dir, name), test_loader=test_loader, device=DEVICE)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    samples_real(os.path.join(result_dir, name), test_loader)

    plot_curve(os.path.join(result_dir, name), nll_val)

    samples_generated(os.path.join(result_dir, name), test_loader, extra_name='FINAL', device=DEVICE)
    samples_diffusion(os.path.join(result_dir, name), test_loader, extra_name='DIFFUSION', device=DEVICE)
