import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/sklearn_mnist_config.yml")
    return parser.parse_args()


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name='', device='cpu'):
    with torch.no_grad():
        # GENERATIONS-------
        model_best = torch.load(name + '.model', map_location=device)
        model_best.eval()

        num_x = 4
        num_y = 4
        x = model_best.sample(batch_size=num_x * num_y, device=device)
        x = x.cpu().detach().numpy()

        fig, ax = plt.subplots(num_x, num_y)
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(x[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.axis('off')

        plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
        plt.close()


def samples_diffusion(name, data_loader, extra_name='', device='cpu'):
    with torch.no_grad():
        x = next(iter(data_loader))
        x = x.to(device)

        # GENERATIONS-------
        model_best = torch.load(name + '.model', map_location=device)
        model_best.eval()

        num_x = 4
        num_y = 4
        z = model_best.sample_diffusion(x)
        z = z.cpu().detach().numpy()

        fig, ax = plt.subplots(num_x, num_y)
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(z[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.axis('off')

        plt.savefig(name + '_generated_diffusion' + extra_name + '.pdf', bbox_inches='tight')
        plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()
