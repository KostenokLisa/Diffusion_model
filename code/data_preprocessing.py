import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from matplotlib import colors, pyplot as plt
from numpy.random import normal, multivariate_normal
from numpy.random import binomial

DATA_MODES = ['train', 'val', 'test']
# define the size of image
RESCALE_SIZE = 224
DEVICE = torch.device("cuda")


def get_files(path_to_train, path_to_test):
    TRAIN_DIR = Path(path_to_train)
    TEST_DIR = Path(path_to_test)
    train_files = sorted(list(TRAIN_DIR.rglob('*.jpeg')))
    test_files = sorted(list(TEST_DIR.rglob('*.jpeg')))
    return train_files, test_files


# input: dataset consists of pictures kept in labelled folders
class CustomDataset(Dataset):
    def __init__(self, files, mode):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # transforming images to tensors
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # adding augmentations to train pictures if necessary
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
        ])

        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        if self.mode == 'test':
            x = transform_test(x)
            return x
        else:
            if self.mode == 'val':
                x = transform_test(x)
            else:
                x = transform_train(x)
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return np.array(image)


def imshow(inp, title=None, plt_ax=plt, default=False):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_samples_imgs(dataset):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 8), sharey=True, sharex=True)
    for fig_x in ax.flatten():
        sample_id = int(np.random.uniform(0, 300))
        img, label = dataset[sample_id]
        img_label = " ".join(map(lambda x: x.capitalize(),
                                 dataset.label_encoder.inverse_transform([label])[0].split('_')))
        imshow(img.data.cpu(), title=img_label, plt_ax=fig_x)

# dataset consists of vectors sampled from gmm distribution
def make_gmm_dataset(dataset_size, mean1=None, mean2=None, var1=None, var2=None, p=None):
    # sampling from 2-dim mixed Gaussian distribution, p - hyperparameter of distribution
    if mean1 is None:
        mean1 = [5.0, 5.0]
    if mean2 is None:
        mean2 = [-5.0, -5.0]
    if var1 is None:
        var1 = [1.0, 1.0]
    if var2 is None:
        var2 = [1.0, 1.0]
    if p is None:
        p = 0.3

    idx1 = binomial(1, p, dataset_size)
    idx2 = np.ones(dataset_size) - idx1

    x1 = multivariate_normal(mean1, np.diag(np.sqrt(var1)), dataset_size)
    x2 = multivariate_normal(mean2, np.diag(np.sqrt(var2)), dataset_size)

    gmm_data = np.diag(idx1) @ x1 + np.diag(idx2) @ x2
    return torch.from_numpy(gmm_data)

# dataset consists of vectors sampled from funnel distribution
def make_funnel_dataset(dataset_size, mean=None, scale=None):
    if mean is None:
        mean = 0.0
    if scale is None:
        scale = 3.0
    x1 = np.ones(dataset_size)
    x2 = normal(mean, scale, dataset_size)

    for idx in range(len(x1)):
        x1[idx] = normal(0, np.exp(x2[idx] / 2), 1)

    x1 = x1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    funnel_data = np.concatenate((x1, x2), axis=1)
    return torch.from_numpy(funnel_data)

# dataset consists of vectors sampled from banana-shaped distribution
def make_banana_shaped_dataset(dataset_size, mean=None, var=None):
    if mean is None:
        mean = [0, 0]
    if var is None:
        var = [[1, 0.95], [0.95, 1]]
    x = multivariate_normal(mean, np.sqrt(var), dataset_size)
    y1 = x[:, 0]
    y2 = x[:, 1] - x[:, 0]**2 - np.ones(dataset_size)

    y1 = y1[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    return torch.from_numpy(np.concatenate((y1, y2), axis=1))


def show_samples_distr(dataset):
    plt.figure(figsize=(8, 6))
    plt.scatter(*dataset.T.cpu(), alpha=0.5, color='red', edgecolor='white', s=40)
    plt.show()



