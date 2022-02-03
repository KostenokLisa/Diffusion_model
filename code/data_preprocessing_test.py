import data_preprocessing
import argparse
from data_preprocessing import CustomDataset, get_files, show_samples
from torch.utils.data import DataLoader
import torch
import os
from zipfile import ZipFile

parser = argparse.ArgumentParser(description="Dataset modes")
parser.add_argument("--dataset")  # name of the dataset
parser.add_argument("--path")  # path to zip.archive containing the dataset
args = parser.parse_args()

if args.dataset == "dogs":
    if not os.path.exists("data/dogs"):
        ZipFile(args.path).extractall("data/dogs")
        # ZipFile("data/dogs.zip").extractall("dogs")

    # define modes of datasets
    DATA_MODES = ['train', 'val', 'test']
    # define the size of image
    RESCALE_SIZE = 224
    DEVICE = torch.device("cuda")

    train_files, test_files = get_files("data/dogs/data/train", "data/dogs/data/test")
    print("Train data samples: ", train_files[:5])
    print("Number of train samples: ", len(train_files))
    print("Test data samples: ", test_files[:5])
    print("Number of test samples: ", len(test_files))

    batch_size = 100
    train_dataset = CustomDataset(train_files, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomDataset(test_files, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    assert (len(train_loader) > 0)
    assert (len(test_loader) > 0)

    # show_samples(test_dataset)
