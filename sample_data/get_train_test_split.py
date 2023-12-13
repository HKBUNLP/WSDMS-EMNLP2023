import argparse
import os
import random
import traceback

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool, global_max_pool  

torch.backends.cudnn.enable=True
torch.backends.cudnn.benchmark=True

def get_train_test_split(args):
    labels_d = torch.load(os.path.join(args.root, args.dataset, 'raw',  f"{args.dataset}_labels.pt"))

    filenames_real = [x for x, y in labels_d.items() if y == 0]
    filenames_fake = [x for x, y in labels_d.items() if y == 1]

    random.shuffle(filenames_real)
    random.shuffle(filenames_fake)

    n_train_real = int(len(filenames_real) * (1-args.test_size))
    n_train_fake = int(len(filenames_fake) * (1-args.test_size))

    # Note: split should be fixed
    filenames_train = filenames_real[:n_train_real] + filenames_fake[n_train_fake:]
    filenames_test = filenames_real[n_train_real:] + filenames_fake[n_train_fake:]
    return filenames_train, filenames_test

if __name__ == "__main__":
    filenames_train, filenames_test = get_train_test_split(args)
    print(filenames_train)


