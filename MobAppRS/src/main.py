import argparse
import numpy as np
from time import time
from train import train
from data_loaders import load_data
import numpy as np

np.random.seed(555)
parser = argparse.ArgumentParser()
parser.add_argument('--aggregator', type=str, default='concat', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of apps and relations embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')#65536
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')#1e-7
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.6, help='size of training dataset')#lr, neigbour, aggregator, epochs 100, batch_Size 

show_loss = False
show_topk = True

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)