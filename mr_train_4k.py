import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import argparse

from tools.Trainer import VAETrainer
from tools.PointCloudDataset import PointCloudDataset
from models.AutoEncoder import MultiResVAE
from models.AutoEncoder import ChamferLoss
from models.AutoEncoder import ChamferWithNormalLoss
from models.AutoEncoder import L2WithNormalLoss

parser = argparse.ArgumentParser(description='Point Cloud Generator.')
parser.add_argument("-d", "--datapath", type=str, help="Dataset path.", default="")
parser.add_argument("-n", "--name", type=str, help="Name of the experiment", default="PointGen")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size", default=64)
parser.add_argument("-e", "--encSize", type=int, help="Encoding size", default=128)
parser.add_argument("-f", "--factorNoise", type=float, help="Noise factor", default=0.0)
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(train=False)


if __name__ == '__main__':
    args = parser.parse_args()

    vae = MultiResVAE(4096, 3, name=args.name, enc_size=args.encSize, 
            noise=args.factorNoise,
            batch_size=args.batchSize)
    #vae.load('checkpoint')
    optimizer = optim.Adam(vae.parameters(), lr=1e-5)
    
    dataset = PointCloudDataset(args.datapath)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
            shuffle=True, num_workers=2)
    trainer = VAETrainer(vae, loader, optimizer, ChamferLoss())
    trainer.train(2000)

