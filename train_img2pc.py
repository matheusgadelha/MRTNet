import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import os

from tools.Trainer import ImageToPCTrainer
from tools.PointCloudDataset import ImageToPointCloudDataset
from models.AutoEncoder import PointCloudVAE
from models.AutoEncoder import ChamferLoss
from models.AutoEncoder import ChamferWithNormalLoss
from models.AutoEncoder import L2WithNormalLoss
from models.ImageToShape import MultiResImageToShape

parser = argparse.ArgumentParser(description='MultiResolution image to shape model.')
parser.add_argument("-n", "--name", type=str, help="Name of the experiment.", default="MRI2PC")
parser.add_argument("-a", "--arch", type=str, help="Encoder architecture.", default="vgg")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size", default=64)
parser.add_argument("-pt", "--pretrained", type=str, help="Use pretrained net", default=True)
parser.add_argument("-c", "--category", type=str, help="Category code (all is possible)", default="all")
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(train=False)


#CHANGE THESE!!!
image_datapath = "/media/mgadelha/hd2/ShapenetRenderings"
pc_datapath = "/media/mgadelha/hd2/shapenet_4k"

if __name__ == '__main__':
    args = parser.parse_args()

    ptrain = None
    if args.pretrained == "False":
        ptrain = False
    elif args.pretrained == "True":
        ptrain = True

    full_name = "{}_{}_{}_{}".format(args.name, args.category, args.arch, ptrain)
    print full_name

    mri2pc = MultiResImageToShape(size=4096, dim=3, batch_size=args.batchSize, 
            name=full_name, pretrained=ptrain, arch=args.arch)
    #mri2pc.load('checkpoint')
    optimizer = optim.Adam(mri2pc.parameters(), lr=0.001)
    
    train_dataset = ImageToPointCloudDataset(image_datapath, pc_datapath, category=args.category, train_mode=True)
    test_dataset = ImageToPointCloudDataset(image_datapath, pc_datapath, category=args.category, train_mode=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batchSize, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=args.batchSize,
            shuffle=True, num_workers=2)

    log_dir = os.path.join("log", full_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trainer = ImageToPCTrainer(mri2pc, train_loader, test_loader,
            optimizer, ChamferLoss(cuda_opt=True), log_dir=log_dir)
    trainer.train(2000)

