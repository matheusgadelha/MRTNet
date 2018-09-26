import numpy as np
import glob
import torch.utils.data
import os

from tools.PointCloudDataset import *


class ImageToPC(torch.utils.data.Dataset):
    
    def __init__(self, root_dir):
        self.filepaths = glob.glob(root_dir+"/*.obj")
        self.root_dir = root_dir

    
    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        return torch.FloatTensor(read_obj(self.filepaths[idx])[:, 0:3].transpose())

