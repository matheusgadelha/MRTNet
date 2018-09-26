import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from Model import Model
from tools.PointCloudDataset import save_objs
from tools import Ops
import tools.DataVis as DataVis

from AutoEncoder import MultiResConvTranspose1d


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(UpConv, self).__init__()
        padding = (kernel_size - 1)/2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                padding=padding)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        return self.conv(self.upsample(x))


class VoxUNet(Model):

    def __init__(self, size, batch_size=64, kernel_size=3, name="VoxUNet"):
        super(VoxUNet, self).__init__(name)

        self.size = size
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        self.z = nn.Parameter(torch.randn(1,1,size,size,size))
        self.enc_modules = nn.ModuleList()
        self.dec_modules = nn.ModuleList()

        custom_nfilters = [1, 16, 32, 64, 128, 256, 512, 1024, 1024, 1024]
        #custom_nfilters.reverse()
        #custom_nfilters = np.array(custom_nfilters)
        #custom_nfilters[1:] /= 2

        nfilters = []

        #Encoder creation
        current_size = size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size > 2:
            in_channels = custom_nfilters[layer_num-1]
            out_channels = custom_nfilters[layer_num]
            conv_enc = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 
                        kernel_size=self.kernel_size,
                        padding=padding,
                        stride=2),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=False))
            current_size /= 2
            in_channels = out_channels
            layer_num += 1
            nfilters.append(out_channels)

            self.enc_modules.append(conv_enc)

        nfilters.reverse()

        #Decoder creation
        current_size = 2
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            if layer_num == 1:
                in_channels = nfilters[layer_num-1]
            else:
                in_channels = nfilters[layer_num-1]*2
            out_channels = nfilters[layer_num]
            conv_dec = nn.Sequential(
                    UpConv(in_channels, out_channels,
                        kernel_size=self.kernel_size),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=False))
            current_size *= 2
            in_channels = out_channels
            layer_num += 1
            nfilters.append(out_channels)

            self.dec_modules.append(conv_dec)

        self.final_conv = nn.Sequential(
                UpConv(nfilters[-2]*2, 1,
                    kernel_size=self.kernel_size),
                nn.Sigmoid())


    def forward(self):
        #Encoding
        enc_tensors = []
        enc_tensors.append(self.z)

        for enc_op in self.enc_modules:
            enc_tensors.append(enc_op(enc_tensors[-1]))

        #Decoding
        dec_tensors = []
        dec_tensors.append(self.dec_modules[0](enc_tensors[-1]))
        print dec_tensors[0].size()

        for i in xrange(1, len(self.dec_modules)-1):
            in_tensor = enc_tensors[-(i+1)]
            in_tensor = torch.cat((in_tensor, dec_tensors[-1]), 1)
            dec_tensors.append(self.dec_modules[i](in_tensor))

        final_input = torch.cat((dec_tensors[-1], enc_tensors[1]), 1)
        out = self.final_conv(final_input)

        return out


    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path)
        print "Points saved."

if __name__ == '__main__':
    net = VoxUNet(256, batch_size=1).cuda()
    net()
