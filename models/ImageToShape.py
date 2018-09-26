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


class MultiResImageToShape(Model):

    def __init__(self, size, dim, batch_size=64, kernel_size=2, name="MRI2S",
        pretrained=False, arch=True):
        super(MultiResImageToShape, self).__init__(name)

        self.size = size
        self.dim = dim
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        if arch == 'vgg':
            self.encoder = torchvision.models.vgg11(pretrained=pretrained)
        elif arch == 'alexnet':
            self.encoder = torchvision.models.alexnet(pretrained=pretrained)
        self.encoder.classifier._modules['6'] = nn.Linear(4096, 16*1024)
        self.dec_modules = nn.ModuleList()
        self.base_size = 16

        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        custom_nfilters = [128, 128, 128, 256, 512, 512, 1024, 1024, 1024]
        custom_nfilters.reverse()
        custom_nfilters = np.array(custom_nfilters)

        current_size = self.base_size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            in_channels = custom_nfilters[layer_num-1]
            print in_channels
            out_channels = custom_nfilters[layer_num]
            conv_dec = MultiResConvTranspose1d('up{}'.format(layer_num),
                    in_channels, out_channels)
            current_size *= 2
            in_channels = out_channels
            layer_num += 1

            self.dec_modules.append(conv_dec)

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(custom_nfilters[-1]*3, 128, 1, stride=1, padding=0))
        self.final_conv.add_module('bn_final', 
                nn.BatchNorm1d(128))
        self.final_conv.add_module('relu_final',
                nn.ReLU(inplace=True))
        self.final_conv.add_module('final_conv2',
                nn.ConvTranspose1d(128, 3, 1, stride=1, padding=0))
        self.final_conv.add_module('tanh_final',
                nn.Tanh())


    def forward(self, x):
        mr_enc0 = self.encoder(x).view(self.batch_size, -1, self.base_size)
        mr_enc1 = self.pool(mr_enc0)
        mr_enc2 = self.pool(mr_enc1)
        mr_enc = [mr_enc0, mr_enc1, mr_enc2]

        dec_tensors = []
        dec_tensors.append(mr_enc)

        for i in xrange(0, len(self.dec_modules)-1):
            dec_tensors.append(self.dec_modules[i](dec_tensors[-1]))

        conv_out = self.dec_modules[-1](dec_tensors[-1])
        out0 = conv_out[0]
        out1 = self.upsample(conv_out[1])
        out2 = self.upsample(self.upsample(conv_out[2]))
        out = torch.cat((out0, out1, out2), 1)
        return self.final_conv(out)


    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path)
        print "Points saved."


