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
from AutoEncoder import MultiResConv1d


class FoldingNet(Model):
    def __init__(self, size, batch_size=64, name="FoldingNet"):
        super(FoldingNet, self).__init__(name)

        self.fold = nn.Sequential(
                nn.Conv1d(2+1024, 1024, 1, stride=1, padding=0),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),

                nn.Conv1d(1024, 512, 1, stride=1, padding=0),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),

                nn.Conv1d(512, 256, 1, stride=1, padding=0),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),

                nn.Conv1d(256, 128, 1, stride=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                nn.Conv1d(128, 3, 1, stride=1, padding=0),
                nn.Tanh())

        #gd = int(np.sqrt(size))
        #grid = np.indices((gd,gd)).T.reshape(-1, 2).T.astype('float32')
        #grid /= gd-1
        #self.z = Variable(torch.from_numpy(grid).unsqueeze(0).cuda())
        self.z = Variable(torch.rand(1, 2, size).cuda())
        self.global_feat = nn.Parameter(torch.rand(1, 1024, 1))

    def forward(self):
        #self.z.data.uniform_(0, 1)
        inp = torch.cat((self.z, self.global_feat.expand(-1, -1, 1024)), 1)
        return self.fold(inp)


class UNetMRTDecoder(Model):

    def __init__(self, size, dim, batch_size=64, enc_size=100, 
            kernel_size=2,
            axis_file='rpt_axis.npy',
            name="PointSeg"):
        super(UNetMRTDecoder, self).__init__(name)

        self.init_channels = 128
        self.size = size
        self.dim = dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.enc_size = enc_size
        self.enc_modules = nn.ModuleList()
        self.dec_modules = nn.ModuleList()
        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        #self.z = nn.Parameter(torch.randn(self.size*3).view(batch_size, 3, -1))
        self.z = Variable(torch.randn(self.size*3).view(batch_size, 3, -1)).cuda()

        #custom_nfilters = [3, 128, 128, 128, 256, 265, 256, 
        #        512, 512, 512, 1024, 1024, 2048]
        custom_nfilters = [3, 4, 8, 16, 32, 64, 128, 
                128, 128, 128, 256, 256, 256]
        custom_nfilters = np.array(custom_nfilters)
        #custom_nfilters[1:] /= 4

        current_size = self.size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        n_channels = []
        while current_size > 64:
            in_channels = custom_nfilters[layer_num-1]
            out_channels = custom_nfilters[layer_num]
            conv_enc = MultiResConv1d('down{}'.format(layer_num),
                    in_channels, out_channels)
#            conv_enc = nn.Sequential()
#            conv_enc.add_module('conv{}'.format(layer_num), 
#                    nn.Conv1d(in_channels, out_channels, self.kernel_size, 
#                        stride=2,
#                        padding=padding))
#                        #axis=self.axis_on_level(layer_num-1)))
#            conv_enc.add_module('bn{}'.format(layer_num), 
#                    nn.BatchNorm1d(out_channels))
#            conv_enc.add_module('lrelu{}'.format(layer_num),
#                    nn.LeakyReLU(0.2, inplace=True))
            current_size /= 2
            in_channels = out_channels
            n_channels.append(out_channels)
            if out_channels < 1024:
                out_channels *= 2
            layer_num += 1

            self.enc_modules.append(conv_enc)

        n_channels.reverse()
        current_size = 64
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size//2:
            if layer_num == 1:
                in_channels = n_channels[layer_num-1]
            else:
                in_channels = n_channels[layer_num-1]*2
            out_channels = n_channels[layer_num]

            conv_dec = MultiResConvTranspose1d('up{}'.format(layer_num),
                    in_channels, out_channels)
#            conv_dec = nn.Sequential()
#            conv_dec.add_module('conv{}'.format(layer_num), 
#                    nn.ConvTranspose1d(in_channels, 
#                        out_channels, 
#                        self.kernel_size, 
#                        stride=2,
#                        padding=padding))
#            conv_dec.add_module('bn{}'.format(layer_num), 
#                    nn.BatchNorm1d(out_channels))
#            conv_dec.add_module('relu{}'.format(layer_num),
#                    nn.ReLU(inplace=True))

            current_size *= 2
            in_channels = out_channels
            layer_num += 1

            self.dec_modules.append(conv_dec)

        conv_dec = MultiResConvTranspose1d('up{}'.format(layer_num),
                in_channels, 256)
        self.dec_modules.append(conv_dec)

#        conv_dec = nn.Sequential()
#        conv_dec.add_module('conv{}'.format(layer_num), 
#                nn.ConvTranspose1d(in_channels, 256, self.kernel_size, 
#                    stride=2,
#                    padding=padding))
#        conv_dec.add_module('bn{}'.format(layer_num), 
#                nn.BatchNorm1d(256))
#        conv_dec.add_module('relu{}'.format(layer_num),
#                nn.ReLU(inplace=True))

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(256*3, 128, 1, stride=1, padding=0))
        self.final_conv.add_module('bn_final', 
                nn.BatchNorm1d(128))
        self.final_conv.add_module('relu_final',
                nn.ReLU(inplace=True))
        self.final_conv.add_module('final_conv2',
                nn.ConvTranspose1d(128, 3, 1, stride=1, padding=0))
        self.final_conv.add_module('tanh_final',
                nn.Tanh())
        
    def multires_cat(self, x, y):
        out0 = torch.cat((x[0], y[0]), 1)
        out1 = torch.cat((x[1], y[1]), 1)
        out2 = torch.cat((x[2], y[2]), 1)

        return [out0, out1, out2]

    def forward(self):
        x0 = self.z
        x1 = self.pool(x0)
        x2 = self.pool(x1)

        enc_tensors = []
        enc_tensors.append([x0, x1, x2])

        for enc_op in self.enc_modules:
            enc_tensors.append(enc_op(enc_tensors[-1]))
#
#        for t in enc_tensors:
#            print t.size()
#
#        print self.dec_modules
#
        #t = enc_tensors[-1].view(self.batch_size, -1)
        #encoding = self.enc_fc(t)

        dec_tensors = []
        #dec_tensors.append(self.dec_fc(encoding).view(self.batch_size, -1, 16))
        dec_tensors.append(self.dec_modules[0](enc_tensors[-1]))

        for i in xrange(1, len(self.dec_modules)-1):
            in_tensor = enc_tensors[-(i+1)]
            #in_tensor = torch.cat((dec_tensors[-1], in_tensor), 1)
            in_tensor = self.multires_cat(in_tensor, dec_tensors[-1])
            dec_tensors.append(self.dec_modules[i](in_tensor))

        conv_out = self.dec_modules[-1](dec_tensors[-1])

        out0 = conv_out[0]
        out1 = self.upsample(conv_out[1])
        out2 = self.upsample(self.upsample(conv_out[2]))

        out = torch.cat((out0, out1, out2), 1)

        return self.final_conv(out)


    def axis_on_level(self, l):
        nlevels = np.log2(self.axis.shape[0]+1)
        level = nlevels - l - 1
        a = int(2**level - 1)
        b = int(2**(level+1) - 1)
        return self.axis[a:b, :]


    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_segs(results, path)
        print "Segmentations saved."



class MRTDecoder(Model):

    def __init__(self, size, dim, batch_size=64, kernel_size=2, name="MRTDecoder"):
        super(MRTDecoder, self).__init__(name)

        self.size = size
        self.dim = dim
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        self.z = nn.Parameter(torch.randn(16*1024))
        self.dec_modules = nn.ModuleList()
        self.base_size = 16

        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        custom_nfilters = [128, 128, 128, 256, 512, 512, 1024, 1024, 1024]
        custom_nfilters.reverse()
        custom_nfilters = np.array(custom_nfilters)
        custom_nfilters[1:] /= 2

        current_size = self.base_size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            in_channels = custom_nfilters[layer_num-1]
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


    def forward(self):
        mr_enc0 = self.z.view(self.batch_size, -1, self.base_size)
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


class SRTDecoder(Model):

    def __init__(self, size, dim, batch_size=64, kernel_size=2, name="SRTDecoder"):
        super(SRTDecoder, self).__init__(name)

        self.size = size
        self.dim = dim
        self.kernel_size = kernel_size
        self.batch_size = batch_size

        self.z = nn.Parameter(torch.randn(16*1024))
        self.dec_modules = nn.ModuleList()
        self.base_size = 16

        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        custom_nfilters = [128, 128, 128, 256, 512, 512, 1024, 1024, 1024]
        custom_nfilters.reverse()
        custom_nfilters = np.array(custom_nfilters)
        custom_nfilters[1:] /= 2

        self.conv_dec = nn.Sequential()

        current_size = self.base_size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            in_channels = custom_nfilters[layer_num-1]
            out_channels = custom_nfilters[layer_num]

            self.conv_dec.add_module('conv{}'.format(layer_num), 
                    nn.ConvTranspose1d(in_channels, out_channels, self.kernel_size, 
                        stride=2,
                        padding=padding))
            self.conv_dec.add_module('bn{}'.format(layer_num), 
                    nn.BatchNorm1d(out_channels))
            self.conv_dec.add_module('lrelu{}'.format(layer_num),
                    nn.LeakyReLU(0.2, inplace=True))

            current_size *= 2
            in_channels = out_channels
            layer_num += 1

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(custom_nfilters[-1], 128, 1, stride=1, padding=0))
        self.final_conv.add_module('bn_final', 
                nn.BatchNorm1d(128))
        self.final_conv.add_module('relu_final',
                nn.ReLU(inplace=True))
        self.final_conv.add_module('final_conv2',
                nn.ConvTranspose1d(128, 3, 1, stride=1, padding=0))
        self.final_conv.add_module('tanh_final',
                nn.Tanh())


    def forward(self):
        feat = self.z.view(self.batch_size, -1, self.base_size)
        feat = self.conv_dec(feat)
        out = self.final_conv(feat)
        return out
        

    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path)
        print "Points saved."


