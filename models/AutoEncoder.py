import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Model import Model
import tools.DataVis as DataVis
from tools.PointCloudDataset import save_objs
from tools import Ops
from modules.nnd import NNDModule


class PointCloudEncoder(nn.Module):

    def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=16, 
            init_channels=16):
        super(PointCloudEncoder, self).__init__()
        self.size = size
        self.dim = dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.enc_size =  enc_size
        self.init_channels = init_channels

        conv_enc = nn.Sequential()

        current_size = self.size
        in_channels = self.dim
        out_channels = self.init_channels
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size > 16:
            conv_enc.add_module('conv{}'.format(layer_num), 
                    nn.Conv1d(in_channels, out_channels, self.kernel_size, 
                        stride=2,
                        padding=padding))
            conv_enc.add_module('bn{}'.format(layer_num), 
                    nn.BatchNorm1d(out_channels))
            conv_enc.add_module('lrelu{}'.format(layer_num),
                    nn.LeakyReLU(0.2, inplace=True))

            current_size /= 2
            in_channels = out_channels
            out_channels *= 2
            layer_num += 1

        self.conv_enc = conv_enc

        self.fc = nn.Linear(16*in_channels, self.enc_size)

    def forward(self, x):
        t = self.conv_enc(x).view(self.batch_size, -1)
        out = self.fc(t)
        return out


class PointCloudDecoder(nn.Module):

    def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=16, 
            init_channels=1024):
        super(PointCloudDecoder, self).__init__()
        self.size = size
        self.dim = dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.enc_size =  enc_size
        self.init_channels = init_channels

        self.fc = nn.Linear(self.enc_size, 16*self.init_channels)

        conv_dec = nn.Sequential()

        current_size = 16*2
        in_channels = self.init_channels
        out_channels = in_channels/2
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            conv_dec.add_module('conv{}'.format(layer_num), 
                    nn.ConvTranspose1d(in_channels, out_channels, self.kernel_size, 
                        stride=2,
                        padding=padding))
            conv_dec.add_module('bn{}'.format(layer_num), 
                    nn.BatchNorm1d(out_channels))
            conv_dec.add_module('lrelu{}'.format(layer_num),
                    nn.LeakyReLU(0.2, inplace=True))

            current_size *= 2
            in_channels = out_channels
            out_channels /= 2
            layer_num += 1

        conv_dec.add_module('conv{}'.format(layer_num), 
                nn.ConvTranspose1d(in_channels, self.dim, self.kernel_size, 
                    stride=2,
                    padding=padding))
        conv_dec.add_module('lrelu{}'.format(layer_num),
                nn.Tanh())

        self.conv_dec = conv_dec


    def forward(self, x):
        t = self.fc(x).view(self.batch_size, self.init_channels, 16)
        out = self.conv_dec(t)
        return out


class MultiResBlock1d(nn.Module):

    def __init__(self, name, in_channels, out_channels, blocktype, activation):
        super(MultiResBlock1d, self).__init__()
        
        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name

        self.conv0 = nn.Sequential()
        self.conv0.add_module('{}_conv0'.format(self.name),
                blocktype(self.in_channels*2, 
                    self.out_channels, 
                    kernel_size=2, 
                    stride=2, 
                    padding=0))
        self.conv0.add_module('{}_bn0'.format(self.name),
                nn.BatchNorm1d(self.out_channels))
        self.conv0.add_module('{}_activation0'.format(self.name),
                activation)

        self.conv1 = nn.Sequential()
        self.conv1.add_module('{}_conv1'.format(self.name),
                blocktype(self.in_channels*3, 
                    self.out_channels, 
                    kernel_size=2, 
                    stride=2, 
                    padding=0))
        self.conv1.add_module('{}_bn1'.format(self.name),
                nn.BatchNorm1d(self.out_channels))
        self.conv1.add_module('{}_activation1'.format(self.name),
                activation)

        self.conv2 = nn.Sequential()
        self.conv2.add_module('{}_conv2'.format(self.name),
                blocktype(self.in_channels*2, 
                    self.out_channels, 
                    kernel_size=2, 
                    stride=2, 
                    padding=0))
        self.conv2.add_module('{}_bn2'.format(self.name),
                nn.BatchNorm1d(self.out_channels))
        self.conv2.add_module('{}_activation2'.format(self.name),
                activation)

    def forward(self, x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        in0 = torch.cat((x0, self.upsample(x1)), 1)
        in1 = torch.cat((self.pool(x0), x1, self.upsample(x2)), 1)
        in2 = torch.cat((self.pool(x1), x2), 1)

        out0 = self.conv0(in0)
        out1 = self.conv1(in1)
        out2 = self.conv2(in2)

        return [out0, out1, out2]


class MultiResConv1d(MultiResBlock1d):

    def __init__(self, name, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super(MultiResConv1d, self).__init__(
                name, in_channels, out_channels, nn.Conv1d, activation=activation)


class MultiResConvTranspose1d(MultiResBlock1d):

    def __init__(self, name, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super(MultiResConvTranspose1d, self).__init__(
                name, in_channels, out_channels, nn.ConvTranspose1d, activation=activation)


class ChamferLoss(nn.Module):

    def __init__(self, n_samples=1024, cuda_opt=True):
        super(ChamferLoss, self).__init__()
        self.n_samples = n_samples
        self.dist = NNDModule()
        self.cuda_opt = cuda_opt


    def chamfer(self, a, b):
        pcsize = a.size()[1]

        a = torch.t(a)
        b = torch.t(b)
        mma = torch.stack([a]*pcsize)
        mmb = torch.stack([b]*pcsize).transpose(0,1)
        d = torch.sum((mma-mmb)**2,2).squeeze()

        return torch.min(d, 1)[0].sum() + torch.min(d, 0)[0].sum()


    def chamfer_batch(self, a, b):
        pcsize = a.size()[-1]
        
        if pcsize != self.n_samples:
            indices = np.arange(pcsize).astype(int)
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices[:self.n_samples]).cuda()
            a = a[:, :, indices]
            b = b[:, :, indices]

        a = torch.transpose(a, 1, 2).contiguous()
        b = torch.transpose(b, 1, 2).contiguous()

        if self.cuda_opt:
            d1, d2 = self.dist(a, b)
            out = torch.sum(d1) + torch.sum(d2)
            return out
        else:
            d = Ops.batch_pairwise_dist(a, b)
            return torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()

        #mma = torch.stack([a]*self.n_samples, dim=1)
        #mmb = torch.stack([b]*self.n_samples, dim=1).transpose(1,2)
        #d = torch.sum((mma-mmb)**2,3).squeeze()
        #d = pd



    def forward(self, a, b):
        batch_size = a.size()[0]
        assert(batch_size == b.size()[0])
        loss = self.chamfer_batch(a, b)
        return loss/(float(batch_size) * 1)


class MultiResChamferLoss(nn.Module):

    def __init__(self, n_samples=1024):
        super(MultiResChamferLoss, self).__init__()
        self.n_samples = n_samples
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def chamfer_batch(self, a, b):
        pcsize = a.size()[-1]
        
        if pcsize > self.n_samples:
            pcsize = self.n_samples

            indices = np.arange(pcsize)
            np.random.shuffle(indices)
            indices = indices[:self.n_samples]

            a = a[:, :, indices]
            b = b[:, :, indices]

        a = torch.transpose(a, 1, 2)
        b = torch.transpose(b, 1, 2)
        #d = Ops.batch_pairwise_dist(a, b)
        mma = torch.stack([a]*self.n_samples, dim=1)
        mmb = torch.stack([b]*self.n_samples, dim=1).transpose(1,2)
        d = torch.sum((mma-mmb)**2,3).squeeze()
        #d = pd

        return (torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum())/float(pcsize)


    def forward(self, a, b):
        batch_size = a[0].size()[0]

        b_samples = []
        b_samples.append(b)
        b_samples.append(self.pool(b_samples[-1]))
        b_samples.append(self.pool(b_samples[-1]))

        loss = 0.0
        for i in [0]:
            loss += self.chamfer_batch(a[i], b_samples[i])

        return 1e3*loss/float(batch_size)



class ChamferWithNormalLoss(nn.Module):

    def __init__(self, normal_weight=0.001, n_samples=1024):
        super(ChamferWithNormalLoss, self).__init__()
        self.normal_weight = normal_weight
        self.nlogger = DataVis.LossLogger("normal component")
        self.n_samples = n_samples
       

    def forward(self, a, b):
        pcsize = a.size()[-1]
        
        if pcsize != self.n_samples:
            indices = np.arange(pcsize)
            np.random.shuffle(indices)
            indices = indices[:self.n_samples]
            a = a[:, :, indices]
            b = b[:, :, indices]

        a_points = torch.transpose(a, 1, 2)[:, :, 0:3]
        b_points = torch.transpose(b, 1, 2)[:, :, 0:3]
        pd = Ops.batch_pairwise_dist(a_points, b_points)
        #mma = torch.stack([a_points]*self.n_samples, dim=1)
        #mmb = torch.stack([b_points]*self.n_samples, dim=1).transpose(1,2)
        d = pd

        a_normals = torch.transpose(a, 1, 2)[:, :, 3:6]
        b_normals = torch.transpose(b, 1, 2)[:, :, 3:6]
        mma = torch.stack([a_normals]*self.n_samples, dim=1)
        mmb = torch.stack([b_normals]*self.n_samples, dim=1).transpose(1,2)
        d_norm = 1 - torch.sum(mma*mmb,3).squeeze()
        d += self.normal_weight * d_norm

        normal_min_mean = torch.min(d_norm, dim=2)[0].mean()
        self.nlogger.update(normal_min_mean)

        chamfer_sym = torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()
        chamfer_sym /= a.size()[0]

        return chamfer_sym


class SampleChamfer(nn.Module):

    def __init__(self, normal_weight=0.001, n_samples=1024):
        super(SampleChamfer, self).__init__()
        self.normal_weight = normal_weight
        self.nlogger = DataVis.LossLogger("normal component")
        self.n_samples = n_samples
 

    def chamfer(self, a, b):

        a_indices = np.arange(a.size()[-1])
        b_indices = np.arange(b.size()[-1])
        np.random.shuffle(a_indices)
        np.random.shuffle(b_indices)
        a_indices = a_indices[:self.n_samples]
        b_indices = b_indices[:self.n_samples]
        a = a[:, a_indices]
        b = b[:, b_indices]

        a = torch.t(a)
        b = torch.t(b)
        mma = torch.stack([a[:, 0:3]]*self.n_samples)
        mmb = torch.stack([b[:, 0:3]]*self.n_samples).transpose(0,1)
        d = torch.sum((mma-mmb)**2,2).squeeze()

        #return torch.min(d, 1)[0].sum() + torch.min(d, 0)[0].sum()
        return torch.min(d, 0)[0].sum()

    def forward(self, a, b):
#        pcsize = a.size()[-1]
#        
#        if pcsize != self.n_samples:
#            indices = np.arange(pcsize)
#            np.random.shuffle(indices)
#            indices = indices[:self.n_samples]
#            a = a[:, :, indices]
#            b = b[:, :, indices]
#
#        a_points = torch.transpose(a, 1, 2)[:, :, 0:3]
#        b_points = torch.transpose(b, 1, 2)[:, :, 0:3]
#        mma = torch.stack([a_points]*self.n_samples, dim=1)
#        mmb = torch.stack([b_points]*self.n_samples, dim=1).transpose(1,2)
#        d = torch.sum((mma-mmb)**2,3).squeeze()
#
#        a_normals = torch.transpose(a, 1, 2)[:, :, 3:6]
#        b_normals = torch.transpose(b, 1, 2)[:, :, 3:6]
#        mma = torch.stack([a_normals]*self.n_samples, dim=1)
#        mmb = torch.stack([b_normals]*self.n_samples, dim=1).transpose(1,2)
#        d_norm = 1 - torch.sum(mma*mmb,3).squeeze()
#        d += self.normal_weight * d_norm
#
#        normal_min_mean = torch.min(d_norm, dim=2)[0].mean()
#        self.nlogger.update(normal_min_mean)
#
#        chamfer_sym = torch.min(d, dim=2)[0].sum() + torch.min(d, dim=1)[0].sum()
#        chamfer_sym /= a.size()[0]

        return self.chamfer(a, b)



class SinkhornLoss(nn.Module):

    def __init__(self, n_iter=20, eps=1.0, batch_size=64, enc_size=512):
        super(SinkhornLoss, self).__init__()
        self.eps = eps
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.normal_noise = torch.FloatTensor(batch_size, enc_size)


    def forward(self, x):
        bsize = x.size()[0]
        assert bsize == self.batch_size

        self.normal_noise.normal_()
        y = Variable(self.normal_noise.cuda())

        #Computes MSE cost
        mmx = torch.stack([x]*bsize)
        mmy = torch.stack([x]*bsize).transpose(0, 1)
        c = torch.sum((mmx-mmy)**2,2).squeeze()

        k = (-c/self.eps).exp()
        b = Variable(torch.ones((bsize, 1))).cuda()
        a = Variable(torch.ones((bsize, 1))).cuda()

        #Sinkhorn iterations
        for l in range(self.n_iter):
            a = Variable(torch.ones((bsize, 1))).cuda() / (torch.mm(k, b))
            b = Variable(torch.ones((bsize, 1))).cuda() / (torch.mm(k.t(), a))

        loss = torch.mm(k * c, b)
        loss = torch.sum(loss*a)
        return loss



class GeodesicChamferLoss(nn.Module):

    def __init__(self):
        super(GeodesicChamferLoss, self).__init__()


    def forward(self, a, b):
        pass


class L2WithNormalLoss(nn.Module):

    def __init__(self):
        super(L2WithNormalLoss, self).__init__()
        self.nlogger = DataVis.LossLogger("normal w/ L2")
        self.L1 = nn.L1Loss()

    def forward(self, a, b):
        position_loss = self.L1(a[:, 0:3, :], b[:, 0:3, :])
        normal_loss = torch.mean(1 - Ops.cosine_similarity(a[:, 3:6, :], b[:, 3:6, :]))
        self.nlogger.update(normal_loss)

        return normal_loss


class PointCloudAutoEncoder(Model):

    def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=16,
            noise=0,
            name="PCAutoEncoder"):
        super(PointCloudAutoEncoder, self).__init__(name)

        self.size = size
        self.dim = dim
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.enc_size =  enc_size
        self.noise_factor = noise
        self.enc_noise = torch.FloatTensor(self.batch_size, self.enc_size)

        self.encoder = PointCloudEncoder(self.size, self.dim, 
                batch_size = self.batch_size, 
                enc_size = self.enc_size, 
                kernel_size = self.kernel_size)

        self.decoder = PointCloudDecoder(self.size, self.dim,
                batch_size = self.batch_size, 
                enc_size = self.enc_size, 
                kernel_size = self.kernel_size)

       # if self.dim == 6:
       #     self.normal_decoder = PointCloudDecoder(self.size, 3, 
       #             batch_size = self.batch_size, 
       #             enc_size = self.enc_size, 
       #             kernel_size = self.kernel_size)


    def forward(self, x):
        encoding = self.encoder(x)
        self.enc_noise.normal_()

        added_noise = Variable(self.noise_factor*self.enc_noise.cuda())

        encoding += added_noise
        x_prime = self.decoder(encoding)
        if self.dim == 6:
            x_normal = x_prime[:, 3:6, :]
            x_normal = F.normalize(x_normal)
            result = torch.cat((x_prime[:, 0:3, :], x_normal), dim=1)
        else:
            result = x_prime
        
        return result


    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path)
        print "Points saved."


class NormalReg(nn.Module):

    def __init__(self):
        super(NormalReg, self).__init__()

    def forward(self, x):

        mean = torch.mean(x, dim=0).pow(2)
        cov = Ops.cov(x)

        cov_loss = torch.mean(
                (Variable(torch.eye(cov.size()[0]).cuda())-cov)
                .pow(2))

        return torch.mean(mean) + cov_loss


class PointCloudVAE(PointCloudAutoEncoder):

    def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=16, 
            reg_fn=NormalReg(),
            noise = 0,
            name="PCVAE"):
        super(PointCloudVAE, self).__init__(size, dim, batch_size, enc_size, kernel_size, 
                noise=noise, name=name)
        self.reg_fn = reg_fn
        self.noise = torch.FloatTensor(self.batch_size, self.enc_size)


    def encoding_regularizer(self, x):
        return self.reg_fn(self.encoder(x))


    def sample(self):
        self.noise.normal_()
        return self.decoder(Variable(self.noise.cuda()))


class MultiResVAE(Model):

    def __init__(self, size, dim, batch_size=64, enc_size=100, kernel_size=2, 
            reg_fn=NormalReg(),
            noise = 0,
            name="MLVAE"):
        super(MultiResVAE, self).__init__(name)

        self.reg_fn = reg_fn

        self.size = size
        self.dim = dim
        self.enc_size = enc_size
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.enc_modules = nn.ModuleList()
        self.dec_modules = nn.ModuleList()
        self.upsample = Ops.NNUpsample1d()
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
	self.noise_factor = noise

        self.enc_noise = torch.FloatTensor(self.batch_size, self.enc_size)

        custom_nfilters = [3, 32, 64, 128, 256, 512, 512, 1024, 1024, 1024]
        custom_nfilters = np.array(custom_nfilters)
        custom_nfilters[1:] /= 2
        self.last_size = 16

        self.noise = torch.FloatTensor(self.batch_size, self.enc_size)

        current_size = self.size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        n_channels = []
        n_channels.append(custom_nfilters[layer_num-1])
        while current_size > self.last_size:
            in_channels = custom_nfilters[layer_num-1]
            out_channels = custom_nfilters[layer_num]
            conv_enc = MultiResConv1d('down{}'.format(layer_num),
                    in_channels, out_channels)
            current_size /= 2
            in_channels = out_channels
            n_channels.append(out_channels)
            layer_num += 1

            self.enc_modules.append(conv_enc)

        self.enc_fc = nn.Linear(3*self.last_size*in_channels, self.enc_size)
        #self.enc_fc_mean = nn.Linear(3*self.last_size*in_channels, self.enc_size)
        #self.enc_fc_var = nn.Linear(3*self.last_size*in_channels, self.enc_size)
        self.dec_fc = nn.Linear(self.enc_size, self.last_size*n_channels[-1])

        self.final_feature = 128
        n_channels.reverse()
        n_channels[-1] = self.final_feature
        current_size = self.last_size
        layer_num = 1
        padding = (self.kernel_size - 1)/2
        while current_size < self.size:
            in_channels = n_channels[layer_num-1]
            out_channels = n_channels[layer_num]
            conv_dec = MultiResConvTranspose1d('up{}'.format(layer_num),
                    in_channels, out_channels)
            current_size *= 2
            in_channels = out_channels
            layer_num += 1

            self.dec_modules.append(conv_dec)

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(self.final_feature*3, 128, 1, stride=1, padding=0))
        self.final_conv.add_module('bn_final', 
                nn.BatchNorm1d(128))
        self.final_conv.add_module('relu_final',
                nn.ReLU(inplace=True))
        self.final_conv.add_module('final_conv2',
                nn.ConvTranspose1d(128, 3, 1, stride=1, padding=0))
        self.final_conv.add_module('tanh_final',
                nn.Tanh())


    def enc_forward(self, x):
        x0 = x
        x1 = self.pool(x)
        x2 = self.pool(x1)

        enc_tensors = []
        enc_tensors.append([x0, x1, x2])

        for enc_op in self.enc_modules:
            enc_tensors.append(enc_op(enc_tensors[-1]))

        t0 = enc_tensors[-1][0]
        t1 = self.upsample(enc_tensors[-1][1])
        t2 = self.upsample(self.upsample(enc_tensors[-1][2]))
        t = torch.cat((t0, t1, t2), 1).view(self.batch_size, -1)

        encoding = self.enc_fc(t)
        return encoding, enc_tensors
        #encoding_mean = self.enc_fc_mean(t)
        #encoding_var = self.enc_fc_var(t)
        #return (encoding_mean, encoding_var)


    def dec_forward(self, x):
 
        mr_enc0 = self.dec_fc(x).view(self.batch_size, -1, self.last_size)
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

#
#    def reparameterize(self, mu, logvar):
#        if self.training:
#          std = logvar.mul(0.5).exp_()
#          eps = Variable(std.data.new(std.size()).normal_())
#          return eps.mul(std).add_(mu)
#        else:
#          return mu
#

    def forward(self, x):
        encoding = self.enc_forward(x)[0]
        self.enc_noise.normal_()

        added_noise = Variable(self.noise_factor*self.enc_noise.cuda())

        encoding += added_noise
        return self.dec_forward(encoding)


    def encoding_regularizer(self, x):
        return self.reg_fn(self.enc_forward(x)[0])


    def sample(self):
        self.noise.normal_()
        return self.dec_forward(Variable(self.noise.cuda()))


    def save_results(self, path, data, start_idx=0):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path, start_idx)
        print "Points saved."


class EncodingSVM(Model):

    def __init__(self, enc_size, n_classes, ae_model, batch_size, name="EncSVM"):
        super(EncodingSVM, self).__init__(name)

        self.batch_size = batch_size
        self.enc_size = enc_size
        self.n_classes = n_classes
        self.ae_model = ae_model

        self.upsample = Ops.NNUpsample1d()

        alpha = 32
        self.pools = []
        self.pools.append(nn.MaxPool1d(kernel_size=alpha, stride=alpha))
        self.pools.append(nn.MaxPool1d(kernel_size=alpha/2, stride=alpha/2))
        self.pools.append(nn.MaxPool1d(kernel_size=alpha/4, stride=alpha/4))
        #self.pools.append(nn.MaxPool1d(kernel_size=alpha/8, stride=alpha/8))

        self.fc = nn.Linear(self.enc_size, self.n_classes)

    def forward(self, x):
        enc, features = self.ae_model.enc_forward(x)
        descriptor = []
        for i, p in enumerate(self.pools):
            t0 = p(features[i][0])
            t1 = self.upsample(p(features[i][1]))
            t2 = self.upsample(self.upsample(p(features[i][2])))
            descriptor.append(torch.cat((t0, t1, t2), 1))

        descriptor = torch.cat(descriptor, 1)
        descriptor = descriptor.view(self.batch_size, -1)

        return self.fc(descriptor)
