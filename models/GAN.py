import torch
import torch.nn as nn
import torch.nn.functional as F

from Model import Model
from tools.PointCloudDataset import save_objs
from tools import Ops


class LinearDiscriminator(Model):

    def __init__(self, input_size=256, n_layers=3, layer_size=256,
            name="LinearGenerator"):
        super(LinearDiscriminator, self).__init__(name)
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(layer_size)

        self.layers = []
        self.bns = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layer_size))
            else:
                self.layers.append(nn.Linear(input_size, layer_size))
            self.bns.append(nn.BatchNorm1d(layer_size))
        
        self.out = nn.Linear(layer_size, 1)


    def cuda(self):
        super(LinearDiscriminator, self).cuda()
        for l in self.layers:
            l.cuda()
        for b in self.bns:
            b.cuda()


    def forward(self, x):

        features = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                features.append(self.activation(self.bns[i](layer(x))))
            else:
                features.append(self.activation(self.bns[i](layer(features[-1]))))
        output = self.out(features[-1])
        return (output, features)


class LinearGenerator(Model):

    def __init__(self, enc_size=100, output_size=256, n_layers=3, layer_size=256,
            name="LinearGenerator"):
        super(LinearGenerator, self).__init__(name)
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.enc_size = enc_size
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(layer_size)

        self.net = nn.Sequential()
        for i in range(n_layers):
            if i == 0:
                self.net.add_module("linear{}".format(i),
                        nn.Linear(enc_size, layer_size))
            else:
                self.net.add_module("linear{}".format(i),
                        nn.Linear(layer_size, layer_size))
            self.net.add_module("bn{}".format(i),
                    nn.BatchNorm1d(layer_size))
            self.net.add_module("relu{}".format(i),
                    nn.ReLU())
        
        self.out = nn.Linear(layer_size, output_size)


    def forward(self, x):
        return self.out(self.net(x))


    def save_results(self, path, data):
        results = data.cpu().data.numpy()
        results = results.transpose(0, 2, 1)
        save_objs(results, path)
        print "Points saved."


class DiscriminatorBCELoss(nn.Module):

    def __init__(self):
        super(DiscriminatorBCELoss, self).__init__()
        self.BCE = nn.BCELoss()


    def forward(self, x):
        input_data, target = x
        input_logits, _ = input_data

        return self.BCE(F.sigmoid(input_logits), target)


class GeneratorFeatureLoss(nn.Module):

    def __init__(self):
        super(GeneratorFeatureLoss, self).__init__()


    def forward(self, x):
        real, fake = x
        _, real_features = real
        _, fake_features = fake

        real_features = torch.cat(real_features, dim=1)
        fake_features = torch.cat(fake_features, dim=1)

        real_mean = torch.mean(real_features, dim=0)
        fake_mean = torch.mean(fake_features, dim=0)

        real_cov = Ops.cov(real_features)
        fake_cov = Ops.cov(fake_features)

        mean_loss = torch.sum((real_mean-fake_mean).pow(2))
        cov_loss = torch.sum((real_cov-fake_cov).pow(2))

        return mean_loss + cov_loss


