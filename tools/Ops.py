import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def cov(data):
    dmean = torch.mean(data, dim=0)
    centered_data = data - dmean.expand_as(data)
    return torch.mm(centered_data.transpose(0,1), centered_data)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def batch_pairwise_dist(a,b):
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P


class NNUpsample1d(nn.Module):

    def __init__(self, scale=4):
        self.scale = scale
        super(NNUpsample1d, self).__init__()

    def forward(self, x):
        out = F.upsample(x.unsqueeze(3), scale_factor=self.scale, 
                mode='nearest')[:, :, :, 0]
        return out


def nplerp(x0, xn, n):
    interps = []
    interps.append(x0)
    for i in xrange(1, n):
	alpha = i * 1.0/(n-1)
	interps.append(x0 + alpha*(xn-x0))
    interps = np.array(interps)
    return interps


def rotmat_2d(theta):
    mat = torch.zeros(2,2)
    mat[0,0] = torch.cos(theta)
    mat[0,1] = -torch.sin(theta)
    mat[1,0] = torch.sin(theta)
    mat[1,1] = torch.cos(theta)

    return mat


def gridcoord_2d(w, h):
    max_dim = max(w, h)
    xs = torch.linspace(-max_dim/w, -max_dim/w, steps=w)
    ys = torch.linspace(-max_dim/h, -max_dim/h, steps=h)

    xc = xs.repeat(h)
    yc = ys.repeat(w,1).t().contiguous().view(-1)

    out = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1)), 1)
    return out


def resample_img(img, gridcoords, method='nearest'):
    if method=="nearest":
        round_coords = torch.round(gridcoords)
        out = img[round_coords]
        out.reshape(*(img.size()))
        return out


def transform_image(img, t):
    height = img.size()[0]
    width = img.size()[1]
    grid2d = gridcoord_2d(width, height)

    transf_grid = torch.mm(grid2d, t)
    resample_img(img, transf_grid)


if __name__ == '__main__':
    g = gridcoord_2d(2, 2)
    from IPython import embed; embed(); exit(-1)
