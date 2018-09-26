import torch
import torch.nn as nn
import time
import numpy as np
from torch.autograd import Variable

from modules.nnd import NNDModule
from tools.Ops import batch_pairwise_dist

dist =  NNDModule()

p1 = torch.rand(64, 3, 4096).cuda()
p2 = torch.rand(64, 3, 4096).cuda()
points1 = Variable(p1.cuda(), requires_grad=True)
points2 = Variable(p2.cuda(), requires_grad=True)

indices = np.arange(4096).astype(int)
np.random.shuffle(indices)
indices = torch.from_numpy(indices[:1024]).cuda()
spoints1 = points1[:, :, :].contiguous()
spoints2 = points2[:, :, indices].contiguous()

#points1 = points1.transpose(1,2).contiguous()
#points2 = points2.transpose(1,2).contiguous()

#points1 = torch.transpose(points1, 1, 2)
#points2 = torch.transpose(points2, 1, 2)

#start = time.time()
#pd = batch_pairwise_dist(spoints1.transpose(1,2).contiguous(), spoints2.transpose(1,2).contiguous())
#loss = pd.min(dim=1)[0].sum() + pd.min(dim=2)[0].sum()
#loss.backward()
#end = time.time()
#print(points1.grad[0, :, :], points2.grad[0, :, :])
#print loss
#print "Runtime: {}s".format(end-start)

start = time.time()
dist1, dist2 = dist(spoints1.transpose(1,2).contiguous(), spoints2.transpose(1,2).contiguous())
#print(dist1, dist2)
loss = dist1.sum() + dist2.sum()
#print(loss)
loss.backward()
end = time.time()
print(points1.grad[0, :, :], points2.grad[0, :, :])
print loss
print "Runtime: {}s".format(end-start)

