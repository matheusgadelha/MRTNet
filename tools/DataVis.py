import numpy as np
import torch
import os

try:
    import visdom
    vis = visdom.Visdom()
except ImportError:
    print "Visdom not available. No loss plotted."
    vis = None


class LossLogger(object):

    def __init__(self, name, caption=None):
        self.name = name
        self.caption = caption
        self.window = None
        self.history = []

    
    def update(self, value, step=10):
        val = value.cpu().data.numpy()
        self.history.append(val)

        if len(self.history) % step == 0:
            xs = np.arange(len(self.history)-step, len(self.history))
            ys = np.array(self.history[-step:]).reshape((step,))

            if vis is not None:
                if self.window is None:
                    self.window = vis.line(X=xs, Y=ys, 
                            opts=dict(title=self.name, caption=self.caption))
                else:
                    vis.line(X=xs, Y=ys, win=self.window, update='append')

            else:
                if not os.path.exists("log"):
                    os.makedirs("log")
                np.save(os.path.join("log", self.name+".npy"), self.history)

                
                

