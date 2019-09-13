import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def forward(self, *input):
        pass


class MLPGenerator(Generator):
    def __init__(self, nz, nx, nhidden, nhiddenlayer, negative_slope=0):
        super(MLPGenerator, self).__init__()
        self.net = nn.Sequential()
        i = 0
        self.net.add_module('linear_%d' % i, nn.Linear(nz, nhidden))
        self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        for i in range(1, nhiddenlayer):
            self.net.add_module('linear_%d' % i, nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % i, nn.LeakyReLU(inplace=True, negative_slope=negative_slope))
        self.net.add_module('linear_%d' % (i + 1), nn.Linear(nhidden, nx))
        # self.net.apply(weights_init)

    def forward(self, inputs):
        return self.net(inputs)
