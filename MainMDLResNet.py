import torch
import random
import argparse
import os
import datetime

#seed = 37
#torch.manual_seed(seed)
#random.seed(seed)

from Datasets import *
from Discriminators import *
from Generators import *
from GANs import *
# import model_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # architecture and hyper parameters
    parser.add_argument('-root', type=str, default='gangen', help='root folder for storing outputs')
    parser.add_argument('-loss', type=str, default='gan', help='loss function: gan | wgan')
    parser.add_argument('-arch', type=str, default='resnet', help='architecture: mlp | dcgan')
    parser.add_argument('-dataset', type=str, default='mnist',
                        help='dataset: fashionmnistImage | celeba | cifar10 | mnistImage')
    parser.add_argument('-real_weight', type=float, default=1, help='weight of a real example')
    parser.add_argument('-fake_weight', type=float, default=1, help='weight of a fake example')
    parser.add_argument('-ndf', type=int, default=64, help='number of filters in D')
    parser.add_argument('-ngf', type=int, default=64, help='number of filters in G')
    parser.add_argument('-lrd', type=float, default=2e-4, help='learning rate for D')
    parser.add_argument('-lrg', type=float, default=2e-4, help='learning rate for G')
    parser.add_argument('-nd', type=int, default=1, help='number of D iterations per GAN iteration')
    parser.add_argument('-ng', type=int, default=1, help='number of G iterations per GAN iteration')
    parser.add_argument('-gp_weight', type=float, default=0., help='weight of grad pen')
    parser.add_argument('-gp_center', type=float, default=0., help='grad pen center')
    parser.add_argument('-gp_inter', type=float, default=None,
                        help='grad pen interpolation: 0 | 1 | None <=> on fake | on real | random')
    parser.add_argument('-optimizer', type=str, default='adam', help='optimizer: adam | sgd')
    parser.add_argument('-beta1', type=float, default=0.0, help='beta1 for adam')
    parser.add_argument('-beta2', type=float, default=0.9, help='beta2 for adam')

    # noise options
    parser.add_argument('-noise_dist', type=str, default='Gaussian', help='noise distribution: Gaussian')
    parser.add_argument('-noise_dim', type=int, default=128, help='dimensionality of noise distribution')

    # training options
    parser.add_argument('-niters', type=int, default=200001, help='number of training iteration')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-lr_decay', type=float, default=0.99, help='lr decay rate')
    parser.add_argument('-lr_decay_interval', type=int, default=1000, help='decay lr after N iterations')
    parser.add_argument('-device', type=str, default='cuda:0', help='device: cuda:x | cpu')

    # log options
    parser.add_argument('-log_interval', type=int, default=10000, help='log interval')
    parser.add_argument('-show_grad', type=bool, default=False, help='show gradients for 2D data')
    parser.add_argument('-show_path', type=bool, default=False, help='show interpolation paths')
    parser.add_argument('-save_image', type=bool, default=True, help='save image for image datasets')
    parser.add_argument('-save_model', type=int, default=100000, help='save model every N iteration')
    parser.add_argument('-show_maxima', type=bool, default=True, help='show f(t) for real datapoints')
    parser.add_argument('-noise_range', type=float, default=100., help='range of t in f(t)')
    parser.add_argument('-noise_step', type=float, default=1., help='step of t in f(t)')
    parser.add_argument('-noise_direct', type=bool, default=True, help='use gradient for direction in f(t)')
    parser.add_argument('-nrow', type=int, default=8, help='nrow in extrema plot')
    parser.add_argument('-ncol', type=int, default=8, help='ncol in extrema plot')
    parser.add_argument('-is_image', action='store_true', default=True, help='work with image')

    # mdl options
    parser.add_argument('-mdl', type=int, default=1, help='compute mdl')
    parser.add_argument('-inter_method', type=str, default='slerp', help='interpolation method: lerp | slerp')
    parser.add_argument('-classifier', type=str, default='~/github/data/mnist/mnist_mlp.t7', help='classifier for MDL')
    parser.add_argument('-n_steps', type=int, default=100, help='number of steps when compute data path length')
    parser.add_argument('-p', type=float, default=2, help='p-norm')
    parser.add_argument('-nbatch', type=int, default=100, help='number of mini-batches when computing data path length')

    # sinkhorn options
    parser.add_argument('-shp', type=float, default=2, help='p-norm for Sinkhorn')
    parser.add_argument('-sheps', type=float, default=0.1, help='eps for Sinkhorn')
    parser.add_argument('-shmaxiter', type=int, default=1000, help='max iter for Sinkhorn')

    args = parser.parse_args()
    print(args)
    prefix = os.path.expanduser('~/github/figs/' + args.root + '/')
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix = prefix + args.loss + '-' + args.arch + '-' + args.dataset + '-'

    for k, v in args.__dict__.items():
        if k != 'loss' and k != 'arch' and k != 'dataset' and k != 'device' \
                and k != 'log_interval' and 'show' not in str(k) and 'save' not in str(k) \
                and 'nrow' not in k and 'ncol' not in k \
                and k != 'classifier' and k != 'root':
            prefix += k[0] + k[-1] + str(v) + '-'

    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix += '/' + str(datetime.datetime.now()).replace(' ', '-')
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix, len(prefix))
    args.prefix = prefix

    img_size = 32
    args.nc = 3
    # if args.dataset == 'cifar10':
    #     img_size = 64
    #     args.nc = 3
    # if args.dataset == 'fashionMNIST' or args.dataset == 'mnistImage':
    #     # img_size = 64
    #     print('args.dataset', args.dataset)
    #     args.nc = 1

    args.nz = args.noise_dim
    args.nx = img_size * img_size
    args.image_size = img_size

    # G = MLPGenerator(nz=args.nz, nx=args.nx, nhidden=args.nhidden, nhiddenlayer=args.nlayer)
    # D = MLPDiscriminator(nx=args.nx,  nhidden=args.nhidden, nhiddenlayer=args.nlayer, wgan_output=(args.loss == 'wgan'))

    # G = DCGenerator(nz=args.nz, ngf=args.ngf, img_size=args.image_size, nc=args.nc)
    # D = DCDiscriminator(img_size=args.image_size, nc=args.nc, ndf=args.ndf)

    # G = model_resnet.ResNetGenerator(z_dim=args.nz)
    # D = model_resnet.ResNetDiscriminator()

    with open(prefix + '/config.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(G) + '\n')
        f.write(str(D) + '\n')

    print(G)
    print(D)
    GAN(G, D, args)
