"""
Code for demonstrating the relationship between Mutual information and Generalization on toy datasets
"""
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils, transforms
import argparse
import datetime

from Datasets import *
from Generators import *
from Discriminators import *
from GANs import *


def entropy1d(data, bins):
    eps = 1e-10
    dx = bins[1] - bins[0]

    hist, _ = np.histogram(data, bins=bins)
    prob = hist / hist.sum() / dx
    stableProb = prob.copy()
    stableProb[stableProb < eps] = eps
    ent = -(prob * np.log(stableProb)).sum() * dx

    # plt.figure(1)
    # plt.plot(prob)
    # plt.show()

    return ent


def entropy2D(data, bins):
    eps = 1e-10
    dx = bins[1] - bins[0]

    hist, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    prob = hist / hist.sum() / dx / dx
    stableProb = prob.copy()
    stableProb[stableProb < eps] = eps
    ent = -(prob * np.log(stableProb)).sum() * dx * dx

    # plt.figure(2)
    # plt.imshow(prob)
    # plt.colorbar()
    # plt.show()

    return ent


def mutualInfo(x, y, xy, nbins, drange):
    drangeMargin = np.linspace(-drange, drange, nbins + 1)
    entX = entropy1d(x, drangeMargin)
    entY = entropy1d(y, drangeMargin)
    entXY = entropy2D(xy, drangeMargin)

    return entX + entY - entXY


def mutualInfoGenerator(G, noise, real, nsamples=100000, nbins=10, drange=2, device='cuda'):
    """
    Monte Carlo estimate of mutual information between noise and data
    :param G: generator, could be the generator in GAN or decoder in VAE
    :param noise: noise dataset
    :param real: real dataset
    :param nsamples: number of noise samples
    :param nbins: number of bins in each dimension, the size of the joint histogram is (nbins^2) x (nbins^2)
    :param drange: data range [-drange, drange]
    :param device: device
    :return: MC estimate of mutual information between noise and data
    """

    batch_size = 100
    nbatch = nsamples // batch_size
    noisedata = []
    fakedata = []
    realdata = []
    bins = np.linspace(-drange, drange, nbins+1)
    with torch.no_grad():
        for i in range(nbatch):
            noisev = noise.next_batch(batch_size=batch_size, device=device)
            fakev = G(noisev)
            realv = real.next_batch(batch_size=batch_size, device=device)
            noisedata.append(noisev)
            fakedata.append(fakev)
            realdata.append(realv)
        noisedata = torch.cat(noisedata, dim=0).cpu().numpy()
        fakedata = torch.cat(fakedata, dim=0).cpu().numpy()
        realdata = torch.cat(realdata, dim=0).cpu().numpy()
        # quantize the data, assign a index to each region
        noisedigit = np.digitize(noisedata, bins=bins) - 1
        fakedigit = np.digitize(fakedata, bins=bins) - 1
        realdigit = np.digitize(realdata, bins=bins) - 1
        noisedigit[:, 0] = (noisedigit[:, 0] * nbins + noisedigit[:, 1]) / nbins / nbins
        fakedigit[:, 0] = (fakedigit[:, 0] * nbins + fakedigit[:, 1]) / nbins / nbins
        realdigit[:, 0] = (realdigit[:, 0] * nbins + realdigit[:, 1]) / nbins / nbins

        jointBins = np.linspace(0, 1, nbins * nbins + 1)
        jointNF = np.copy(noisedigit)
        jointNF[:, 1] = fakedigit[:, 0]
        muInNF = entropy1d(noisedigit, jointBins) + entropy1d(fakedigit, jointBins) - entropy2D(jointNF, jointBins)

        return muInNF

        # jointNR, _, _ = np.histogram2d(noisedigit[:, 0], realdigit[:, 0], bins=jointBins)
        # muInNR = entropy1d(noisedigit, jointBins) + entropy1d(realdigit, jointBins) - entropy2D(jointNR, jointBins)
        # plt.show(1)
        # return muInNR, jointNR, noisedata, fakedata, realdata


def disp_muin(muInFRs, jointFRs, muInNRs, jointNRs, noisedata, fakedata, realdata, iteration, args):
    fig, ax = plt.subplots(1, 1, size=(4, 4))
    ax.imshow(jointNRs[-1])
    fig.savefig(args.path + '/jointNR_%05d.pdf' % iteration, bbox_inches='tight')
    fig.close()


if __name__ == '__main__':
    # npoints = 10000000
    # drange = 4
    # nbins = 10
    # bins = np.linspace(-drange, drange, nbins + 1)

    # z = np.random.randn()
    # x = z * 0.5 + 0.1 + np.random.randn(npoints) * 0.1
    # data = np.random.randn(npoints) * 0.5 + 0.1
    # print(entropy1d(data, bins))

    # mean = np.array([0, 0])
    # cov = np.array([[1, 0.5], [0.5, 1]])
    # z = np.random.multivariate_normal(mean=mean, cov=cov, size=npoints)
    # x = np.random.multivariate_normal(mean=mean, cov=cov, size=npoints)

    # print(entropy2D(data2d, bins))

    # z = np.random.randn(npoints)
    # x = z + np.random.randn(npoints) / 5
    # zx, _, _, _ = plt.hist2d(z, x, bins=bins)
    # plt.colorbar()
    # plt.show()
    # mutualInfo(z, x, zx, nbins, drange)

    parser = argparse.ArgumentParser()
    parser.add_argument('-loss', type=str, default='gan', help='loss function: gan | wgan')
    parser.add_argument('-real_weight', type=float, default=1, help='weight of a real example')
    parser.add_argument('-fake_weight', type=float, default=1, help='weight of a real example')
    parser.add_argument('-arch', type=str, default='mlp', help='architecture: mlp | dcgan')
    parser.add_argument('-dataset', type=str, default='8Gaussian',
                        help='dataset: mnist | celeba | 8Gaussian | 25Gaussian | Swissroll')
    parser.add_argument('-nhidden', type=int, default=128, help='number of hidden unit in MLP')
    parser.add_argument('-nlayer', type=int, default=2, help='number of layer in MLP')
    parser.add_argument('-lrd', type=float, default=3e-2, help='learning rate for D')
    parser.add_argument('-lrg', type=float, default=1e-2, help='learning rate for G')
    parser.add_argument('-nd', type=int, default=1, help='number of D iterations per GAN iteration')
    parser.add_argument('-ng', type=int, default=1, help='number of G iterations per GAN iteration')
    parser.add_argument('-gp_weight', type=float, default=0., help='weight of grad pen')
    parser.add_argument('-gp_center', type=float, default=0., help='grad pen center')
    parser.add_argument('-gp_inter', type=float, default=1.,
                        help='grad pen interpolation: 0 | 1 | None <=> on fake | on real | random')
    parser.add_argument('-optimizer', type=str, default='sgd', help='optimizer: adam | sgd')
    parser.add_argument('-momentum', type=float, default=0.0, help='momentum for sgd')
    parser.add_argument('-beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('-beta2', type=float, default=0.99, help='beta2 for adam')
    parser.add_argument('-drop_centers', type=str, default=None, help='list of centers to drop, e.g. 1,10,3')

    parser.add_argument('-noise_dist', type=str, default='Gaussian', help='noise distribution: gaussian')
    parser.add_argument('-noise_dim', type=int, default=2, help='dimensionality of noise distribution')

    parser.add_argument('-niters', type=int, default=200001, help='number of training iteration')
    parser.add_argument('-log_interval', type=int, default=1000, help='log interval')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-stddev', type=float, default=0.02, help='stddev of Gaussians in toy datasets')
    parser.add_argument('-scale', type=float, default=1, help='scale of the toy datasets')
    parser.add_argument('-device', type=str, default='cuda:0', help='device: cuda:x | cpu')

    parser.add_argument('-show_grad', action='store_true', default=True, help='show gradients for 2D data')
    parser.add_argument('-show_path', action='store_true', default=False, help='show interpolation paths')
    parser.add_argument('-save_image', action='store_true', default=False, help='save image for image datasets')
    parser.add_argument('-save_model', type=int, default=100000, help='save model after iterations')
    parser.add_argument('-show_maxima', action='store_true', help='show f(t) for real datapoints')
    parser.add_argument('-inter_step', type=float, default=1e-2,
                        help='step of the interpolation, 1/inter_step is the number of interpolation steps')
    parser.add_argument('-inter_method', type=str, default='lerp', help='interpolation method: lerp | slerp')

    parser.add_argument('-muin', type=int, default=1, help='estimate mutual information')

    args = parser.parse_args()
    print(args)
    prefix = os.path.expanduser('~') + '/github/figs/'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix = prefix + 'muin_' + args.loss + '_' + args.arch + '_' + args.dataset + '_'

    args.is_image = False

    for k, v in args.__dict__.items():
        if k != 'loss' and k != 'arch' and k != 'dataset' and k != 'device' \
                and k != 'log_interval' and 'show' not in str(k) and 'save' not in str(k) \
                and 'inter_' not in k and 'noise' not in k:
            prefix += k[0] + k[-1] + '_' + str(v) + '_'

    if not os.path.exists(prefix):
        os.mkdir(prefix)
    prefix += '/' + str(datetime.datetime.now())
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix, len(prefix))
    args.prefix = prefix

    G = MLPGenerator(2, 2, args.nhidden, args.nlayer)
    D = MLPDiscriminator(2, args.nhidden, args.nlayer, wgan_output=(args.loss == 'wgan'))

    with open(prefix + '/config.txt', 'w') as f:
        f.write(str(args) + '\n')
        f.write(str(G) + '\n')
        f.write(str(D) + '\n')

    print(G)
    print(D)

    GAN(G, D, args)
    torch.save(G, prefix + '/G.t7')
    torch.save(D, prefix + '/D.t7')
