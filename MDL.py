import torch
import torch.nn as nn
import numpy as np

from Generators import *
from Discriminators import *
from Classifier import *
from Sinkhorn import *
from Datasets import *

INFINITY = 1e10


def slerp(start, end, n_steps):
    """
    Return spherical interpolation from start to end
    :param start:
    :param end:
    :param n_steps:
    :return:
    """
    angle = ((start * end).sum(dim=1, keepdim=True) / start.norm(2, dim=1, keepdim=True)
             / end.norm(2, dim=1, keepdim=True)).acos()
    sinangle = torch.sin(angle)
    step = angle / n_steps
    inter_list = []
    for i in range(n_steps + 1):
        inter = torch.sin(angle - i * step) / sinangle * start + torch.sin(i * step) / sinangle * end
        inter_list.append(inter)
    return inter_list


def lerp(start, end, n_steps):
    """
    Return linear interpolation from `start` to `end`.
    :param start: B x C x W x H
    :param end:  B x C x W x H
    :param n_steps: number of steps
    :return: list of (n_steps+1) tensors of size B x C x W x H
    """
    step = (end - start) / n_steps
    inter_list = []
    for i in range(n_steps + 1):
        inter_list.append(start + i * step)

    return inter_list


def p_distance(x, y, p):
    batch_size = x.size(0)
    return (x - y).view(batch_size, -1).norm(p=p, dim=1)


def data_path_length(z_start, z_end, interpolation_method, n_steps, p,
                     G: Generator, D: Discriminator, classifier: Classifier):
    z_list = interpolation_method(z_start, z_end, n_steps)
    dists = torch.zeros(z_start.size(0)).to(z_start.device)
    with torch.no_grad():
        xp = G(z_start)
        for i in range(1, n_steps + 1):
            xi = G(z_list[i])
            dists += p_distance(xp, xi, p=p)
            xp = xi

        # required for inter class distance
        start_labels = None
        end_labels = None
        if classifier is not None:
            start_labels = classifier(G(z_start)).argmax(dim=1, keepdim=True)
            end_labels = classifier(G(z_end)).argmax(dim=1, keepdim=True)

    # print(dists.size(), start_labels.size())
    return dists, start_labels, end_labels


def class_pair_path_length(dists, start_labels, end_labels, nclasses):
    """
    Compute average distance between the two classes in each pair. Return list of class pairs sorted by distance
    :param dists:
    :param start_labels:
    :param end_labels:
    :param nclasses:
    :return:
    """
    nlabelpair = nclasses * nclasses
    labelid = start_labels * nclasses + end_labels
    labelid.squeeze_()
    class_len = {}
    len_mat = torch.empty(nclasses, nclasses).fill_(INFINITY)
    for i in range(nlabelpair):
        class_len[i] = INFINITY
        di = dists[labelid == i]
        if di.nelement() > 0:
            class_len[i] = di.mean().item()
        r = i // nclasses
        c = i % nclasses
        len_mat[r, c] = class_len[i]
    class_len = {k: v for k, v in sorted(class_len.items(), key=lambda item: item[1])}
    return class_len, len_mat


def perceptual_path_length(start, end, interpolation_method, G: Generator, D: Discriminator, inception_model):
    pass


def complexity(G: Generator, D: Discriminator, C: Classifier, noise: NoiseDataset):
    pass

