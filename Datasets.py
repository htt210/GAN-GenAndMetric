import numpy as np
import torch
import random
from torchvision import transforms, datasets, utils
import sklearn.datasets


class NoiseDataset:
    def __init__(self, distr='Gaussian', dim=2, is_image=False):
        self.distr = distr
        self.dim = dim
        self.is_image = is_image

    def next_batch(self, batch_size=64, device=None):
        if self.distr == 'Gaussian':
            if self.is_image:
                return torch.randn(batch_size, self.dim, 1, 1, device=device)
            return torch.randn(batch_size, self.dim, device=device)
        else:
            return 'Not supported distribution'


class ToyMissingDataset:
    def __init__(self, dataset='8Gaussian', stddev=0.02, drop_centers=None, scale=1):
        self.centers = []
        self.stddev = stddev
        self.scale = scale
        self.range = 1
        if dataset == '8Gaussian':
            self.centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
        elif dataset == '25Gaussian':
            for x in range(-2, 3):
                for y in range(-2, 3):
                    # point[0] += 2 * x
                    # point[1] += 2 * y
                    self.centers.append((2 * x, 2*y))
            self.range = 2
        elif dataset == 'Swissroll':
            pass
        self.dataset = dataset
        if drop_centers is not None:
            drop_centers = sorted(drop_centers, reverse=True)
            for center in drop_centers:
                del self.centers[center]

    def next_batch(self, batch_size, device):
        if self.dataset == 'Swissroll':
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            return torch.FloatTensor(data).to(device) * self.scale

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * self.stddev
            center = random.choice(self.centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        return torch.FloatTensor(dataset).to(device) * self.scale


class MNISTDataset:
    def __init__(self, train=True):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('~/github/data/mnist', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1, shuffle=True)

        self.data = []
        for i, (x, y) in enumerate(self.train_loader):
            self.data.append(x.view(1, -1))
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc: self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)


class CelebADataset:
    def __init__(self, dataroot, image_size):
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = datasets.ImageFolder(root=dataroot,
                                       transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=True, num_workers=4)

        self.data = []
        for i, x in enumerate(dataloader):
            self.data.append(x)
        self.loc = 0
        self.n_samples = len(self.data)

    def next_batch(self, batch_size, device):
        if self.loc + batch_size > self.n_samples:
            random.shuffle(self.data)
            self.loc = 0

        batch = self.data[self.loc: self.loc + batch_size]
        self.loc += batch_size
        batch = torch.cat(batch, 0)
        return batch.to(device)

        # Decide which device we want to run on
        # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Plot some training images
        # real_batch = next(iter(dataloader))
        # plt.figure(figsize=(8, 8))
        # plt.axis("off")
        # plt.title("Training Images")
        # plt.imshow(
        #     np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))


def load_dataset(dataset, args):
    if dataset == 'mnist':
        return MNISTDataset(train=True)
    elif dataset == 'celeba':
        return CelebADataset(dataroot=args.dataroot, image_size=args.image_size)
    else:
        return ToyMissingDataset(dataset, stddev=args.stddev,
                                 drop_centers=args.drop_centers, scale=args.scale)
