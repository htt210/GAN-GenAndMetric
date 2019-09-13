import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torchvision
from Discriminators import *
from Generators import *
from Datasets import *
from Logger import *


def cal_grad_pen(G: Generator, D: Discriminator, real_batch, fake_batch, args):
    # print(args.gp_inter)
    alpha = args.gp_inter
    device = args.device
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_batch.size(0), 1, device=device)
    alpha = alpha.expand(real_batch.size())

    interpolates = alpha * real_batch + ((1 - alpha) * fake_batch)
    interpolates.requires_grad_(True)
    disc_interpolates = D(interpolates)
    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - args.gp_center) ** 2).mean()
    return gradient_penalty


def GAN(G: Generator, D: Discriminator, args):
    D.to(args.device)
    G.to(args.device)

    ones = torch.ones(args.batch_size, 1, device=args.device)
    zeros = torch.zeros(args.batch_size, 1, device=args.device)

    optim_d = None
    optim_g = None
    if args.optimizer == 'adam':
        optim_g = optim.Adam(lr=args.lrg, params=G.parameters(), betas=(args.beta1, args.beta2))
        optim_d = optim.Adam(lr=args.lrd, params=D.parameters(), betas=(args.beta1, args.beta2))
    elif args.optimizer == 'sgd':
        optim_g = optim.SGD(lr=args.lrg, params=G.parameters(), momentum=args.momentum)
        optim_d = optim.SGD(lr=args.lrd, params=D.parameters(), momentum=args.momentum)
    else:
        raise Exception('Not supported optimizer: ' + args.optimizer)

    noise_data = NoiseDataset(distr=args.noise_dist, dim=args.noise_dim, is_image=args.is_image)
    real_data = load_dataset(args.dataset, args)

    criterion = nn.BCELoss()

    z_start = noise_data.next_batch(batch_size=32, device=args.device)
    z_end = noise_data.next_batch(batch_size=32, device=args.device)
    fixed_real = real_data.next_batch(args.nrow * args.ncol, device=args.device)
    fixed_noise = noise_data.next_batch(args.nrow * args.ncol, device=args.device)
    if args.noise_dim > 2:  # image data
        torchvision.utils.save_image(
            fixed_real.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
            args.prefix + '/real.png', nrow=args.nrow)
    noise_direction = torch.rand_like(fixed_real[0], device=args.device) if args.noise_direct else None

    for it in range(args.niters):
        if it % args.log_interval == 0:
            print('Iteration %d' % it)
            if args.show_grad:
                disp_grad(G, D, noise_data, real_data, criterion, it, args)
                disp_map(G, D, noise_data, real_data, criterion, it, args)
                disp_path(G, D, noise_data, real_data, z_start, z_end, criterion, it, args)
            if args.show_maxima:
                _, _, scores = compute_extrema(D, fixed_real, noise_direction, args.noise_range, args.noise_step)
                disp_extrema(scores, args.prefix + '/extrema_%05d.pdf' % it,
                             args.noise_range, args.noise_step, args.nrow, args.ncol)

                with torch.no_grad():
                    fixed_fake = G(fixed_noise)
                    torchvision.utils.save_image(
                        fixed_fake.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                        args.prefix + '/fake_%05d.png' % it, nrow=args.nrow)

        # train D for nd iterations
        for i in range(args.nd):
            optim_d.zero_grad()

            # train real
            real_batch = real_data.next_batch(args.batch_size, args.device)
            pred_real = D(real_batch)
            loss_real = criterion(pred_real, ones)
            # train fake
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch).detach()
            pred_fake = D(fake_batch)
            loss_fake = criterion(pred_fake, zeros)
            # grad pen
            grad_pen = 0
            if args.gp_weight > 0:
                grad_pen = args.gp_weight * cal_grad_pen(G, D, real_batch.detach(), fake_batch.detach(), args)
            # compute total loss
            loss = loss_real + loss_fake + grad_pen
            # update params
            loss.backward()
            optim_d.step()

        # train G for ng iterations
        for i in range(args.ng):
            optim_g.zero_grad()
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch)
            pred_fake = D(fake_batch)
            loss_g = criterion(pred_fake, ones)
            loss_g.backward()
            optim_g.step()

    return G, D
