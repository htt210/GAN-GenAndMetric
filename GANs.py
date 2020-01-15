import torch
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torchvision
from Discriminators import *
from Generators import *
from Datasets import *
# from catastrophic.MuInToy import *
from Logger import *
import MDL
import Sinkhorn
import pickle as pkl
from Classifier import *


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

    noise_data = NoiseDataset(distr=args.noise_dist, dim=args.noise_dim, is_image=not args.arch == 'mlp')
    real_data = load_dataset(args.dataset, args)
    test_data = load_dataset(args.dataset, args, train=False)

    criterion = nn.BCELoss()

    z_start = noise_data.next_batch(batch_size=32, device=args.device)
    z_end = noise_data.next_batch(batch_size=32, device=args.device)

    if args.is_image:
        fixed_real = real_data.next_batch(args.nrow * args.ncol, device=args.device)
        fixed_noise = noise_data.next_batch(args.nrow * args.ncol, device=args.device)
        fixed_fakes = []
        fixed_fakes_scores = []
        noise_direction = torch.rand_like(fixed_real[0], device=args.device) if args.noise_direct else None
        torchvision.utils.save_image(
            fixed_real.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
            args.prefix + '/real.png', nrow=args.nrow, normalize=args.nc != 1)
        noise_range = torch.arange(-args.noise_range, args.noise_range, step=args.noise_step,
                                   device='cpu').view(-1).numpy()
        for i in range(len(noise_range)):
            noise_level = noise_range[i]
            # print(noise_level)
            real_noise = fixed_real + noise_level * noise_direction / noise_direction.norm()
            torchvision.utils.save_image(
                real_noise.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                args.prefix + '/real_noise_%06d.png' % i, nrow=args.nrow, normalize=args.nc != 1)

    gen_time = [5000, 10000, 20000, -1]
    ffs_idx = [[] for _ in range(len(gen_time))]
    gen_idx = 0

    # mdl logging variables
    wass_dists = []
    wass_disttes = []
    path_lengths = []
    its = []
    classifier = None

    for it in range(args.niters):
        if it % 100 == 0:
            print('Iteration %d' % it)

        if args.is_image and it == gen_time[gen_idx]:
            gen_idx += 1
            fixed_fakes.append(G(fixed_noise).data)
            torchvision.utils.save_image(
                fixed_fakes[-1].view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                args.prefix + '/fixed_fake_%06d.png' % it, nrow=1, normalize=args.nc != 1)
            fixed_fakes_scores.append([])

        if it % args.log_interval == 0:
            its.append(it)
            print('Loggin Iteration %d' % it)

            if args.show_grad:
                disp_grad(G, D, noise_data, real_data, criterion, it, args)
                disp_map(G, D, noise_data, real_data, criterion, it, args)
                disp_path(G, D, noise_data, real_data, z_start, z_end, criterion, it, args)

            if args.is_image:
                # calculate score for fixed fake
                for ffi in range(len(fixed_fakes)):
                    ffs_idx[ffi].append(it)
                    fixed_fakes_scores[ffi].append(D(fixed_fakes[ffi]).data.cpu().numpy())
                    print(it, 'fixed fake score', fixed_fakes_scores[ffi][-1][1][0])

                if args.show_maxima:
                    _, _, scores = compute_extrema(D, fixed_real, noise_direction, args.noise_range, args.noise_step)
                    disp_extrema(scores, args.prefix + '/extrema_%06d.pdf' % it,
                                 args.noise_range, args.noise_step, args.nrow, args.ncol)
                    with open(args.prefix + '/extrema.txt', 'a') as f:
                        f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

                    with torch.no_grad():
                        fixed_fake = G(fixed_noise)
                        torchvision.utils.save_image(
                            fixed_fake.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
                            args.prefix + '/fake_%06d.png' % it, nrow=args.nrow, normalize=args.nc != 1)
                        _, _, scores = compute_extrema(D, fixed_fake, noise_direction, args.noise_range,
                                                       args.noise_step)
                        disp_extrema(scores, args.prefix + '/extrema_fake_%06d.pdf' % it,
                                     args.noise_range, args.noise_step, args.nrow, args.ncol)
                        with open(args.prefix + '/extrema_fake.txt', 'a') as f:
                            f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')

            if args.mdl:
                if classifier is None:
                    print('loading classifier')
                    try:
                        classifier = torch.load(os.path.expanduser(args.classifier))
                        classifier.to(args.device)
                    except Exception as e:
                        print('Cannot load classifier\n', e)
                print('Computing MDL')
                print('Fake data path length')
                # compute fake data path length
                dists_list = []
                start_labels_list = []
                end_labels_list = []
                for i in range(args.nbatch):
                    # compute path length for 100 mini batches
                    z_starti = noise_data.next_batch(batch_size=args.batch_size, device=args.device)
                    z_endi = noise_data.next_batch(batch_size=args.batch_size, device=args.device)
                    dists, start_labels, end_labels = MDL.data_path_length(z_start=z_starti, z_end=z_endi,
                                                                           interpolation_method=MDL.slerp,
                                                                           n_steps=args.n_steps, p=args.p, G=G, D=D,
                                                                           classifier=classifier)
                    dists_list.append(dists)
                    start_labels_list.append(start_labels)
                    end_labels_list.append(end_labels)
                dists = torch.cat(dists_list, dim=0)
                dist_mean = dists.mean().item()
                path_lengths.append(dist_mean)
                start_labels = torch.cat(start_labels_list, dim=0)
                end_labels = torch.cat(end_labels_list, dim=0)
                class_len, len_mat = MDL.class_pair_path_length(dists=dists, start_labels=start_labels,
                                                                end_labels=end_labels, nclasses=classifier.nclasses())
                with open(args.prefix + '/class_len.txt', 'a') as clf:
                    clf.write('It_%06d_%f\n' % (it, dist_mean))
                    clf.write(str(class_len) + '\n')
                disp_mat(len_mat, args.prefix + '/len_mat_%06d.png' % it)
                # plt.figure()
                # plt.imshow(len_mat)
                # plt.show(5)

                print('Sinkhorn distance')
                shdist, shP, shC = Sinkhorn.point_cloud_dist_g(G=G, noise_data=noise_data, real_data=real_data,
                                                               n_samples=args.nbatch * args.batch_size,
                                                               batch_size=args.batch_size, eps=args.sheps, p=args.shp,
                                                               max_iter=args.shmaxiter, device=args.device)
                print('Train', shdist)
                shdistte, shPte, shCte = Sinkhorn.point_cloud_dist_g(G=G, noise_data=noise_data, real_data=test_data,
                                                                     n_samples=args.nbatch * args.batch_size,
                                                                     batch_size=args.batch_size, eps=args.sheps,
                                                                     p=args.shp,
                                                                     max_iter=args.shmaxiter, device=args.device)
                print('Test', shdistte)
                wass_dists.append(shdist.item())
                wass_disttes.append(shdistte.item())
                # display the trajectory
                with open(args.prefix + '/sinkhorn_dist.txt', 'a') as shf:
                    shf.write('It_%06d_%f\n' % (it, shdist))
                disp_mdl(path_length=path_lengths, wass_dist=wass_dists, it=its,
                         outfile=args.prefix + '/mdl_%06d.pdf' % it)
                disp_mdl(path_length=path_lengths, wass_dist=wass_disttes, it=its,
                         outfile=args.prefix + '/mdl_test_%06d.pdf' % it)
                with open(args.prefix + '/mdl.txt', 'w') as mdlf:
                    for i, w, wte, d in zip(its, wass_dists, wass_disttes, path_lengths):
                        mdlf.write('%d, %f, %f, %f\n' % (i, w, wte, d))

        if it % args.save_model == args.save_model - 1:
            print('Saving model')
            torch.save(G, args.prefix + '/G_%06d.t7' % it)
            torch.save(D, args.prefix + '/D_%06d.t7' % it)

        # train D for nd iterations
        for i in range(args.nd):
            optim_d.zero_grad()

            # train real
            real_batch = real_data.next_batch(args.batch_size, args.device)
            pred_real = D(real_batch)
            # train fake
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch).detach()
            pred_fake = D(fake_batch)
            # grad pen
            grad_pen = 0
            if args.gp_weight > 0:
                grad_pen = args.gp_weight * cal_grad_pen(G, D, real_batch.detach(), fake_batch.detach(), args)
            # compute total loss
            if args.loss == 'gan':
                loss_real = criterion(pred_real, ones) * args.real_weight
                loss_fake = criterion(pred_fake, zeros) * args.fake_weight
                loss = loss_real + loss_fake + grad_pen
            elif args.loss == 'wgan':
                loss = pred_fake.mean() - pred_real.mean() + grad_pen
            else:
                raise NotImplementedError(args.loss + ' loss is not implemented.')
            # update params
            loss.backward()
            optim_d.step()

        # train G for ng iterations
        for i in range(args.ng):
            optim_g.zero_grad()
            noise_batch = noise_data.next_batch(args.batch_size, args.device)
            fake_batch = G(noise_batch)
            pred_fake = D(fake_batch)
            if args.loss == 'gan':
                loss_g = criterion(pred_fake, ones)
            elif args.loss == 'wgan':
                loss_g = -pred_fake.mean()
            else:
                raise NotImplementedError(args.loss + ' loss is not implemented.')  # impossible to reach this but hey!
            loss_g.backward()
            optim_g.step()

    # plot scores of fixed_fakes
    nimg = args.nrow * args.ncol
    pkl.dump((gen_time, fixed_fakes, fixed_fakes_scores, ffs_idx), open(args.prefix + '/fixed_fake.t7', 'wb'))
    print(ffs_idx)
    for idx, (gi, ff, ffs) in enumerate(zip(gen_time, fixed_fakes, fixed_fakes_scores)):
        fffig, ffax = plt.subplots(nimg, 1, figsize=(4, 128))
        ffs = np.array(ffs)
        print(ffs.shape)
        for i in range(nimg):
            ffax[i].plot(ffs_idx[idx], ffs[:, i])
            ffax[i].set_ylim(-0.1, 1.1)
        plt.savefig(args.prefix + '/fixed_fake_scores_%06d.pdf' % gi, bbox_inches='tight')

    return G, D

# def WGAN(G: Generator, D: Discriminator, args):
#     D.to(args.device)
#     G.to(args.device)
#
#     ones = torch.ones(args.batch_size, 1, device=args.device)
#     zeros = torch.zeros(args.batch_size, 1, device=args.device)
#
#     optim_d = None
#     optim_g = None
#     if args.optimizer == 'adam':
#         optim_g = optim.Adam(lr=args.lrg, params=G.parameters(), betas=(args.beta1, args.beta2))
#         optim_d = optim.Adam(lr=args.lrd, params=D.parameters(), betas=(args.beta1, args.beta2))
#     elif args.optimizer == 'sgd':
#         optim_g = optim.SGD(lr=args.lrg, params=G.parameters(), momentum=args.momentum)
#         optim_d = optim.SGD(lr=args.lrd, params=D.parameters(), momentum=args.momentum)
#     else:
#         raise Exception('Not supported optimizer: ' + args.optimizer)
#
#     noise_data = NoiseDataset(distr=args.noise_dist, dim=args.noise_dim, is_image=args.is_image)
#     real_data = load_dataset(args.dataset, args)
#
#     criterion = nn.BCELoss()
#
#     z_start = noise_data.next_batch(batch_size=32, device=args.device)
#     z_end = noise_data.next_batch(batch_size=32, device=args.device)
#     fixed_real = real_data.next_batch(args.nrow * args.ncol, device=args.device)
#     fixed_noise = noise_data.next_batch(args.nrow * args.ncol, device=args.device)
#     noise_direction = torch.rand_like(fixed_real[0], device=args.device) if args.noise_direct else None
#     if args.noise_dim > 2:  # image data
#         torchvision.utils.save_image(
#             fixed_real.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
#             args.prefix + '/real.png', nrow=args.nrow, normalize=True)
#         noise_range = torch.arange(-args.noise_range, args.noise_range, step=args.noise_step,
#                                    device='cpu').view(-1).numpy()
#         for i in range(len(noise_range)):
#             noise_level = noise_range[i]
#             # print(noise_level)
#             real_noise = fixed_real + noise_level * noise_direction / noise_direction.norm()
#             torchvision.utils.save_image(
#                 real_noise.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
#                 args.prefix + '/real_noise_%06d.png' % i, nrow=args.nrow, normalize=True)
#
#     for it in range(args.niters):
#         if it % 100 == 0:
#             print('Iteration %d' % it)
#         if it % args.log_interval == 0:
#             print('Iteration %d' % it)
#             if args.show_grad:
#                 disp_grad(G, D, noise_data, real_data, criterion, it, args)
#                 disp_map(G, D, noise_data, real_data, criterion, it, args)
#                 disp_path(G, D, noise_data, real_data, z_start, z_end, criterion, it, args)
#             if args.show_maxima:
#                 _, _, scores = compute_extrema(D, fixed_real, noise_direction, args.noise_range, args.noise_step)
#                 disp_extrema(scores, args.prefix + '/extrema_%06d.pdf' % it,
#                              args.noise_range, args.noise_step, args.nrow, args.ncol)
#                 with open(args.prefix + '/extrema.txt', 'a') as f:
#                     f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')
#
#                 with torch.no_grad():
#                     fixed_fake = G(fixed_noise)
#                     torchvision.utils.save_image(
#                         fixed_fake.view((args.nrow * args.ncol, args.nc, args.image_size, args.image_size)),
#                         args.prefix + '/fake_%06d.png' % it, nrow=args.nrow, normalize=True)
#                     _, _, scores = compute_extrema(D, fixed_fake, noise_direction, args.noise_range, args.noise_step)
#                     disp_extrema(scores, args.prefix + '/extrema_fake_%06d.pdf' % it,
#                                  args.noise_range, args.noise_step, args.nrow, args.ncol)
#                     with open(args.prefix + '/extrema_fake.txt', 'a') as f:
#                         f.write('Iteration:' + str(it) + '\n' + str(scores) + '\n')
#
#         if it % args.save_model == args.save_model - 1:
#             print('Saving model')
#             torch.save(G, args.prefix + '/G_%06d.t7' % it)
#             torch.save(D, args.prefix + '/D_%06d.t7' % it)
#
#         # train D for nd iterations
#         for p in D.parameters():
#             p.requires_grad_(True)
#         for i in range(args.nd):
#             optim_d.zero_grad()
#
#             # train real
#             real_batch = real_data.next_batch(args.batch_size, args.device)
#             score_real = D(real_batch).mean()
#             # train fake
#             noise_batch = noise_data.next_batch(args.batch_size, args.device)
#             fake_batch = G(noise_batch).detach()
#             score_fake = D(fake_batch).mean()
#             # grad pen
#             grad_pen = 0
#             if args.gp_weight > 0:
#                 grad_pen = args.gp_weight * cal_grad_pen(G, D, real_batch.detach(), fake_batch.detach(), args)
#             # compute total loss
#             loss_d = score_fake - score_real + grad_pen
#             # update params
#             loss_d.backward()
#             optim_d.step()
#
#         # train G for ng iterations
#         for p in D.parameters():
#             p.requires_grad_(False)
#         for i in range(args.ng):
#             optim_g.zero_grad()
#             noise_batch = noise_data.next_batch(args.batch_size, args.device)
#             fake_batch = G(noise_batch)
#             score_fake = D(fake_batch).mean()
#             loss_g = -score_fake
#             loss_g.backward()
#             optim_g.step()
#
#     return G, D
