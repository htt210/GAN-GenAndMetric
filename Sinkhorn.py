import torch
import torch.nn as nn
import numpy as np

import Generators

# Adapted from https://github.com/dfdazac/wassdistance
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        p (float): p-norm of the distance, default p=2
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, p=2, reduction='none', device='cuda'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.p = p
        self.device = device

    def forward(self, x, y):
        print('x, y', x.size(), y.size())
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = y_points = C.size(0)
        # x_points = x.shape[-2]
        # y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=self.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu, device=self.device)
        v = torch.zeros_like(nu, device=self.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        print(u.size(), v.size(), C.size())
        return (-C + u.unsqueeze(-1).to(self.device) + v.unsqueeze(-2).to(self.device)) / self.eps

    def _cost_matrix(self, x, y, batch_size=100):
        "Returns the matrix of $|x_i-y_j|^p$."
        if x.dim() == 2:
            if x.size(0) > batch_size:
                nbatchx = int(np.floor(x.size(0) / batch_size))
                nbatchy = int(np.floor(y.size(0) / batch_size))
                print('nbatchx %d, nbatchy %d' % (nbatchx, nbatchy))
                C = None
                startx = 0
                for i in range(nbatchx):
                    print('i', i)
                    batchx = x[startx: startx + batch_size]
                    Ci = None
                    starty = 0
                    for j in range(nbatchy):
                        # print('j', j)
                        batchy = y[starty: starty + batch_size]
                        batchx_col = batchx.unsqueeze(-2).to(self.device)
                        batchy_lin = batchy.unsqueeze(-3).to(self.device)
                        Cij = torch.sum((torch.abs(batchx_col - batchy_lin)) ** self.p, -1)
                        starty += batch_size
                        # print('Cij', Cij.size())
                        Ci = Cij if Ci is None else torch.cat((Ci, Cij), dim=1)
                        # print('Ci', Ci.size())
                    startx += batch_size
                    C = Ci if C is None else torch.cat((C, Ci), dim=0)
                return C

        x_col = x.unsqueeze(-2).to(self.device)
        y_lin = y.unsqueeze(-3).to(self.device)
        C = torch.sum((torch.abs(x_col - y_lin)) ** self.p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def point_cloud_dist(x, y, eps, p, max_iter, device='cuda'):
    x = x.to(device)
    y = y.to(device)
    sinkhorn = SinkhornDistance(eps=eps, p=p, max_iter=max_iter, device=device)
    nx = x.size(0)
    ny = y.size(0)
    assert nx == ny, "Currently only compare two point clouds with the same size"
    x = x.view(nx, -1)
    y = y.view(ny, -1)

    dist, P, C = sinkhorn(x, y)
    return dist, P, C


def point_cloud_dist_g(G: Generators.Generator, noise_data, real_data, n_samples, batch_size, eps, p, max_iter, device):
    n_batches = int(np.ceil(n_samples / batch_size))
    fakes = []
    reals = []
    with torch.no_grad():
        for bidx in range(n_batches):
            noise = noise_data.next_batch(batch_size=batch_size, device=device)
            fake = G(noise)
            fakes.append(fake)
            real = real_data.next_batch(batch_size=batch_size, device=device)
            reals.append(real)
    fakes = torch.cat(fakes, dim=0)
    reals = torch.cat(reals, dim=0)
    return point_cloud_dist(x=fakes, y=reals, eps=eps, p=p, max_iter=max_iter, device=device)
