import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensity(nn.Module):
    """Defines a mixture density module.

    Params:
       input_dim: dimensionality of input
       output_dim: dimensionality of output
       num_guassians: number of guassians used in the approximation

    Returns:
       dict containing the parameters of a mixture of gaussians distribution
    """
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MixtureDensity, self).__init__()
        self.output_dim = (output_dim + 2) * num_gaussians
        self.num_gaussians = num_gaussians
        self.dense = nn.Linear(input_dim, self.output_dim)
        nn.init.xavier_normal(self.dense.weight)

    def forward(self, x):
        x = self.dense(x)
        n_g = self.num_gaussians
        # subtract largest value to help numerical stability
        alpha = F.softmax(x[:, :n_g] - torch.max(x[:, :n_g]))
        # add small value also for numerical stability
        sigma = torch.exp(x[:, n_g:2*n_g]) + 0.00001
        mu = x[:, 2*n_g:]
        return {'alpha': alpha, 'sigma': sigma, 'mu': mu}


class MLP3MDN(nn.Module):
    """A multilayer perceptron with a mixture density output layer."""

    def __init__(self, input_dim, output_dim, num_gaussians,
                 hidden_dims=(128, 128, 128)):
        super(MLP3MDN, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dims[0])
        self.dense2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dense3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.mdn = MixtureDensity(hidden_dims[0], output_dim, num_gaussians)

        nn.init.xavier_normal(self.dense1.weight)
        nn.init.xavier_normal(self.dense2.weight)
        nn.init.xavier_normal(self.dense3.weight)

    def forward(self, x):
        x = F.relu(self.dense1(x), inplace=True)
        x = F.relu(self.dense2(x), inplace=True)
        x = F.relu(self.dense3(x), inplace=True)
        return self.mdn(x)


class MDNLoss(nn.Module):
    def __init__(self, num_gaussians, output_dim):
        super(MDNLoss, self).__init__()
        self.ng = num_gaussians
        self.c = output_dim
        self.norm_constant = 1.0 / np.power(2*np.pi, self.c/2.0)

    def forward(self, dist_params, targets):
        # extract parameters and prepare for computation
        alpha, sigma, mu = dist_params['alpha'], dist_params['sigma'], dist_params['mu']
        mu = mu.contiguous()
        reshape_mu = mu.view(-1, self.ng, self.c)
        reshape_t = targets.repeat(1, self.ng).view(-1, self.ng, self.c)

        # evaluate multivariate pdf on targets t
        squares = torch.pow(reshape_mu - reshape_t, 2)
        norm_sq = torch.sum(squares, dim=2)
        inv_sigma_sq = 1.0 / (torch.pow(sigma, 2))
        div = -0.5 * norm_sq * inv_sigma_sq
        sigma_pow = torch.pow(sigma, self.c)
        final = torch.exp(div) * self.norm_constant * (1. / sigma_pow)

        # compute neg log loss
        weighted = final * alpha
        prob = torch.sum(weighted, dim=1)
        logprob = -torch.log(prob)
        return torch.mean(logprob)
