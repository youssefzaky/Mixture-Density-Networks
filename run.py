import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.autograd import Variable

from network import MLP3MDN, MDNLoss

from data import data_1d

config = yaml.load(open('configs/1d_function_example.yml'))


def numpy_loss(t, alpha, sigma, mu):
    from scipy.stats import multivariate_normal
    logprob = []
    for i in range(t.shape[0]):
        prob = 0
        for j in range(config['num gaussians']):
            mean = mu[i, j]
            cov = np.diag([sigma[i, j] ** 2] * config['output dim'])
            prob += alpha[i, j] * multivariate_normal.pdf(t[i], mean=mean, cov=cov)
        logprob.append(np.log(prob))
    return -np.mean(np.array(logprob))


data_dict = data_1d(config['num samples'])
data_in, data_out = data_dict['input'], data_dict['output']
data_in, data_out = data_out, data_in
batchsize = config['batchsize']

net = MLP3MDN(input_dim=1, output_dim=config['output dim'],
              num_gaussians=config['num gaussians'])
net.train()
loss_module = MDNLoss(config['num gaussians'], config['output dim'])
optimizer = Adam(net.parameters(), lr=0.0001)

for epoch in range(config['num epochs']):
    p = np.random.permutation(len(data_in))
    inp, out = data_in[p], data_out[p]

    for batch in range(len(data_in) // batchsize):
        index = batch*batchsize
        # note inputs and outputs are reversed here
        batch_inputs = inp[index:index + batchsize]
        batch_targets = out[index:index + batchsize]

        optimizer.zero_grad()
        pred = net(Variable(torch.from_numpy(batch_inputs)))
        loss = loss_module(pred, Variable(torch.from_numpy(batch_targets)))
        loss.backward()
        optimizer.step()

    print('Epoch: %s, Train Loss: %s' % (epoch, loss.data[0]))


params = net(Variable(torch.from_numpy(data_in)))
NGAUSSIAN = config['num gaussians']
NOUTPUTS = config['output dim']
NSAMPLE = config['num samples']
alpha = params['alpha'].data.numpy()
sigma = params['sigma'].data.numpy()
mu = params['mu'].data.numpy()

samples = np.zeros((NSAMPLE, NOUTPUTS))

for i in range(NSAMPLE):
    alpha_idx = np.random.choice(NGAUSSIAN, size=1, p=alpha[i])[0]
    s_sigma = sigma[i, alpha_idx]
    mu_idx = alpha_idx * NOUTPUTS
    s_mu = mu[i, mu_idx:mu_idx + NOUTPUTS]
    cov = np.diag([s_sigma ** 2] * NOUTPUTS)
    samples[i, :] = np.random.multivariate_normal(s_mu, cov)

plt.figure(figsize=(10, 10))
plt.subplot(121, adjustable='box', aspect=1)
plt.plot(data_in, data_out, "o")
plt.subplot(122, adjustable='box', aspect=1)
plt.plot(data_in, samples, "o")
plt.show()
