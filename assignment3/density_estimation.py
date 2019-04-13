#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4''
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

from samplers import distribution1, distribution4
from torch import optim
from torch.autograd import grad

class WDLoss(nn.Module):
    def __init__(self, lamda):
        super(WDLoss, self).__init__()

        self.lamda = lamda

    def forward(self, p, q, grad_z):
        return p.mean() - q.mean() + self.lamda*(grad_z.norm(dim=1) - 1).mean()**2


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, p, q):
        log_2 = 0.30102999566

        log_e_p = torch.mean(torch.log(p))
        log_e_q = torch.mean(torch.log(1 - q))

        return -1*(log_2 + 0.5*log_e_p + 0.5*log_e_q)

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()

        self.fcs = []
        self.acs = []

        for i, (_in, _out) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(_in, _out)
            nn.init.xavier_uniform_(fc.weight)

            self.fcs.append(fc)
            self.acs.append(nn.Sigmoid() if (i + 1 == len(layers) - 1) else nn.ReLU())

        self.linears = nn.ModuleList(self.fcs)

    def forward(self, x):
        for fc, ac in zip(self.fcs, self.acs):
            x = ac(fc(x))

        return x

num_epochs = 30
batch_size = 512
epoch_size = 75
dim = 2

jsd_output = []
wd_output = []

thetas = np.arange(-1.0, 1.0, 0.1)

for i, theta in enumerate(thetas):

    jsd_discr = MLP([dim, 10, 20, 1])
    wd_discr = MLP([dim, 10, 20, 1])

    jsd_optimizer = optim.SGD(jsd_discr.parameters(), lr=1e-3, momentum=0.9)
    wd_optimizer = optim.SGD(wd_discr.parameters(), lr=1e-3, momentum=0.9)

    jsd_criterion = JSDLoss()
    wd_criterion = WDLoss(15)

    print('[theta %d] training of delimiting functions started...' % (i))

    for epoch in range(num_epochs):
        jsd_running_loss = 0.0
        wd_running_loss = 0.0

        for i, (p_batch, q_batch) in enumerate(zip(distribution1(0, batch_size), distribution1(theta, batch_size))):
            if i == epoch_size:
                break

            jsd_optimizer.zero_grad()
            wd_optimizer.zero_grad()

            p_batch = torch.from_numpy(p_batch).type(torch.FloatTensor)
            q_batch = torch.from_numpy(q_batch).type(torch.FloatTensor)

            u_d = torch.from_numpy(np.random.uniform(0, 1, (batch_size, dim))).type(torch.FloatTensor)
            z_batch = u_d*p_batch + (1 - u_d)*q_batch
            z_batch.requires_grad = True

            jsd_p_o, jsd_q_o = (jsd_discr(batch) for batch in [p_batch, q_batch])
            wd_p_o, wd_q_o, wd_z_o = (wd_discr(batch) for batch in [p_batch, q_batch, z_batch])

            wd_z_o_grad = grad(wd_z_o.sum(), z_batch)[0]

            jsd_loss = jsd_criterion(jsd_p_o, jsd_q_o)
            jsd_loss.backward()

            wd_loss = wd_criterion(wd_p_o, wd_q_o, wd_z_o_grad)
            wd_loss.backward()

            jsd_optimizer.step()
            wd_optimizer.step()

            wd_running_loss += wd_loss.item()
            jsd_running_loss += jsd_loss.item()
            if i % 50 == 49:
                print('[%s][%d, %d] loss: %.3f' %
                    ('JSD', epoch + 1, i + 1, jsd_running_loss / 50))
                print('[%s][%d, %d] loss: %.3f' %
                    ('WD', epoch + 1, i + 1, wd_running_loss / 50))

                jsd_running_loss = 0.0
                wd_running_loss = 0.0

    print('[theta %d] training of delimiting functions completed.' % (i))

    p_batch, q_batch =  next(zip(distribution1(0, batch_size), distribution1(theta, batch_size)))

    p_batch = torch.from_numpy(p_batch).type(torch.FloatTensor)
    q_batch = torch.from_numpy(q_batch).type(torch.FloatTensor)

    jsd_output.append(jsd_discr(q_batch).mean())
    wd_output.append(wd_discr(q_batch).mean())

############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(thetas, jsd_output, 'r')
plt.plot(thetas, wd_output, 'b')
plt.legend(['JSD', 'WD'])
plt.title(r'$D(x)$')
plt.savefig('q1_3.png')

estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
