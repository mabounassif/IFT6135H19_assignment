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

from samplers import distribution3, distribution4
from torch import optim
from torch.autograd import grad

class WDLoss(nn.Module):
    def __init__(self, lamda):
        super(WDLoss, self).__init__()

        self.lamda = lamda

    def forward(self, p, q, grad_z):
        return p.mean() - q.mean() + self.lamda*(grad_z.norm(p=2) - 1).mean()**2


class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, p, q):
        log_2 = 0.30102999566

        log_e_p = torch.mean(torch.log(p))
        log_e_q = torch.mean(torch.log(1 - q))

        return -1*(log_2 + 0.5*log_e_p + 0.5*log_e_q)

class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()

        self.fcs = []
        self.acs = []

        for i, (_in, _out) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(_in, _out)
            nn.init.xavier_uniform_(fc.weight)

            self.fcs.append(fc)
            self.acs.append(nn.Sigmoid() if (i == len(layers) - 2) else nn.ReLU())

        self.linears = nn.ModuleList(self.fcs)

    def forward(self, x):
        for fc, ac in zip(self.fcs, self.acs):
            x = ac(fc(x))

        return x

num_epochs = 1000
batch_size = 512
epoch_size = 1000
dim = 1

discr = Discriminator([dim, 10, 20, 1])
optimizer = optim.SGD(discr.parameters(), lr=1e-3, momentum=0.9)
#  criterion = JSDLoss()
criterion = WDLoss(15)

print('training started...')

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, (p_batch, q_batch) in enumerate(zip(distribution3(batch_size), distribution4(batch_size))):
        if i == epoch_size:
            break

        optimizer.zero_grad()

        p_batch = torch.from_numpy(p_batch).type(torch.FloatTensor)
        q_batch = torch.from_numpy(q_batch).type(torch.FloatTensor)

        # generating z
        u_d = torch.from_numpy(np.random.uniform(0, 1, (batch_size, dim))).type(torch.FloatTensor)
        z_batch = u_d*p_batch + (1 - u_d)*q_batch
        z_batch.requires_grad=True

        p_o = discr(p_batch)
        q_o = discr(q_batch)
        z_o = discr(z_batch)

        z_o_grad = grad(z_o.sum(), z_batch, create_graph=True)[0]

        loss = criterion(p_o, q_o, z_o_grad)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('training done.')

############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density

r = xx # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')
