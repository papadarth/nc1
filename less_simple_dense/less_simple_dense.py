#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pytorch test 1
"""

__author__  = "Tobias Wagner"

import numpy as np
import pylab as plt
import torch
import torch.nn as nn
from torch.autograd import Variable



## generate data ##
n   = 1000
t   = 20
a   = np.zeros((n,t),dtype=np.float32) 
## a
a[::4,5] = -1
a[::4,15] = -1
## b
a[1::4,5] = 1
a[1::4,15] = 1
## c
a[2::4,5] = -1
a[2::4,15] = 1
## d
a[3::4,5] = 1
a[3::4,15] = -1

a   += np.random.randn(n,t)*.3
b   = np.zeros((n), dtype=np.float32)
b[1::4] = 1
b[2::4] = 2
b[3::4] = 3


x_train = a[:800,:]
y_train = b[:800]

x_test  = a[800:,:]
y_test  = a[800:]


#plt.plot(a[:10,:].T);plt.show()
# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  
    
    def forward(self, x):
        out = self.linear(x)
	return out

class DenseLayers(nn.Module):
    def __init__(self, input_size, hidden1_size, output_size):
        super(DenseLayers, self).__init__()
        self.linear1    = nn.Linear(input_size, hidden1_size)
        self.batch_n    = nn.BatchNorm1d(hidden1_size)
        self.relu       = nn.ReLU()
        self.linear2    = nn.Linear(hidden1_size, output_size)

    def forward(self, x):
        h   = self.relu(self.batch_n(self.linear1(x)))
        out   = self.linear2(h)
        return out

input_size	= t
output_size	= 1
hidden1_size    = 100

### Linear Regression Model
#model = LinearRegression(input_size, output_size)
### 2 Dense Layer Model
model = DenseLayers(input_size, hidden1_size, output_size)

criterion	= nn.MSELoss()
optimizer	= torch.optim.SGD(model.parameters(), lr=.001)
rnd             = np.arange(0,800)

for i in range(10):
    print i
    for epoch in range(500):
        rnd = np.random.permutation(rnd)
        inputs = Variable(torch.from_numpy(x_train[rnd,:]))
        targets = Variable(torch.from_numpy(y_train[rnd]))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print epoch,':',loss.data[0]

    predicted = model(Variable(torch.from_numpy(x_test))).data.numpy()
    plt.hist(predicted[::4])
    plt.hist(predicted[1::4])
    plt.hist(predicted[2::4])
    plt.hist(predicted[3::4])
    plt.show()
#print predicted
#plt.plot(a, y_train, 'ro', label='Original data')
#plt.plot(a, predicted, label='Fitted line')
#plt.legend()
#plt.show()
