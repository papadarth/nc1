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
t   = 64
a   = np.zeros((n,t),dtype=np.float32) 
## a
a[::4,16] = -1
a[::4,32] = -1
## b
a[1::4,16] = 1
a[1::4,32] = 1
## c
a[2::4,16] = -1
a[2::4,32] = 1
## d
a[3::4,16] = 1
a[3::4,32] = -1

a   += np.random.randn(n,t)*.3
b   = np.zeros((n), dtype=np.int64)
#b   = np.zeros((n), dtype=np.float32)
b[1::4] = 1
b[2::4] = 2
b[3::4] = 3


x_train = a[:800,:]
y_train = b[:800]

x_test  = a[800:,:]
y_test  = a[800:]


#plt.plot(a[:10,:].T);plt.show()

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

class firstCNN(nn.Module):
    def __init__(self, input_size, filter_size1, filter_size2, output_size):
        super(firstCNN, self).__init__()
        self.layer1     = nn.Sequential(
            nn.Conv1d(input_size, filter_size1, kernel_size=4),
            nn.BatchNorm1d(filter_size1),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )
        self.layer2     = nn.Sequential(
            nn.Conv1d(filter_size1, filter_size2, kernel_size=4),
            nn.BatchNorm1d(filter_size2),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )
        self.fc = nn.Linear(filter_size2,4)

    def forward(self, x):
        x   = x.unsqueeze_(1)
        print x.size()

        out = self.layer1(x)
        out = self.layer2(out)
        print out.size()
        out = self.fc(out)
        return out

input_size	= t
output_size	= 4
filter_size1    = 16
filter_size2    = 32
hidden1_size    = 128

### 2 Dense Layer Model
#model = DenseLayers(input_size, hidden1_size, output_size)
### first CNN
model = firstCNN(1, filter_size1, filter_size2, output_size)

criterion	= nn.CrossEntropyLoss()
optimizer	= torch.optim.Adam(model.parameters(), lr=.001)
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
    plt.plot(predicted[::4,:].T,'b.')
    plt.plot(predicted[1::4,:].T,'g+')
    plt.plot(predicted[2::4,:].T,'rx')
    plt.plot(predicted[3::4,:].T,'ko')
    plt.show()
