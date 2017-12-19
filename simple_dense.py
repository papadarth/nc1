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
n   = 200
t   = 10
a   = np.zeros((n,t),dtype=np.float32) 
a[::1,5] = -1
a[1::2,5] = 1
a   += np.random.randn(n,t)*.5
b   = np.zeros((n), dtype=np.float32)
b[1::2] = 1

#plt.plot(a[:10,:].T);plt.show()
# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  
    
    def forward(self, x):
        out = self.linear(x)
	return out

input_size	= t
output_size	= 1

model = LinearRegression(input_size, output_size)

criterion	= nn.MSELoss()
optimizer	= torch.optim.SGD(model.parameters(), lr=.001)

for i in range(10):
    print i
    for epoch in range(100):
        inputs = Variable(torch.from_numpy(a))
        targets = Variable(torch.from_numpy(b))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print epoch,':',loss.data[0]

    predicted = model(Variable(torch.from_numpy(a))).data.numpy()
    plt.hist(predicted[::2])
    plt.hist(predicted[1::2])
    plt.show()
#print predicted
#plt.plot(a, y_train, 'ro', label='Original data')
#plt.plot(a, predicted, label='Fitted line')
#plt.legend()
#plt.show()
