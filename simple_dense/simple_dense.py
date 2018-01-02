#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simple dense classifier
	one layer simple dense 
	prediting two classes for a simple time series model; one POI, no temporal dependencies
	test result (MSEloss) and activation of the layer (weight) are plottet
"""

__author__  = "Tobias Wagner"
__version__ = '0.2'

import numpy as np
import pylab as plt
import torch
import torch.nn as nn
from torch.autograd import Variable



## generate data ##
n   = 400
t   = 10
a   = np.zeros((n,t),dtype=np.float32) 
a[::1,5] = -1
a[1::2,5] = 1
c = np.copy(a)
a   += np.random.randn(n,t)*.5
c   += np.random.randn(n,t)*.5
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

	predicted = model(Variable(torch.from_numpy(c))).data.numpy()
	plt.figure(figsize=(8,4))
	plt.subplot(121)
	plt.hist(predicted[::2])
	plt.hist(predicted[1::2])
	plt.subplot(122)
	plt.plot(model.linear.weight[0].data.numpy())
	plt.vlines([5],0,.3,'r')
	plt.savefig('prediction_e%02d'%(i))
	plt.show()
