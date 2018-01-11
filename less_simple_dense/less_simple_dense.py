#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
learning pytorch
	testing less simple / 2 layer network

updates:
	0.3 : the pytorch structures Dataset and DataLoader is used
"""

__author__  = "Tobias Wagner"
__version__ = 0.3

import numpy as np
import pylab as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
## generate data ##
class MyDataGenerator(Dataset):
    """
    testing my first pytorch-Dataset implementing an
        data set generator at start-up
    """
    def __init__(self,train=True):
        """
        function where intial logic happens like reading, assiging parameter, generating a set of data
        """
        if train == True: 
            n   = 1000
        else:
            n = 200
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
        self.data   = a
        self.target = b
        """
        x_train = a[:800,:]
        y_train = b[:800]

        x_test  = a[800:,:]
        y_test  = a[800:]
        """
    def __getitem__(self,index):
        """
        retrun a tuple of data set and label of data set
            index is one or more realizations?
        """
        return (self.data[index,:],self.target[index])

    def __len__(self):
        """
        functions returns count of given realizations
        """
        return self.target.shape[0]

#dataset = MyDataGenerator
train_loader = DataLoader(MyDataGenerator(train=True),shuffle=True,batch_size=800)
test_loader = DataLoader(MyDataGenerator(train=False),batch_size=200)

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
t = 20
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
    model.train() 	## model.train() and model.test()
			## By default all the modules are initialized to train mode (self.training = True). 
			## Some layers have different behavior during train or evaluation 
			## (like BatchNorm, Dropout) setting.
			## As a rule of thumb, try to explicitly state your intent 
			## and set model.train() and model.eval() when necessary.
    for epoch in range(100):
		for (inputs, targets) in train_loader:
				inputs = Variable(inputs)
				targets = Variable(targets)
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()
        
		print epoch,':',loss.data[0]
    model.eval() ## for explanation, please look at model.train()
    for test_data,targets in test_loader:
			predicted = model(Variable(test_data)).data.numpy()
			plt.hist(predicted[::4])
			plt.hist(predicted[1::4])
			plt.hist(predicted[2::4])
			plt.hist(predicted[3::4])
			plt.show()
