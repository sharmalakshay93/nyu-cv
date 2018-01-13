from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy
from torch.autograd import Variable

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def load_data(filename):
    return torch.load(filename)

def plot2DTorchTensor(data):
    plt.scatter(data[:,0].numpy(), data[:,1].numpy(), alpha=0.7)
    plt.title('2D tensor plot')
    plt.show(block=True)

def center2DTensor(data):
    mean_x = torch.mean(data[:,0])
    mean_y = torch.mean(data[:,1])
    centered = torch.stack([data[:,0]-mean_x, data[:,1]-mean_y], 1)
    plt.scatter(centered[:,0].numpy(), centered[:,1].numpy(), alpha=0.7)
    plt.title('Centered data')
    plt.show(block=True)
    return centered

def decorr(data):
    sigma = np.cov(data.numpy().T)
    w, v = np.linalg.eig(sigma)
    v_trans = v.T
    d = np.linalg.inv(v).dot(sigma).dot(v)
    transformer_matrix = scipy.linalg.fractional_matrix_power(d, -0.5).dot(v_trans)
    decorrelated = transformer_matrix.dot(data.numpy().T)
    decorrelated = decorrelated.T
    plt.scatter(decorrelated[:,0], decorrelated[:,1], alpha=0.7)
    plt.title('Whitened data')
    plt.show(block=True)
    return decorrelated

def whitening_function (filename):
    data = load_data(filename)
    plot2DTorchTensor(data)
    centered = center2DTensor(data)
    decorred = decorr(centered)

# Whitening only got rid of linear dependencies in the data. 
# By plotting the whitened data, we can easily see that the data still 
# has non-linear / higher-order dependencies. Looking at the graph, 
# it appears that the dependency is quadratic in nature.

plt.clf()
whitening_function("./assign0_data.py")

dtype = torch.FloatTensor
x = Variable(torch.arange(-np.pi, np.pi, 0.01).type(dtype), requires_grad=False)
x = x.view(x.size()[0], 1)
y = Variable(torch.cos(x.data).type(dtype), requires_grad=False)
N, D_in, H, D_out = x.size()[0], 1, 10, 1

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

plt.clf()
plt.plot(x.data.numpy(), y.data.numpy(), label="actual")

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)
    
    
    if (t == 0):
        plt.plot(x.data.numpy(), y_pred.data.numpy(), label="initial")
        
    elif (t == 499):
        new_y = y_pred.data.numpy
        plt.plot(x.data.numpy(), y_pred.data.numpy(), label="final")
    

    loss = loss_fn(y_pred, y)
    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

plt.title('Single layer NN with tanh non-linearity')
plt.legend(loc='best')
plt.show(block=True)