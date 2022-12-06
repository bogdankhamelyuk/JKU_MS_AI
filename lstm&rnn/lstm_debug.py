# %matplotlib inline
import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt

class FullyRecurrentNetwork(object):
    def __init__(self, D, I, K):
        self.W = np.random.uniform(-0.01, 0.01, (I, D))
        self.R = np.random.uniform(-0.01, 0.01, (I, I))
        self.V = np.random.uniform(-0.01, 0.01, (K, I))
    
    def forward(self, x, y):
        # helper function for numerically stable loss
        def f(z):
            return np.log1p(np.exp(-np.absolute(z))) + np.maximum(0, z)
        
        # infer dims
        T, D = x.shape
        K, I = self.V.shape

        # init result arrays
        self.x = x
        self.y = y
        self.a = np.zeros((T, I))

        # iterate forward in time 
        # trick: access model.a[-1] in first iteration
        for t in range(T):
            self.a[t] = np.tanh(self.W @ x[t] + self.R @ self.a[t-1])
            
        self.z = model.V @ self.a[t] 
        loss = y * f(-self.z) + (1-y) * f(self.z)
        return loss

T, D, I, K = 10, 3, 5, 1
model = FullyRecurrentNetwork(D, I, K)
model.forward(np.random.uniform(-1, 1, (T, D)), 1)

import random 

def gen(T):
    mu, sigma = 0, 0.2
    for i in range(0,T):
        yield np.random.normal(mu, sigma)


def generate_data(T):
    ########## YOUR SOLUTION HERE ##########
    
    x1 = np.fromiter(gen(T),dtype=float)
    x2 = np.fromiter(gen(T),dtype=float)
    
    x1 = np.reshape(x1,newshape=(T,1))
    x2 = np.reshape(x2,newshape=(T,1))

    x1[0][0] = 1
    y1 = np.array([1.0],dtype=float)

    x2[0][0] = -1.0
    y2 = np.array([0.0],dtype=float)

    choice = np.random.choice([1,-1],1,p=[0.5,0.5])
    if choice[0] == 1: 
        return x1,y1# ,"positive"
         
    else:
        return x2,y2# ,"negative"

np.random.seed(0xDEADBEEF)
# comment this out and upper last categories to check what is distribution of "negative" and "positive" is

# from collections import Counter
# Counter(generate_data(5)[2] for i in range(10000))

x,y = generate_data(5)
print("x shape: ",x.shape)
print("x[0]:",x[0]," y[0]:",y[0])
print("y shape: ",y.shape)

#EXERCISE 3

def backward(self):
    
    T = len(self.z)
    psi = np.zeros(T,)
    delta = np.zeros(T,)
    gradR = 0
    gradW = 0
    gradV = 0

    for t in reversed(range(0, T)):
        # calculate psi[t]
        frac = - self.z[t]/( np.abs(self.z[t]) * (1+np.exp(self.z[t])))
        if self.z[t] > 0:
            frac += (1 - self.y)
        psi[t] = frac

        # calculate s[t] 
        s_t = self.W @ self.x[t]
        if t > 0: # avoid out of range for self.a
            s_t += self.R @ self.a[t-1]

        # calculate delta[t] 
        dL_da = self.V.T @ [psi[t]]
        da_ds = 1/(np.cosh(s_t))**2
        if t != (len(self.z) - 1): # in case we aren't at the beginng and already have t+1 step ahead
            dL_da += delta[t+1] # so we can add "previous" delta, since we're going from T-1 to 0 
        delta[t] = dL_da @ da_ds
    
    # calculate gradients 
    for t in range(0, T):
        #calculate gradR
        if t!=0:       
            gradR += delta[t] * self.a[t-1]
        # calculate gradW
        gradW += delta[t] * self.x[t]
        # calculate gradV
        gradV += psi[t] * self.a[t]
    

FullyRecurrentNetwork.backward = backward
model.backward()

