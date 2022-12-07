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
            self.W @ x[t]
            self.R @ self.a[t-1]
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

# x,y = generate_data(5)
# print("x shape: ",x.shape)
# print("x[0]:",x[0]," y[0]:",y[0])
# print("y shape: ",y.shape)

#EXERCISE 3

def backward(self):
    
    T = len(self.x)
    psi = np.zeros(len(self.z),)
    delta = np.zeros(T,)
    self.gradR = 0
    self.gradW = 0
    self.gradV = 0

    # calculate psi[t]
    frac = - self.z[-1]/( np.abs(self.z[-1]) * (1+np.exp(self.z[-1])) )
    if self.z[-1] > 0:
        frac += (1 - self.y)
    psi[-1] = frac

    for t in reversed(range(0, T)):
        # calculate s[t] 
        s_t = self.W @ self.x[t]
        if t > 0: # avoid out of range for self.a
            s_t += self.R @ self.a[t-1]

        # calculate delta[t]
        if len(psi)!=1: 
            dL_da = self.V.T @ [psi[t]]
        if len(psi)==1 and t == (T-1):
            dL_da = self.V.T @ [psi[-1]]
        if t < (T - 1): # in case we aren't at the beginng and already have t+1 step ahead
            self.R[:,:]*=delta[t+1]
        da_ds = 1/(np.cosh(s_t))**2
        
        delta[t] = dL_da @ da_ds
    
    # calculate gradients 
    for t in range(0, T):
        #calculate self.gradR
        if t!=0:       
            self.gradR += delta[t] * self.a[t-1]
        # calculate gradW
        self.gradW += delta[t] * self.x[t]
        # calculate self.gradV
        if len(psi)!=1:
            self.gradV += psi[t] * self.a[t]
    if len(psi) == 1:
        self.gradV += psi[-1]*self.a[-1]
    

FullyRecurrentNetwork.backward = backward
model.backward()

#EXERCISE 4
def grad_check(self, eps, thresh):
    ########## YOUR SOLUTION HERE ##########
    self.eps = eps
    self.thresh = thresh

    # add eps to all weights
    self.W = self.W + self.eps
    self.V = self.V + self.eps
    self.R = self.R + self.eps
    #call forward() to make forward pass
    lossPlusEps = self.forward(self.x,self.y)

    # minus 2eps from to prev added weights, to make it x-e
    self.W = self.W - 2*self.eps
    self.V = self.V - 2*self.eps
    self.R = self.R - 2*self.eps
    # call forward() to make another forward pass
    lossMinusEps = self.forward(self.x,self.y)

    # calculate numerical gradient
    gradApprox = (lossPlusEps - lossMinusEps)/(2*self.eps)
    
    for dV in self.gradV:
        diff = np.linalg.norm(dV-gradApprox)/(np.linalg.norm(dV)+np.linalg.norm(gradApprox))
        if diff > self.thresh:
            print("error")
    for dW in self.gradW:
        diff = np.linalg.norm(dW-gradApprox)/(np.linalg.norm(dW)+np.linalg.norm(gradApprox))
        if diff > self.thresh:
            print("error")
    for dR in self.gradR:
        diff = np.linalg.norm(dR-gradApprox)/(np.linalg.norm(dR)+np.linalg.norm(gradApprox))
        if diff > self.thresh:
            print("error")
   
    
    # restore model weights to the default ones
    self.W = self.W + self.eps
    self.V = self.V + self.eps
    self.R = self.R + self.eps




FullyRecurrentNetwork.grad_check = grad_check
model.grad_check(1e-7, 1e-7)