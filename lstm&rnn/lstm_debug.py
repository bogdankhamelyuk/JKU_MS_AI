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
        return y * f(-self.z) + (1-y) * f(self.z)

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
        return x1,y1 #
         
    else:
        return x2,y2 #



#EXERCISE 3

def backward(self):
    
    delta = np.zeros((T,I))
    self.gradR = 0
    self.gradW = 0
    self.gradV = 0
    dL_da = np.zeros((T,I))
    
    # calculate psi[t]
    # Loss formula was directly taken from forward pass 
    psi = - (np.sign(self.z))/(np.exp(np.abs(self.z))+1) \
          + (np.max([0,self.z])/self.z)*(1-self.y)

    # just before start looping, calculate only one time 
    # dL_da for the last position using psi, since we won't have psi[t]
    dL_da[-1] = psi.reshape(1,1) @ self.V
    
    # calculate delta[t]:
    for t in reversed(range(0, T)):
        
        # 1) calculate dL/da[t]:
        if t < (T-1):
            dL_da[t] += delta[t+1] @ self.R 
        
        # 2) calculate da[t]/ds[t] 
        s_t = self.W @ self.x[t].reshape(D,1)
        if t > 0: # avoid out of range for self.a
            s_t += self.R @ self.a[t-1].reshape(I,1)
        da_ds = np.diag(1/(np.cosh(s_t))**2) 

        delta[t] = dL_da[t].reshape(I,1) @ da_ds 
    
    # calculate gradients 
    for t in range(0, T):
        #calculate self.gradR
        if t!=0:       
            self.gradR += delta[t].reshape(I,1) @ self.a[t-1].reshape(1,I)
        # calculate gradW
        self.gradW += delta[t].reshape(I,1) @ self.x[t].reshape(1,D)
        # calculate self.gradV
        if len(psi)!=1:
            self.gradV += psi[t].reshape(K,K) @ self.a[t].reshape(K,I)
    if len(psi) == 1:
        self.gradV += psi[-1].reshape(K,K) @ self.a[-1].reshape(K,I)
    

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
    
    for row in self.gradV:
        for dV in row:
            diff = np.linalg.norm(gradApprox-dV)/(np.linalg.norm(dV)+np.linalg.norm(gradApprox))
            if diff > self.thresh:
                print("error")
    for row in self.gradW:
        for dW in row:
            diff = np.linalg.norm(-dW+gradApprox)/(np.linalg.norm(dW)+np.linalg.norm(gradApprox))
            if diff > self.thresh:
                print("schmerror")
    for row in self.gradR:
        for dR in row:
            diff = np.linalg.norm(-dR+gradApprox)/(np.linalg.norm(dR)+np.linalg.norm(gradApprox))
            if diff > self.thresh:
                print("herror")
   
    # restore model weights to the default ones
    self.W = self.W + self.eps
    self.V = self.V + self.eps
    self.R = self.R + self.eps




FullyRecurrentNetwork.grad_check = grad_check
model.grad_check(1e-7, 1e-7)