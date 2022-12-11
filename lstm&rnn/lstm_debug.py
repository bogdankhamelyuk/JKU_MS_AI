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
    # Loss formula was directly taken from forward pass   #
    psi = - (np.sign(self.z))/(np.exp(np.abs(self.z))+1) \
          + (np.max([0,self.z])/self.z)*(1-self.y) 
    # derivative of the unstable bce loss    
    # psi = (np.exp(self.z)*(self.y-1)+self.y) / (np.exp(self.z)+1)              
                                    

    # just before start looping, calculate dL_da for the last position once using psi,
    # since we won't have psi[t] anymore
    dL_da[-1] = psi.reshape(1,1) @ self.V

    # calculate delta[t]
    # delta = dL/da * da/ds
    for t in reversed(range(0, T)):

        # complete dL_da in case we've got already delta[t+1], i.e. we aren't at the end     
        if t < (T-1): 
            dL_da[t] += delta[t+1] @ self.R 

        # calculate s_t for diag(f's(t)), i.e. dor da/ds
        s_t = self.W @ self.x[t].reshape(D,1)
        # complete s_t for two cases: t>0 and t==0
        if t > 0:
            s_t += self.R @ self.a[t-1].reshape(I,1)
        else:
            s_t += self.R @ np.zeros_like(self.a[t]).reshape(I,1)
        # insert s_t into formula, so we have da_ds
        da_ds = np.diag(1/(np.cosh(s_t))**2)
        
        # finally, calculate delta
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
    # self._W = self.W + self.eps
    # self._V = self.V + self.eps
    # self._R = self.R + self.eps

    self.W = self.W + self.eps
    self.V = self.V + self.eps
    self.R = self.R + self.eps

    # lol = self._W - self.W
    # kek = self._R - self.R
    # lok = self._V - self.V
    #call forward() to make forward pass
    lossPlusEps = self.forward(self.x,self.y)

    # minus 2eps from to prev added weights, to make it x-e
    self.W = self.W - 2*self.eps
    self.V = self.V - 2*self.eps
    self.R = self.R - 2*self.eps
    # call forward() to make another forward pass
    lossMinusEps = self.forward(self.x,self.y)
    
    # calculate numerical gradient
    gradApprox = ((lossPlusEps - lossMinusEps)/(2*self.eps))[0]
    
    for row in self.gradV:
        for dV in row:
            diff = np.abs(gradApprox - dV) 
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

def update(self, eta):
    self.W -= eta * self.gradW
    self.V -= eta * self.gradV
    self.R -= eta * self.gradR   

FullyRecurrentNetwork.update = update
model.update(0.001)


# ex6
import pandas as pd
np.random.seed(0xDEADBEEF)
D,I,K = 3,32,1

model = FullyRecurrentNetwork(D,I,K)

max_epochs = 100
lr = 0.1

Ts = [1,2,3,4,5,10,15,20]
losses = []


#fig = plt.figure(figsize=(8,3))
for T in Ts:
    T_loss = []
    X = np.random.uniform(-1, 1,(T,D))
    y = np.random.choice([1,0],1,p=[0.5,0.5])[0]
    for epochs in range(max_epochs):
        T_loss.append(model.forward(X,1)[0])
        model.backward()
        model.update(lr)
    plt.plot(T_loss, label=T)
    losses.append(T_loss)    


plt.xlabel("n epochs")
plt.ylabel("Losses")

plt.legend()
plt.axis([0, max_epochs, 0, 0.8])
# function to show the plot
plt.show()
pass