import numpy as np
import matplotlib.pyplot as plt

def gauss_function(x, mu, sigma):
    """ This is the 1D gaussian probability density function with mean mu and standard deviation sigma """
    
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def generate_data(mu, sigma, samples):
    """ This funtion generates the data """
    
    x = np.linspace(-15,25,samples)
    
    # get density over x
    density = gauss_function(x, mu, sigma)
    
    # sample for the gaussian distribution
    X_samples = sigma*np.random.randn(samples) + mu
    
    return x, density, X_samples

# here we generate 3 datasets with 500 samples each with same mu = 5 but different sigmas = 1,3,10

n = 500
mu = 5.0
sigma_1 = 1.0
sigma_2 = 3.0
sigma_3 = 10.0

x, p1, X1 = generate_data(mu, sigma_1, samples=n)
_, p2, X2 = generate_data(mu, sigma_2, samples=n)
_, p3, X3 = generate_data(mu, sigma_3, samples=n)

# Calculate the log-likehood function; it should return a scalar value

def log_likelihood(data, mu, sigma):
    """ Calculates the log likelihood"""
    
    ########## YOUR SOLUTION HERE ##########
    print(-n/2*np.log(2*np.pi))
    lnL = -n/2*np.log(2*np.pi) - n/2*np.log(sigma**2) 
    print(data.shape)
    for xi in data:
        print("xi: ",xi)
        print("sigma: ",sigma)
        lnL+= -1/(2*sigma**2)*((xi - mu)**2)
        print(lnL.shape)
    return lnL

print("ln(L)(X1,mu=5,sigma1) = %8.1f" % log_likelihood(X1, mu=5, sigma=sigma_1))
print("ln(L)(X1,mu=5,sigma2) = %8.1f" % log_likelihood(X1, mu=5, sigma=sigma_2))
print("ln(L)(X1,mu=5,sigma3) = %8.1f" % log_likelihood(X1, mu=5, sigma=sigma_3))
print("\n")
print("ln(L)(X2,mu=5,sigma1) = %8.1f" % log_likelihood(X2, mu=5, sigma=sigma_1))
print("ln(L)(X2,mu=5,sigma2) = %8.1f" % log_likelihood(X2, mu=5, sigma=sigma_2))
print("ln(L)(X2,mu=5,sigma3) = %8.1f" % log_likelihood(X2, mu=5, sigma=sigma_3))
print("\n")
print("ln(L)(X3,mu=5,sigma1) = %8.1f" % log_likelihood(X3, mu=5, sigma=sigma_1))
print("ln(L)(X3,mu=5,sigma2) = %8.1f" % log_likelihood(X3, mu=5, sigma=sigma_2))
print("ln(L)(X3,mu=5,sigma3) = %8.1f" % log_likelihood(X3, mu=5, sigma=sigma_3))