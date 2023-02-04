import numpy as np
from nnumpy import Module  # , Flatten
from nnumpy.testing import gradient_check

rng = np.random.default_rng(1856)

def initialiser(fn):
    """ 
    Function decorator for initialisation functions that
    enables initialisation of multiple weight arrays at once. 
    """
    
    def init_wrapper(*parameters, **kwargs):
        for par in parameters:
            par[:] = fn(par.shape, **kwargs)
            #par.zero_grad() - Bogdan: since im using torch tensors, grads are automatically set to
    
    init_wrapper.__name__ = fn.__name__ + "_init"
    init_wrapper.__doc__ = fn.__doc__
    return init_wrapper


## start my solution
import torch



class BatchNormalisation(Module):
    """ NNumpy implementation of batch normalisation. """
    w = [] # experience showed that initaliazing weights as instance variable will make them to zero everytime function compute_grads is called 
           # so instead make them as class variable, to avoid reinitialization 
    def __init__(self, dims: tuple, eps: float = 1e-8):
        """
        Parameters
        ----------
        dims : tuple of ints
            The shape of the incoming signal (without batch dimension).
        eps : float, optional
            Small value for numerical stability.
        """
        super().__init__()
        self.dims = tuple(dims)
        self.eps = float(eps)
        
        self.gamma = self.register_parameter('gamma', np.ones(self.dims))
        self.beta = self.register_parameter('beta', np.zeros(self.dims))
        
        self.running_count = 0
        self.running_stats = np.zeros((2, ) + self.dims)

        rate = 0.5
        self.rate = float(rate)
        self.rng = np.random.default_rng(1856)
        self.q = 1-self.rate # keep probability

    def compute_outputs(self, x):
        if len(BatchNormalisation.w)==0: 
            BatchNormalisation.w = self.rng.normal(size=x.shape)

        dropout_mask = self.rng.binomial(n=1,p=self.q,size=x.shape)

        if self.predicting:
            mean = self.running_stats[0]
            var = self.running_stats[1]
            normalized_input = (x - mean) / np.sqrt(var + self.eps)
            normalized_input = self.gamma * normalized_input + self.beta
            output = normalized_input * BatchNormalisation.w
            return output, BatchNormalisation.w
            # raise NotImplementedError("TODO: implement prediction mode of BatchNormalisation.compute_outputs!")
        else:
            
            mean = np.mean(x, axis=0)
            var = np.var(x,axis=0)
       
            self.running_stats[0] += mean
            self.running_stats[0] /=2

            self.running_stats[1] += var
            self.running_stats[1] /= 2

            normalized_input = (x - mean) / np.sqrt(var + self.eps)
            normalized_input = self.gamma * normalized_input + self.beta

            output = normalized_input*dropout_mask 
            output *= BatchNormalisation.w
            output /= self.q

            return output, BatchNormalisation.w
        
        #raise NotImplementedError("TODO: implement training mode of BatchNormalisation.compute_outputs!")

    def compute_grads(self, grads, cache):
        BatchNormalisation.w -= cache*grads
        return BatchNormalisation.w
        raise NotImplementedError("TODO: implement BatchNormalisation.compute_grads!")

# sanity check
x = rng.uniform(0, 3, size=(7, 3, 5))
batch_norm = BatchNormalisation(x.shape[1:])
s = batch_norm(x)
print(f"mean: {s.mean()}, var: {s.var()}")