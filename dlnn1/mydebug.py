import numpy as np

from nnumpy import Module
from nnumpy.utils import sig2col
from nnumpy.testing import gradient_check

rng = np.random.default_rng(1856)
def multi_channel_convolution2d(x, k):
    """
    Compute the multi-channel convolution of multiple samples.
    
    Parameters
    ----------
    x : (N, Ci, A, B)
    k : (Co, Ci, R1, R2)
    
    Returns
    -------
    y : (N, Co, A', B')
    
    See Also
    --------
    sig2col : can be used to convert (N, Ci, A, B) ndarray 
              to (N, Ci, A', B', R1, R2) ndarray.
    """
    N = x.shape[0]
    Co,Ci,R1,R2 = k.shape
    
    x_new = sig2col(x,(R1,R2))


    A_prime,B_prime = x_new.shape[2],x_new.shape[3]
    x_new = x_new.reshape(N,A_prime,B_prime,Ci*R1*R2) # ∈ ℝ^(N x A' x B' x (Cin*R1*R2))

    k = k.reshape(Co,Ci*R1*R2)

    y = np.dot(x_new,k.T)
    y = y.reshape(N,Co,A_prime, B_prime)
    return y
    raise NotImplementedError("TODO: implement multi_channel_convolution2d function!")


class Conv2d(Module):
    """ Numpy DL implementation of a 2D convolutional layer. """
    
    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        
        # register parameters 'w' and 'b' here (mind use_bias!)
        self.w = np.random.randn(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.b = np.random.randn(self.out_channels)
        
        #raise NotImplementedError("TODO: register parameters in Conv2D.__init__!")
        self.reset_parameters()
        
    def reset_parameters(self, seed: int = None):
        """ 
        Reset the parameters to some random values.
        
        Parameters
        ----------
        seed : int, optional
            Seed for random initialisation.
        """
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(size=self.w.shape)
        if self.use_bias:
            self.b = np.zeros_like(self.b)
        
    def compute_outputs(self, x):
        """
        Parameters
        ----------
        x : (N, Ci, H, W) ndarray
        
        Returns
        -------
        feature_maps : (N, Co, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        """
        self.x = x
        cache = x
        output = multi_channel_convolution2d(self.x, self.w)
        feature_maps = [output[:,c] + self.b[c] for c in range(output.shape[1])]
        
        return feature_maps, cache
        raise NotImplementedError("TODO: implement Conv2D.compute_outputs function!")
    
    def compute_grads(self, grads, cache):
        """
        Parameters
        ----------
        grads : (N, Co, H', W') ndarray
        cache : ndarray or tuple of ndarrays
        
        Returns
        -------
        dx : (N, Ci, H, W) ndarray
        """
        x = cache
        paddingH = self.kernel_size[0]-1
        paddingW = self.kernel_size[1]-1
        gradsPad = np.pad(grads,((0,0),(0,0),(paddingH,paddingH),(paddingW,paddingW)))
        dx = multi_channel_convolution2d(gradsPad, np.flip(np.swapaxes(self.w,0,1),(-2,-1)))
        self.w.grad = np.swapaxes(multi_channel_convolution2d(np.swapaxes(x,0,1),np.swapaxes(grads,0,1)),0,1)
        return dx
        raise NotImplementedError("TODO: implement Conv2D.compute_grads function!")
        

conv2d = Conv2d(3, 8, (5, 3))
conv_check = gradient_check(conv2d, rng.standard_normal(size=(15, 3, 13, 13)), debug=True)
print("gradient check for Conv2D:", "passed" if conv_check else "failed")