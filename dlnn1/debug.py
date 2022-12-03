import numpy as np





def convolution1d(x, k):
    L = x.shape[0]
    R = k.shape[0]
    features = []
    for a in range(L):
         conv_product = []
         for i in range(R):
             h = a + i - 1
             if h >= 0 and h < L:
                 conv_product.append(k[i]*x[h])
         if len(conv_product) == R:
             features.append(sum(conv_product))
    return features
    raise NotImplementedError("TODO: implement convolution1d function!")

rng = np.random.default_rng(1856)
x = rng.standard_normal(11)
k = rng.standard_normal(3)

conv = convolution1d(x, k)