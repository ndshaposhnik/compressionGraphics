import numpy as np
import numpy as np
from scipy.special import softmax

sigma = lambda x : 1 / (1 + np.exp(-x))

def gradf(theta, X, y):
    N, dim = X.shape
    res = np.zeros(dim)
    for i in range(N):
        res += (y[i] - sigma(theta @ X[i].T)) * X[i]
    return res

def calculateLoss(theta, X, y):
    N, dim = X.shape
    res = 0
    for i in range(N):
        res += y[i] * np.log(sigma(theta @ X[i].T)) + (1 - y[i]) * np.log(1 - sigma(theta @ X[i].T))
    return res
