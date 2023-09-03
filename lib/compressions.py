import numpy as np
from scipy.special import softmax


class Compressor:
    def __init__(compression, k):
        self.compression = compression
        self.k = k
        self.probabilit

    def compress(x):
        return self.compression()



def getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


def topK(gradient, k):
    return gradient * getTopKMask(gradient, k)


def uniformCompression(gradient, k):
    d = gradient.shape[0]
    mask = np.append(np.ones(k), np.zeros(d - k))
    np.random.shuffle(mask)
    return gradient * mask


def compressedGD(function, x0=None, compression=None, k=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    iteration = 0
    gradients = []
    conv_array = []
    if not compression:
        k = x.shape[0]
    if not k:
        k = x.shape[0] // 2
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        if compression:
            gradient = compression(gradient, k=k)
        x = x + alpha * gradient
        iteration += 1
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": list(range(0, iteration * k, k)),
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k": k,
    }


def ban2InARow(gradient0, probability, k):
    gradient = gradient0.copy()
    mask = getTopKMask(gradient * probability, k)
    probability = np.ones_like(mask) - mask
    return gradient * mask, probability


def reduceProbability(gradient0, probability, k, alpha=0.7):
    d = gradient0.shape[0]
    gradient = gradient0.copy()
    mask = getTopKMask(gradient * probability, k)
    inv_mask = np.ones_like(mask) - mask
    probability = softmax(gradient)
    sumReduced = np.sum(mask * probability * (1 - alpha))
    probability -= mask * probability * (1 - alpha)
    probability += inv_mask * sumReduced / (d - k)
    return gradient * mask, probability


def stochasticCompressedGD(function, compression, x0=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    d = x.shape[0]
    iteration = 0
    gradients = []
    conv_array = []
    probability = np.ones(d) / d
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient, probability = compression(gradient, probability, k)
        x = x + alpha * gradient
        iteration += 1
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": list(range(0, iteration * k, k)),
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k": k,
    }
