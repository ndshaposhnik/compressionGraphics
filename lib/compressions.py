import numpy as np
from scipy.special import softmax


class Compressor:
    def __init__(self, compression, k):
        self.compression = compression
        self.k = k

    def compress(self, x):
        return self.compression(x, k=self.k)


class SmartCompressor(Compressor):
    def __init__(self, compression, k):
        super().__init__(compression, k)
        
    def compress(self, x, probability):
        return self.compression(x, probability=probability, k=self.k)


def _getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


def topK(gradient, k):
    return gradient * _getTopKMask(gradient, k)


def uniformCompression(gradient, k):
    d = gradient.shape[0]
    mask = np.append(np.ones(k), np.zeros(d - k))
    np.random.shuffle(mask)
    return gradient * mask


def ban2InARow(gradient, probability, k):
    mask = _getTopKMask(gradient * probability, k)
    probability = np.ones_like(mask) - mask
    return gradient * mask, probability


def reduceProbability(gradient, probability, k, alpha=0.5):
    d = gradient.shape[0]
    mask = _getTopKMask(gradient * probability, k)
    inv_mask = np.ones_like(mask) - mask
    probability = softmax(gradient)
    sumReduced = np.sum(mask * probability * (1 - alpha))
    probability -= mask * probability * (1 - alpha)
    probability += inv_mask * sumReduced / (d - k)
    return gradient * mask, probability


def reduceProbabilitySoftMax(gradient0, probability, k, alpha=0.5):
    gradient = gradient0.copy()
    mask = _getTopKMask(gradient * probability, k)
    inv_mask = np.ones_like(mask) - mask
    gradient *= (inv_mask + mask * alpha)
    probability = softmax(gradient)
    return gradient0 * mask, probability


def compressionWithPenalty(gradient0, penalty, k, alpha=0.01):
    gradient = gradient0.copy()
    mask = _getTopKMask(gradient * penalty, k)

