import numpy as np
from scipy.special import softmax


class Compressor:
    def __init__(self, compression, k):
        self.compression = compression
        self.k = k

    def compress(self, x):
        return self.compression(x, k=self.k)


class AdaptiveCompressor:
    def __init__(self, compression):
        self.compression = compression

    def compress(self, x):
        return self.compression(x)


class SmartCompressor(Compressor):
    def __init__(self, compression, k):
        super().__init__(compression, k)
        
    def compress(self, x, probability):
        return self.compression(x, probability=probability, k=self.k)


def _getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d, dtype=np.intc)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


def topK(gradient, k):
    return gradient * _getTopKMask(gradient, k)


def uniformCompression(gradient, k):
    d = gradient.shape[0]
    mask = np.append(np.ones(k, dtype=np.intc), np.zeros(d - k, dtype=np.intc))
    np.random.shuffle(mask)
    return gradient * mask


def ban2InARow(gradient, probability, k):
    mask = _getTopKMask(gradient * probability, k)
    probability = np.ones_like(mask) - mask
    return gradient * mask, probability


def reduceProbability(gradient, probability, k, penalty=0.5):
    d = gradient.shape[0]
    mask = _getTopKMask(gradient * probability, k)
    inv_mask = np.ones_like(mask, dtype=np.intc) - mask
    probability = softmax(gradient)
    sumReduced = np.sum(mask * probability * (1 - penalty))
    probability -= mask * probability * (1 - penalty)
    probability += inv_mask * sumReduced / (d - k)
    return gradient * mask, probability


def reduceProbabilitySoftMax(gradient0, probability, k, penalty=0.5):
    gradient = gradient0.copy()
    mask = _getTopKMask(gradient * probability, k)
    inv_mask = np.ones_like(mask, dtype=np.intc) - mask
    gradient *= (inv_mask + mask * penalty)
    probability = softmax(gradient)
    return gradient0 * mask, probability


def compressionWithPenalty(gradient, probability, k, dropsTo=0.0, step=0.25):
    mask = _getTopKMask(gradient * probability, k)
    probability += step * np.ones_like(probability)
    probability = np.minimum(probability, np.ones_like(probability))
    inv_mask = np.ones_like(mask, dtype=np.intc) - mask
    probability = inv_mask * probability + dropsTo * mask
    return gradient * mask, probability


def topUnknown(gradient, beta=1):
    d = gradient.shape[0]
    bound = beta * np.linalg.norm(gradient) / np.sqrt(d)
    gradientSorted = sorted(list(enumerate(gradient)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d, dtype=np.intc)
    for i in range(d):
        if abs(gradientSorted[i][1]) < bound:
            break
        mask[gradientSorted[i][0]] = 1
    if np.sum(mask) == 0:
        mask = _getTopKMask(gradient, 1)
    return gradient * mask, np.sum(mask)
