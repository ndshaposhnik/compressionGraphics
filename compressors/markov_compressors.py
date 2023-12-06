import numpy as np
from scipy.special import softmax
import random

from .base_compressor import BaseCompressor
from .common import getTopKMask


class RandKCompressor(BaseCompressor):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.name = f"RandK, alpha={alpha}"

    def compress(self, tensor):
        mask = np.append(np.ones(self.k, dtype=np.intc), np.zeros(self.dim - self.k, dtype=np.intc))
        np.random.shuffle(mask)
        return (tensor * mask, self.k)


class MarkovRandKCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"Markov RandK, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        inv_mask = np.ones_like(mask) - mask
        sumReduced = np.sum(mask * self.probability * (1 - self.penalty))
        self.probability -= mask * self.probability * (1 - self.penalty)
        self.probability += inv_mask * sumReduced / (self.dim - self.k)
        return tensor * mask, self.k
