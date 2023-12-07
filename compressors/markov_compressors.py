import numpy as np
from scipy.special import softmax
import random

from .base_compressor import BaseCompressor
from .common import *


class NoneCompressor(BaseCompressor):
    def __init__(self, dim):
        self.dim = dim
        self.name = "Without compression"

    def compress(self, tensor):
        return (tensor.copy(), self.dim)


class RandKCompressor(BaseCompressor):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.name = f"RandK, alpha={alpha}"

    def compress(self, tensor):
        mask = np.append(np.ones(self.k, dtype=np.intc), np.zeros(self.dim - self.k, dtype=np.intc))
        np.random.shuffle(mask)
        return (tensor * mask, self.k)


class MultiplicationPenaltyCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"MultiplicationPenalty, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class SubtractionPenaltyCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"SubtractionPenalty, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = getTopKMask(self.probability, self.k)
        self.probability = change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class ExpSmoothingCompressor(BaseCompressor):
    def __init__(self, dim, alpha, beta):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.penalty = np.zeros(self.dim)
        self.beta = beta
        self.name = f"ExpSmoothing, alpha={alpha}, beta={beta}"

    def compress(self, tensor):
        mask = getTopKMask(self.penalty, self.k)
        inv_mask = np.ones_like(mask) - mask
        self.penalty = self.beta *self.penalty + (1 - self.beta) * inv_mask
        return tensor * mask, self.k
