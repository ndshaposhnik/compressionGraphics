import numpy as np
from scipy.special import softmax
import random

from .base_compressor import BaseCompressor
from .common import getTopKMask


class NoneCompressor(BaseCompressor):
    def __init__(self, dim):
        self.dim = dim
        self.name = "Without compression"

    def compress(self, tensor):
        return (tensor.copy(), self.dim)


class TopKCompressor(BaseCompressor):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.name = f"TopK, alpha={alpha}"

    def compress(self, tensor):
        return (tensor * getTopKMask(tensor, self.k), self.k)


class ReduceProbabilityCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"ReduceProbability, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = getTopKMask(tensor * self.probability, self.k)
        self.probability = softmax(tensor) # difference with NewReduceProbability
        self.probability = change_probability(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class NewReduceProbabilityCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"NewReduceProbability, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = getTopKMask(tensor * self.probability, self.k)
        self.probability = change_probability(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class TopUnknownCompressor(BaseCompressor):
    def __init__(self, dim, beta):
        self.dim = dim
        self.beta = beta
        self.name = f"TopUnknown, beta={beta}"

    def compress(self, tensor):
        bound = self.beta * np.linalg.norm(tensor) / np.sqrt(self.dim)
        mask = tensor >= bound
        if np.sum(mask) == 0:
            mask = getTopKMask(tensor, 1)
        return tensor * mask, np.sum(mask)


class PenaltyCompressor(BaseCompressor):
    def __init__(self, dim, alpha, dropsTo, step):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.dropsTo = dropsTo
        self.step = step
        self.penalty = np.ones(self.dim) / self.dim
        self.name = f"Penalty, alpha={alpha}, dropsTo={dropsTo}, step={step}"

    def compress(self, tensor):
        mask = getTopKMask(tensor * self.penalty, self.k)
        self.penalty += self.step * np.ones_like(self.penalty)
        self.penalty = np.minimum(self.penalty, np.ones_like(self.penalty))
        inv_mask = np.ones_like(mask) - mask
        self.penalty = inv_mask * self.penalty + self.dropsTo * mask
        return tensor * mask, self.k


class MarinaCompressor(BaseCompressor):
    def __init__(self, dim, p, compressor):
        self.dim = dim
        self.p = p
        self.compressor = compressor
        self.prevG = None
        self.prevNabla = None
        self.name = f'Marina, p={p}, compressor={compressor.name}'

    def compress(self, nabla):
        c = np.random.binomial(size=1, n=1, p=self.p)[0]
        if c == 1 or self.prevG is None:
            self.prevG = nabla
            self.prevNabla = nabla
            return nabla, self.dim
        result, k = self.compressor.compress(nabla - self.prevNabla)
        result += self.prevG
        self.prevNabla = nabla
        self.prevG = result
        return result, k


class RandomizedKCompressor(BaseCompressor):
    def __init__(self, dim, compressor, minAlpha, maxAlpha):
        self.dim = dim
        self.compressor = compressor
        self.minK = int(dim * minAlpha)
        self.maxK = int(dim * maxAlpha)
        assert self.minK <= self.maxK
        self.name = f'RandomizedKCompressor, ({minAlpha}, {maxAlpha})'

    def getK(self):
        return random.randint(self.minK, self.maxK)

    def compress(self, tensor):
        self.compressor.k = self.getK()
        return self.compressor.compress(tensor)
