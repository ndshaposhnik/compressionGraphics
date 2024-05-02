import numpy as np
from scipy.special import softmax
import random

from .base_compressor import BaseCompressor
from . import common


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
        mask = common.getTopKMask(self.probability, self.k)
        self.probability = common.change_probability_multiplication(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class SubtractionPenaltyCompressor(BaseCompressor):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"SubtractionPenalty, alpha={alpha}, penalty={penalty}"

    def compress(self, tensor):
        mask = common.getTopKMask(self.probability, self.k)
        self.probability = common.change_probability_subtraction(self.probability, mask, self.penalty)
        return tensor * mask, self.k


class ExpSmoothingCompressor(BaseCompressor):
    def __init__(self, dim, alpha, beta):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.penalty = np.zeros(self.dim)
        self.beta = beta
        self.name = f"ExpSmoothing, alpha={alpha}, beta={beta}"

    def compress(self, tensor):
        mask = common.getTopKMask(self.penalty, self.k)
        inv_mask = np.ones_like(mask) - mask
        self.penalty = self.beta *self.penalty + (1 - self.beta) * inv_mask
        return tensor * mask, self.k


class BanLastMCompressor(BaseCompressor):
    def __init__(self, dim, alpha, M):
        self.dim = dim
        self.name = f"BanLast {M}, alpha={alpha}"
        self.k = int(self.dim * alpha)
        self.M = M
        self.history = np.zeros(self.dim, dtype=np.intc)

    def _get_mask(self):
        zeros = np.nonzero((self.history == 0))[0]
        assert len(zeros) >= self.k, f'{len(zeros)}, {self.k}'
        np.random.shuffle(zeros)
        indices = zeros[:self.k]
        assert len(indices) == self.k
        result = np.zeros_like(self.history)
        np.put(
            result, indices, np.ones(self.k, dtype=np.intc),
        )
        return result

    def _update_history(self, mask):
        self.history -= np.ones_like(self.history)
        self.history = np.maximum(self.history, np.zeros_like(self.history))
        self.history += mask * self.M

    def compress(self, tensor):
        mask = self._get_mask()
        self._update_history(mask)
        assert len(mask.nonzero()) > 0
        return tensor * mask, self.k

