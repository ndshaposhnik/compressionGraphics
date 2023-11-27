import numpy as np
from scipy.special import softmax

from abc import ABC


def _getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d, dtype=np.intc)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


class Compression: # добавить информацию по компрессору
    def __init__(self, dim):
        pass

    def compress(self, tensor):
        pass


class NoneCompressor(Compression):
    def __init__(self, dim):
        self.dim = dim
        self.name = "Without compression"

    def compress(self, tensor):
        assert tensor.size == self.dim, "tensor size is not equal to compressor dim"
        return (tensor, self.dim)


class TopKCompressor(Compression):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.name = f"TopK Compressor, alpha={self.alpha}"

    def compress(self, tensor):
        return (tensor * _getTopKMask(tensor, self.k), self.k)


class RandKCompressor(Compression):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.name = f"RandK Compressor, alpha={self.alpha}"

    def compress(self, tensor):
        mask = np.append(np.ones(self.k, dtype=np.intc), np.zeros(self.dim - self.k, dtype=np.intc))
        np.random.shuffle(mask)
        return (tensor * mask, self.k)


class ReduceProbabilityCompressor(Compression):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"ReduceProbabilityCompressor, alpha={self.alpha}, penalty={self.penalty}"

    def compress(self, tensor):
        mask = _getTopKMask(tensor * self.probability, self.k)
        inv_mask = np.ones_like(mask) - mask
        self.probability = softmax(tensor)
        sumReduced = np.sum(mask * self.probability * (1 - self.penalty))
        self.probability -= mask * self.probability * (1 - self.penalty)
        self.probability += inv_mask * sumReduced / (self.dim - self.k)
        return tensor * mask, self.k


class NewReduceProbabilityCompressor(Compression):
    def __init__(self, dim, alpha, penalty):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.probability = np.ones(self.dim) / self.dim
        self.penalty = penalty
        self.name = f"newreduceprobabilitycompressor, alpha={self.alpha}, penalty={self.penalty}"

    def compress(self, tensor):
        mask = _getTopKMask(tensor * probability, self.k)
        inv_mask = np.ones_like(mask) - mask
        sumReduced = np.sum(mask * self.probability * (1 - self.penalty))
        probability -= mask * self.probability * (1 - self.penalty)
        probability += inv_mask * sumReduced / (self.dim - self.k)
        return tensor * mask, self.k


class TopUnknownCompressor(Compression):
    def __init__(self, dim, beta):
        self.dim = dim
        self.beta = beta
        self.name = f"TopUnknownCompressor, beta={self.beta}"

    def compress(self, tensor):
        bound = self.beta * np.linalg.norm(tensor) / np.sqrt(self.dim)
        mask = tensor >= bound
        return tensor * mask, np.sum(m)


class PenaltyCompressor(Compression):
    def __init__(self, dim, alpha, dropsTo, step):
        self.dim = dim
        self.k = int(self.dim * alpha)
        self.dropsTo = dropsTo
        self.step = step
        self.penalty = torch.full((self.dim,), 1 / self.dim, dtype=torch.float, device='cuda:0')
        self.name = f"PenaltyCompressor, alpha={self.alpha}, dropsTo={self.dropsTo}, step={self.step}"

    def compress(self, tensor):
        mask = _getTopKMask(tensor * self.penalty, self.k)
        self.penalty += self.step * torch.ones_like(self.penalty)
        self.penalty = np.minimum(self.penalty, torch.ones_like(self.penalty))
        inv_mask = torch.ones_like(mask) - mask
        self.penalty = inv_mask * self.penalty + self.dropsTo * mask
        return tensor * mask, self.k

