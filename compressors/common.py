import numpy as np


def getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d, dtype=np.intc)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


def change_probability_multiplication(probability, mask, penalty):
    assert probability.shape == mask.shape, 'probability and shape are not the same shape'
    n = probability.shape[0]
    k = np.sum(mask)
    assert k > 0, 'empty mask'
    inv_mask = np.ones_like(mask) - mask
    sumReduced = np.sum(mask * probability * penalty)
    probability -= mask * probability * penalty
    probability += inv_mask * sumReduced / (n - k)
    return probability


def change_probability_subtraction(probability, mask, penalty):
    assert probability.shape == mask.shape, 'probability and shape are not the same shape'
    n = probability.shape[0]
    k = np.sum(mask)
    assert k > 0, 'empty mask'
    inv_mask = np.ones_like(mask) - mask
    
    tmp_probability = np.copy(probability)
    tmp_probability -= mask * penalty
    tmp_probability = np.maximum(tmp_probability, 0)

    sumReduced = np.sum(probability - tmp_probability)
    probability = tmp_probability

    probability += inv_mask * sumReduced / (n - k)
    return probability
