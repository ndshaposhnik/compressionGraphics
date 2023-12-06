import numpy as np


def getTopKMask(x, k):
    d = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(d, dtype=np.intc)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask
