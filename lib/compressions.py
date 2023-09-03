import numpy as np
from scipy.special import softmax


def getTopKMask(x, k):
    N = x.shape[0]
    xSorted = sorted(list(enumerate(x)), key=lambda elem : abs(elem[1]), reverse=True)
    mask = np.zeros(N)
    for i in range(k):
        mask[xSorted[i][0]] = 1
    return mask


def topK(gradient, k=None):
    if not k:
        k = gradient.shape[0] // 2
    return gradient * getTopKMask(gradient, k)


def uniformCompression(gradient, k=None):
    N = gradient.shape[0]
    if not k:
        k = N // 2
    mask = np.append(np.ones(k), np.zeros(N - k))
    np.random.shuffle(mask)
    return gradient * mask


def compressedGD(function, x0=None, compression=None, k=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.get_initial_x()
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
            gradient = compression(gradient)
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


def stochasticCompression(gradient0, probability, k=None, alpha=0.05):
    gradient = gradient0.copy()
    N = gradient.shape[0]
    if not k:
        k = N // 2
    mask = getTopKMask(gradient * probability, k)
    gradient *= mask
    probability = np.ones(N) - mask
    return gradient, probability


def stochasticCompressedGD(function, compression, x0=None, k=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.get_initial_x()
    d = x.shape[0]
    iteration = 0
    gradients = []
    conv_array = []
    probability = np.ones(d)
    if not k:
        k = d // 2
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient, probability = compression(gradient, probability)
        x = x + alpha * gradient
        iteration += 1
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    print(f'end {name}')
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": list(range(0, iteration * k, k)),
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k": k,
    }
