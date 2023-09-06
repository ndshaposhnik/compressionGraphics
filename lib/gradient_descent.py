import numpy as np


def compressedGD(function, compressor, x0=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    iteration = 0
    gradients = []
    conv_array = []
    coords = []
    coords_cnt = 0
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient = compressor.compress(gradient)
        x = x + alpha * gradient
        iteration += 1
        coords_cnt += compressor.k
        coords.append(coords_cnt)
        if np.linalg.norm(gradient) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": coords,
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
    }


def stochasticCompressedGD(function, compressor, x0=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    d = x.shape[0]
    iteration = 0
    gradients = []
    conv_array = []
    coords = []
    coords_cnt = 0
    probability = np.ones(d) / d
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient, probability = compressor.compress(gradient, probability)
        x = x + alpha * gradient
        iteration += 1
        coords_cnt += compressor.k
        coords.append(coords_cnt)
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": coords,
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
    }


def adaptiveCompressedGD(function, compressor, x0=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    d = x.shape[0]
    iteration = 0
    gradients = []
    conv_array = []
    coords = []
    coords_cnt = 0
    k_distribution = {}
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient, k = compressor.compress(gradient)
        assert k != 0, "k = 0, aborting"
        x = x + alpha * gradient
        iteration += 1
        coords_cnt += k
        k_distribution[k] = k_distribution.get(k, 0) + 1
        coords.append(coords_cnt)
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": coords,
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k_distribution": k_distribution,
    }
