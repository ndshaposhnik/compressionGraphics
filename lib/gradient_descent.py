import numpy as np


def compressedGD(function, x0=None, compressor=None, name="", max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    iteration = 0
    gradients = []
    conv_array = []
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        if compressor:
            gradient = compressor.compress(gradient)
        x = x + alpha * gradient
        iteration += 1
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    k = compressor.k if compressor else x.shape[0]
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": list(range(0, iteration * k, k)),
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k": k,
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
    history = np.ones(d) / d
    while True:
        conv_array.append(x)
        gradient0 = function.gradient(x)
        gradients.append(np.linalg.norm(gradient0))
        gradient = gradient0.copy()
        gradient, history = compressor.compress(gradient, history)
        x = x + alpha * gradient
        iteration += 1
        if np.linalg.norm(gradient0) < tol:
            break
        if iteration >= max_iter:
            break
    return {
        "num_iter": iteration,
        "gradients": gradients,
        "coords": list(range(0, iteration * compressor.k, compressor.k)),
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": name,
        "k": compressor.k,
    }
