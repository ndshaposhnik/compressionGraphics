import numpy as np


def compressedGD(function, compressor, x0=None, max_iter=1000000, tol=1e-2, alpha=0.04):
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
        gradient = function.gradient(x)
        gradients.append(np.linalg.norm(gradient))
        compressedGradient, k = compressor.compress(gradient)
        x += alpha * compressedGradient
        iteration += 1
        coords_cnt += k
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
        "name": compressor.name,
    }
