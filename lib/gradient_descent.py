import numpy as np
from tqdm import tqdm


def compressedGD(function, compressor, x0=None, max_iter=1000000, tol=1e-2, alpha=0.04):
    if x0:
        x = x0.copy()
    else:
        x = function.getInitialX()
    iteration = 0
    loss = []
    conv_array = []
    coords = []
    coords_cnt = 0
    for _ in tqdm(range(max_iter)):
        conv_array.append(x)
        gradient = function.gradient(x)
        loss.append(function.loss(x))
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
        "loss": loss,
        "coords": coords,
        "tol": np.linalg.norm(gradient),
        "conv_array": conv_array,
        "name": compressor.name,
    }
