import numpy as np


def cross_entropy(Y_pred, Y_true):
    N = Y_true.shape[0]
    log_p = -np.log(np.clip(Y_pred, 1e-12, 1.0))  
    return np.sum(log_p * Y_true) / N
