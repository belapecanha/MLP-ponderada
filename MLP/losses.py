import numpy as np


def cross_entropy(Y_pred, Y_true):
    N = Y_true.shape[0]
    # clip protege contra precisão numérica: softmax pode retornar valores próximos de zero e log(0) = -inf quebraria o cálculo    log_p = -np.log(np.clip(Y_pred, 1e-12, 1.0))
    return np.sum(log_p * Y_true) / N
