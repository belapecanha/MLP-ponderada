def sgd(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[key]
    return params
