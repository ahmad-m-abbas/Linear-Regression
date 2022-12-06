import numpy as np


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def linear_regression(training_data, training_y, test_data, test_y):
    x = np.array(training_data)
    xt = np.transpose(x)
    xtx = np.dot(xt, x)
    xtx = xtx + np.identity(len(xtx)) * 0.0001

    while not is_invertible(xtx):
        xtx = xtx + np.identity(len(xtx)) * 0.0001
    w = np.linalg.inv(np.dot(xt, x))
    w = np.dot(w, xt)
    w = np.dot(w, np.array(training_y))
    y = np.dot(test_data, w)
    u = (y - test_y) ** 2
    u = np.sum(u)
    v = (test_y - test_y.mean()) ** 2
    v = np.sum(v)
    r2 = 1 - u / v
    return w, r2
