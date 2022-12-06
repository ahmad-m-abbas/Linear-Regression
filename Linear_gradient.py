import numpy as np


def linear_regression_gradient_descent(training_data, training_y, test_data, test_y, learning_rate, iterations):
    x = np.array(training_data)
    xt = np.transpose(x)
    w = np.zeros(len(xt))
    for i in range(iterations):
        y = np.dot(x, w)
        u = y - training_y
        u = np.dot(xt, u)
        w = w - learning_rate * u
    y = np.dot(test_data, w)
    u = (y - test_y) ** 2
    u = np.sum(u)
    v = (test_y - test_y.mean()) ** 2
    v = np.sum(v)
    r2 = 1 - u / v
    return w, r2
