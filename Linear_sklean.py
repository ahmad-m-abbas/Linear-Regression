import numpy as np
from sklearn import linear_model


def linear_regression_lib(training_data, training_y, test_data, test_y):
    linear = linear_model.LinearRegression()
    linear.fit(training_data, training_y)
    acc = linear.score(test_data, test_y)

    y = linear.predict(test_data)
    u = (y - test_y) ** 2
    u = np.sum(u)
    v = (test_y - test_y.mean()) ** 2
    v = np.sum(v)
    r2 = 1 - u / v
    linear.coef_
    return linear.coef_, r2, acc
