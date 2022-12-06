import pandas as pd

import common
from Linear_closed_form import linear_regression
from Linear_gradient import linear_regression_gradient_descent
from Linear_sklean import linear_regression_lib
from common import data_to_train, clean_data

data = pd.read_csv('grades.csv')

data = clean_data(data)
normalization = ['normal_z_score', 'normal_min_max', 'normal_scale_data']
for norm in normalization:
    method = getattr(common, norm)
    temp_data = method(data)
    training_data, training_y, test_data, test_y = data_to_train(temp_data)
    print(norm, ": closed_form")
    print(linear_regression(training_data, training_y, test_data, test_y)[0])
    print(norm, ": sklearn")
    print(linear_regression_lib(training_data, training_y, test_data, test_y)[1])
    print(norm, ": gradient_descent")
    print(linear_regression_gradient_descent(training_data, training_y, test_data, test_y, .001, 100)[1])
    print()
