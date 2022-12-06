import math
import random

from matplotlib import pyplot as plt


def normal_z_score(data):
    for parameter in data.columns:
        if parameter != 'Final':
            data[parameter] = (data[parameter] - data[parameter].mean()) / data[parameter].std()
    return data


def normal_min_max(data):
    for parameter in data.columns:
        if parameter != 'Final':
            data[parameter] = (data[parameter] - data[parameter].min()) / (data[parameter].max() - data[parameter]
                                                                           .min())

    return data


def normal_scale_data(data):
    for parameter in data.columns:
        if parameter != 'Final':
            data[parameter] = data[parameter] / pow(10, int(math.log10(max(data[parameter]))))
    return data


def clean_data(data):
    for parameter in data.columns:
        summation = 0
        size = 0
        for number in range(0, len(data[parameter])):
            if data[parameter][number] != 0:
                summation += data[parameter][number]
                size += 1
        mean = summation / size
        for number in range(0, len(data[parameter])):
            if data[parameter][number] == 0:
                data[parameter][number] = mean
    return data


def data_to_train(data):
    headers = data.columns
    used = []
    for parameter in headers:
        if data[parameter].corr(data['Final']) > 0.75:
            if 'Final' not in used and parameter != 'Final':
                used.append(parameter)

    data['constant'] = 1
    used.append('constant')

    training = data.iloc[random.sample(range(0, len(data)), int(len(data) * 0.8))]
    testing = data.drop(training.index)
    training = training.reset_index(drop=True)
    testing = testing.reset_index(drop=True)

    training_data = training[used]
    training_y = training['Final']
    test_data = testing[used]
    test_y = testing['Final']
    return training_data, training_y, test_data, test_y
