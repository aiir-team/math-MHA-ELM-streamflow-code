#!/usr/bin/env python
# Created by "Thieu" at 10:02, 06/08/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def round_function(value, decimal=None):
    if decimal is None:
        if 0 < value < 1:
            return round(value, 6)
        else:
            return round(value, 3)
    else:
        return round(value, decimal)

def accuracy(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    return np.equal(y_true, y_pred).mean()

def categorical_accuracy(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return np.equal(y_true, y_pred).mean()

def binary_crossentropy(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred)
    term_1 = y_true * np.log(y_pred)
    return -np.mean(term_0 + term_1, axis=0)

def hinge(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    # replacing 0 = -1
    y_true[y_true == 0] = -1
    y_pred[y_pred == 0] = -1
    return np.mean([max(0, 1 - x * y) for x, y in zip(y_true, y_pred)])

def squared_hinge(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    # replacing 0 = -1
    y_true[y_true == 0] = -1
    y_pred[y_pred == 0] = -1
    return np.mean([max(0, 1 - x * y)**2 for x, y in zip(y_true, y_pred)])

def categorical_hinge(y_true, y_pred):
    # https://github.com/keras-team/keras/blob/v2.9.0/keras/losses.py#L998-L1054
    y_true = np.squeeze(np.array(y_true))       # 0-1 and multi-columns
    y_pred = np.squeeze(np.array(y_pred))       # float and multi-columns

    ## This is for multi-columns (OneHotEncoder - n-nodes in output layer)
    neg = np.max((1 - y_true) * y_pred, axis=1)
    pos = np.sum(y_true * y_pred, axis=1)
    temp = neg - pos + 1
    temp[temp < 0] = 0
    return np.mean(temp)

def categorical_crossentropy(y_true, y_pred):
    # https://github.com/keras-team/keras/blob/v2.9.0/keras/losses.py#L1743-L1788
    y_true = np.squeeze(np.array(y_true))   # 0-1 and multi-labels
    y_pred = np.squeeze(np.array(y_pred))   # float and multi-labels
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    temp = -np.sum(y_true*np.log(y_pred), axis=1)
    return np.mean(temp)

def kl_divergence(y_true, y_pred):
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    y_true = np.clip(y_true, 1e-7, 1 - 1e-7)
    return np.sum(y_true*np.log(y_true/y_pred))


# print(binary_crossentropy(np.array([1, 1, 1]).reshape(-1, 1),
#                          np.array([1, 1, 0]).reshape(-1, 1)))

# y_true = np.array([[0, 1], [0, 0]])
# y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
# print(kl_divergence(y_true, y_pred))


def expand_chebyshev(x, n_funcs):
    x1 = x
    x2 = 2 * np.power(x, 2) - 1
    x3 = 4 * np.power(x, 3) - 3 * x
    x4 = 8 * np.power(x, 4) - 8 * np.power(x, 2) + 1
    x5 = 16 * np.power(x, 5) - 20 * np.power(x, 3) + 5 * x
    my_list = [x1, x2, x3, x4, x5]
    return np.concatenate(my_list[:n_funcs], axis=1)

def expand_legendre(x, n_funcs):
    x1 = x
    x2 = 1 / 2 * (3 * np.power(x, 2) - 1)
    x3 = 1 / 2 * (5 * np.power(x, 3) - 3 * x)
    x4 = 1 / 8 * (35 * np.power(x, 4) - 30 * np.power(x, 2) + 3)
    x5 = 1 / 40 * (9 * np.power(x, 5) - 350 * np.power(x, 3) + 75 * x)
    my_list = [x1, x2, x3, x4, x5]
    return np.concatenate(my_list[:n_funcs], axis=1)


def expand_laguerre(x, n_funcs):
    x1 = -x + 1
    x2 = 1 / 2 * (np.power(x, 2) - 4 * x + 2)
    x3 = 1 / 6 * (-np.power(x, 3) + 9 * np.power(x, 2) - 18 * x + 6)
    x4 = 1 / 24 * (np.power(x, 4) - 16 * np.power(x, 3) + 72 * np.power(x, 2) - 96 * x + 24)
    x5 = 1 / 120 * (-np.power(x, 5) + 25 * np.power(x, 4) - 200 * np.power(x, 3) + 600 * np.power(x, 2) - 600 * x + 120)
    my_list = [x1, x2, x3, x4, x5]
    return np.concatenate(my_list[:n_funcs], axis=1)


def expand_power(x, n_funcs):
    x1 = x
    x2 = x1 + np.power(x, 2)
    x3 = x2 + np.power(x, 3)
    x4 = x3 + np.power(x, 4)
    x5 = x4 + np.power(x, 5)
    my_list = [x1, x2, x3, x4, x5]
    return np.concatenate(my_list[:n_funcs], axis=1)


def expand_trigonometric(x, n_funcs):
    x1 = x
    x2 = np.sin(np.pi * x) + np.cos(np.pi * x)
    x3 = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x)
    x4 = np.sin(3 * np.pi * x) + np.cos(3 * np.pi * x)
    x5 = np.sin(4 * np.pi * x) + np.cos(4 * np.pi * x)
    my_list = [x1, x2, x3, x4, x5]
    return np.concatenate(my_list[:n_funcs], axis=1)



## https://en.wikipedia.org/wiki/Radial_basis_function

def kernel_euclidean(center=np.ndarray, point=np.ndarray, epsilon=1.0):
    return np.linalg.norm(center - point)


def kernel_gaussian(center=np.ndarray, point=np.ndarray, epsilon=2.0):
    r = np.linalg.norm(center - point)
    return np.exp(-(epsilon * r ** 2))


def kernel_multiquadratic(center=np.ndarray, point=np.ndarray, epsilon=1.0):
    r = np.linalg.norm(center - point)
    return np.sqrt(1 + (epsilon * r) ** 2)


def kernel_inverse_quadratic(center=np.ndarray, point=np.ndarray, epsilon=1.0):
    r = np.linalg.norm(center - point)
    return 1.0 / (1.0 + (epsilon * r) ** 2)


def kernel_inverse_multiquadratic(center=np.ndarray, point=np.ndarray, epsilon=1.0):
    r = np.linalg.norm(center - point)
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)


def kernel_bump(center=np.ndarray, point=np.ndarray, epsilon=1.0):
    r = np.linalg.norm(center - point)
    return np.exp(-1 / (1 - (epsilon * r) ** 2)) if r < 1.0 / epsilon else 0


#################### Activation functions ##################################

def relu(x):
    return np.maximum(0, x)


def prelu(x, alpha=0.5):
    return np.where(x < 0, alpha*x, x)


def gelu(x, alpha=0.044715):
    return x/2 * (1 + np.tanh(np.sqrt(2.0/np.pi) * (x + alpha*x**3)))


def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)


def selu(x, alpha=1.67326324, scale=1.05070098):
    return np.where(x < 0, scale*alpha*(np.exp(x) - 1), scale*x)


def rrelu(x, lower=1./8, upper=1./3):
    alpha = np.random.uniform(lower, upper)
    return np.where(x < 0, alpha*x, x)


def tanh(x):
    return np.tanh(x)


def hard_tanh(x, lower=-1., upper=1.):
    return np.where(x < lower, -1, np.where(x > upper, upper, x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hard_sigmoid(x, lower=-2.5, upper=2.5):
    return np.where(x < lower, 0, np.where(x > upper, 1, 0.2*x + 0.5))


def swish(x):
    return x / (1. + np.exp(-x))


def hard_swish(x, lower=-3., upper=3.):
    return np.where(x <= lower, 0, np.where(x >= upper, x, x*(x + 3)/6))


def soft_plus(x, beta=1.0):
    return 1.0/beta * np.log(1 + np.exp(beta * x))


def mish(x, beta=1.0):
    return x * np.tanh(1.0/beta * np.log(1 + np.exp(beta * x)))


def soft_sign(x):
    return x / (1 + np.abs(x))


def tanh_shrink(x):
    return x - np.tanh(x)


def soft_shrink(x, alpha=0.5):
    return np.where(x < -alpha, x + alpha, np.where(x > alpha, x - alpha, 0))


def hard_shrink(x, alpha=0.5):
    return np.where(-alpha < x < alpha, x, 0)












