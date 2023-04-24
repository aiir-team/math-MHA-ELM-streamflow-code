# !/usr/bin/env python
# Created by "Thieu" at 15:35, 24/02/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import keras.backend as K


def MSE(y_true, y_pred):
    return K.mean(K.pow(y_true - y_pred, 2))

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def MAE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def ME(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))

def accuracy(y_true, y_pred):
    res = y_pred.numpy()
    res = np.ceil(res)
    res[res == 0] = -1
    y_pred = K.constant(res)
    return K.mean(K.equal(y_true, y_pred))
