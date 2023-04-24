# !/usr/bin/env python
# Created by "Thieu" at 18:20, 08/12/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from scipy.special import boxcox, inv_boxcox
import numpy as np
from config import Config
from utils.io_util import load_dataset


class MiniBatch:
    def __init__(self, X_train, y_train, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size

    def random_mini_batches(self, seed_number=None):
        X, Y = self.X_train.T, self.y_train.T
        mini_batch_size = self.batch_size

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.seed(seed_number)

        # Step 1: Shuffle (X, Y)
        permu = list(np.permutation(m))
        shuffled_X = X[:, permu]
        shuffled_Y = Y[:, permu].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(np.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


class DataScaler:
    """
    For feature scaler
    """
    def __init__(self, scale_type="std"):
        self.scale_type = scale_type

    def fit(self, data):
        self.data_mean, self.data_std = data.mean(axis=0), data.std(axis=0)
        self.data_min, self.data_max = data.min(axis=0), data.max(axis=0)
        return self

    def transform(self, data):
        if self.scale_type == "std":
            data_new = (data - self.data_mean) / self.data_std
        elif self.scale_type == "minmax":
            data_new = (data - self.data_min) / (self.data_max - self.data_min)
        elif self.scale_type == "loge":
            data_new = np.log(data)
        elif self.scale_type == "kurtosis":
            data_new = np.sign(data - self.data_mean) * np.power(np.abs(data - self.data_mean), 1.0/3)
        elif self.scale_type == "boxcox":
            data_new, self.lamda_boxcox = boxcox(data.flatten())
            data_new = data_new.reshape(data.shape)

        elif self.scale_type == "kurtosis_std":
            self.data_kurtosis = np.sign(data - self.data_mean) * np.power(np.abs(data - self.data_mean), 1.0 / 3)
            self.data_kur_mean, self.data_kur_std = self.data_kurtosis.mean(axis=0), self.data_kurtosis.std(axis=0)
            data_new = (self.data_kurtosis - self.data_kur_mean) / self.data_kur_std
        elif self.scale_type == "boxcox_std":
            self.data_boxcox, self.lamda_boxcox = boxcox(data.flatten())
            self.data_boxcox = self.data_boxcox.reshape(data.shape)
            self.data_boxcox_mean, self.data_boxcox_std = self.data_boxcox.mean(axis=0), self.data_boxcox.std(axis=0)
            data_new = (self.data_boxcox - self.data_boxcox_mean) / self.data_boxcox_std
        return data_new

    def inverse_transform(self, data):
        if self.scale_type == "std":
            return self.data_std * data + self.data_mean
        elif self.scale_type == "minmax":
            return data * (self.data_max - self.data_min) + self.data_min
        elif self.scale_type == "loge":
            return np.exp(data)
        elif self.scale_type == "kurtosis":
            return np.power(data, 3) + self.data_mean
        elif self.scale_type == "boxcox":
            return inv_boxcox(data, self.lamda_boxcox)

        elif self.scale_type == "kurtosis_std":
            return np.power(self.data_kur_std * data + self.data_kur_mean, 3) + self.data_mean
        elif self.scale_type == "boxcox_std":
            return inv_boxcox(self.data_boxcox_std * data + self.data_boxcox_mean, self.lamda_boxcox)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class ObjectiveScaler:
    """
    For label scaler in classification (binary and multiple classification)
    """
    def __init__(self, obj_name="sigmoid", ohe_scaler=None):
        """
        ohe_scaler: Need to be an instance of One-Hot-Encoder for softmax scaler (multiple classification problem)
        """
        self.obj_name = obj_name
        self.ohe_scaler = ohe_scaler

    def transform(self, data):
        if self.obj_name == "sigmoid":
            return data
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data[np.where(data == 0)] = -1
            return data
        elif self.obj_name == "softmax":
            data = self.ohe_scaler.transform(np.reshape(data, (-1, 1)))
            return data

    def inverse_transform(self, data):
        if self.obj_name == "sigmoid":
            data = np.squeeze(np.array(data))
            data = np.rint(data).astype(int)
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data = np.ceil(data).astype(int)
            data[np.where(data == -1)] = 0
        elif self.obj_name == "softmax":
            data = np.squeeze(np.array(data))
            data = np.argmax(data, axis=1)
        return data


def get_single_file_data(path_read, name_file, name_input_x, name_output_y, validation, test_size, valid_size, seed):
    x_valid, y_valid = None, None
    X_data, Y_data = load_dataset(path_read, name_file, name_input_x, name_output_y)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size, random_state=seed)
    if validation:
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=seed)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_multiple_file_data(path_read, name_files, name_input_x, name_output_y, validation):
    x_valid, y_valid = None, None
    x_train, y_train = load_dataset(path_read, name_files[0], name_input_x, name_output_y)
    x_test, y_test = load_dataset(path_read, name_files[1], name_input_x, name_output_y)
    if validation:
        x_valid, y_valid = load_dataset(path_read, name_files[2], name_input_x, name_output_y)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_dataset_normal(data, validation=False, scale_type="minmax", save_original=False):
    X_valid, Y_valid = None, None
    x_train, y_train, x_valid, y_valid, x_test, y_test = data
    feature_scaler = DataScaler(scale_type=scale_type).fit(x_train)
    X_train, X_test = feature_scaler.transform(x_train), feature_scaler.transform(x_test)
    if validation:
        X_valid = feature_scaler.transform(x_valid)

    label_scaler = DataScaler(scale_type=scale_type).fit(y_train)
    Y_train, Y_test = label_scaler.transform(y_train), label_scaler.transform(y_test)
    if validation:
        Y_valid = label_scaler.transform(y_valid)

    data_original = [x_train, y_train, x_valid, y_valid, x_test, y_test]
    data_final = [X_train, Y_train, X_valid, Y_valid, X_test, Y_test]
    scaler = {
        "feature": feature_scaler,
        "label": label_scaler
    }
    if save_original:
        return data_original, data_final, scaler
    else:
        return None, data_final, scaler


def get_dataset_kfold(X, y, kfold=3, shuffle=True, scale_type="minmax", save_original=False):
    data_scaled = []
    data_unscaled = []
    scaler_list = []
    kf = KFold(n_splits=kfold, shuffle=shuffle, random_state=Config.SEED)
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        feature_scaler = DataScaler(scale_type=scale_type).fit(x_train)
        X_train, X_test = feature_scaler.transform(x_train), feature_scaler.transform(x_test)

        label_scaler = DataScaler(scale_type=scale_type).fit(y_train)
        Y_train, Y_test = label_scaler.transform(y_train), label_scaler.transform(y_test)

        data_original = [x_train, y_train, x_test, y_test]
        data_final = [X_train, Y_train, X_test, Y_test]
        scaler = {
            "feature": feature_scaler,
            "label": label_scaler
        }
        data_unscaled.append(data_original)
        data_scaled.append(data_final)
        scaler_list.append(scaler)
    if save_original:
        return data_unscaled, data_scaled, scaler_list, kf
    else:
        return None, data_scaled, scaler_list, kf


def get_scaler(mode:str, X_data:None, lb=None, ub=None):
    """
    mode = "dataset" --> Get scaler based on input X
    mode = "lbub" --> get scaler based on lower bound, upper bound in phase 2
    """
    scaler = MinMaxScaler()  # Data scaling using the MinMax method
    if mode == "lbub":
        if lb is None or ub is None:
            print("Lower bound and upper bound for lbub scaling method are required!")
            exit(0)
        lb = np.squeeze(np.array(lb))
        ub = np.squeeze(np.array(ub))
        X_data = np.array([lb, ub])
        scaler.fit(X_data)
    else:           # mode == "dataset":
        scaler.fit(X_data)
    return scaler


def transform_one_hot_to_label(data):
    data = np.squeeze(np.array(data))
    return np.argmax(data, axis=1)

