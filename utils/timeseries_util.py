#!/usr/bin/env python
# Created by "Thieu" at 18:22, 21/04/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class CheckDataset:
    def __init__(self):
        pass

    def _checking_consecutive__(self, df, time_name="timestamp", time_different=300):
        """
        :param df: Type of this must be dataframe
        :param time_name: the column name of date time
        :param time_different by seconds: 300 = 5 minutes
            https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html
        :return:
        """
        consecutive = True
        for i in range(df.shape[0] - 1):
            diff = (df[time_name].iloc[i + 1] - df[time_name].iloc[i]).seconds
            if time_different != diff:
                print("===========Not consecutive at: {}, different: {} ====================".format(i + 3, diff))
                consecutive = False
        return consecutive


class TimeSeries:
    def __init__(self, data=None, train_split=0.8, scale_type="std", separate=True):
        self.data_original = data
        if train_split < 1.0:
            self.train_split = int(train_split * self.data_original.shape[0])
        else:
            self.train_split = train_split
        self.scale_type = scale_type
        self.separate = separate
        if separate:
            self.data_mean, self.data_std = self.data_original[:self.train_split].mean(axis=0), self.data_original[:self.train_split].std(axis=0)
            self.data_min, self.data_max = self.data_original[:self.train_split].min(axis=0), self.data_original[:self.train_split].max(axis=0)
        else:
            self.data_mean, self.data_std = self.data_original.mean(axis=0), self.data_original.std(axis=0)
            self.data_min, self.data_max = self.data_original.min(axis=0), self.data_original.max(axis=0)
        self.data_new = None

    def scale(self):
        if self.scale_type == "std":
            self.data_new = (self.data_original - self.data_mean) / self.data_std
        elif self.scale_type == "minmax":
            self.data_new = (self.data_original - self.data_min) / (self.data_max - self.data_min)
        elif self.scale_type == "loge":
            self.data_new = np.log(self.data_original)

        elif self.scale_type == "kurtosis":
            self.data_new = np.sign(self.data_original - self.data_mean) * np.power(np.abs(self.data_original - self.data_mean), 1.0/3)
        elif self.scale_type == "kurtosis_std":
            self.data_kurtosis = np.sign(self.data_original - self.data_mean) * np.power(np.abs(self.data_original - self.data_mean), 1.0 / 3)
            self.data_mean_kur, self.data_std_kur = self.data_original[:self.train_split].mean(axis=0), self.data_original[:self.train_split].std(axis=0)
            self.data_new = (self.data_kurtosis - self.data_mean_kur) / self.data_std_kur

        elif self.scale_type == "boxcox":
            self.data_new, self.lamda_boxcox = boxcox(self.data_original.flatten())
        elif self.scale_type == "boxcox_std":
            self.data_boxcox, self.lamda_boxcox = boxcox(self.data_original.flatten())
            self.data_boxcox = self.data_boxcox.reshape(-1, 1)
            self.data_mean, self.data_std = self.data_boxcox[:self.train_split].mean(axis=0), self.data_boxcox[:self.train_split].std(axis=0)
            self.data_new = (self.data_boxcox - self.data_mean) / self.data_std
        return self.data_new

    def inverse_scale(self, data=None):
        if self.scale_type == "std":
            return self.data_std * data + self.data_mean
        elif self.scale_type == "minmax":
            return data * (self.data_max - self.data_min) + self.data_min
        elif self.scale_type == "loge":
            return np.exp(data)

        elif self.scale_type == "kurtosis":
            return np.power(data, 3) + self.data_mean
        elif self.scale_type == "kurtosis_std":
            temp = self.data_std_kur * data + self.data_mean_kur
            return np.power(temp, 3) + self.data_mean

        elif self.scale_type == "boxcox":
            return inv_boxcox(data, self.lamda_boxcox)
        elif self.scale_type == "boxcox_std":
            boxcox_invert = self.data_std * data + self.data_mean
            return inv_boxcox(boxcox_invert, self.lamda_boxcox)

    def make_univariate_data(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param history_column: python list time in the past you want to use. (1, 2, 5) means (t-1, t-2, t-5) predict time t
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param pre_type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + np.array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + history_column[-1]])
        if pre_type == "3D":
            return np.array(data), np.array(labels)
        return np.reshape(np.array(data), (-1, history_size)), np.array(labels)