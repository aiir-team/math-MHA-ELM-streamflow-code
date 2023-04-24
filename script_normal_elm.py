#!/usr/bin/env python
# Created by "Thieu" at 21:36, 15/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import pandas as pd
from sklearn.model_selection import ParameterGrid
from models.base_elm import GradientElm
from config import Config, MhaConfig
from utils.timeseries_util import TimeSeries
import time


if __name__ == '__main__':
    start_time = time.perf_counter()
    data = pd.read_csv(f"{Config.DATA_INPUT}/{Config.NAME_DATASET}")[Config.FEATURE_X].values
    for key_lags, lags in Config.DICT_LAGS_X.items():
        timeseries = TimeSeries(data, 1 - Config.TEST_SIZE, Config.ELM_SCALER, separate=True)
        data_new = timeseries.scale()
        X_train, y_train = timeseries.make_univariate_data(data_new, lags, 0, timeseries.train_split)
        X_test, y_test = timeseries.make_univariate_data(data_new, lags, timeseries.train_split, None)
        dataset = {
            "X_train": X_train, "Y_train": y_train, "X_valid": None, "Y_valid": None, "X_test": X_test, "Y_test": y_test
        }
        for act_name in Config.ELM_ACT_NAMES:
            for size_hidden in Config.ELM_HIDDEN_SIZES:
                for obj in Config.OBJ_FUNCS:
                    for trial in range(0, Config.N_TRIALS):
                        base_paras = {
                            "obj": obj,
                            "size_hidden": size_hidden,
                            "act_name": act_name,
                            "mode_train": Config.MHA_MODE_TRAIN_PHASE1,
                            "verbose": Config.VERBOSE,
                            "pathsave": f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}/trial-{trial}/ELM",
                            "name_model": "ELM",
                        }
                        for paras_temp in list(ParameterGrid(MhaConfig.elm)):
                            flnn_paras = dict((key, paras_temp[key]) for key in MhaConfig.elm.keys())
                            md = GradientElm(base_paras, flnn_paras)
                            md.processing(dataset, Config.VALIDATION_USED, timeseries)
    print('Finished ELM model after: {} seconds'.format(time.perf_counter() - start_time))
