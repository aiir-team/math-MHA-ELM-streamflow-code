#!/usr/bin/env python
# Created by "Thieu" at 17:49, 24/02/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import pandas as pd
from sklearn.model_selection import ParameterGrid
import concurrent.futures as parallel
from models.base_elm import MhaHybridElm
from config import Config, MhaConfig
from utils.timeseries_util import TimeSeries
import time


def develop_model(algorithm):
    print(f"Start running: {algorithm['name']}")

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
                            "pathsave": f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}/trial-{trial}/{algorithm['name']}",
                            "name_model": algorithm['name'],
                        }
                        hybrid_paras = {
                            "lb": Config.MHA_LB,
                            "ub": Config.MHA_UB,
                        }
                        for paras_temp in list(ParameterGrid(algorithm["param_grid"])):
                            mha_paras = dict((key, paras_temp[key]) for key in algorithm["param_grid"].keys())
                            md = MhaHybridElm(base_paras, hybrid_paras, mha_paras, algorithm["name"].split("-")[0])
                            md.processing(dataset, Config.VALIDATION_USED, timeseries)


if __name__ == '__main__':
    time_start = time.perf_counter()
    print(f"MHA-ELM Start!!!")
    # develop_model(MhaConfig.models[0])
    with parallel.ProcessPoolExecutor(Config.N_CPUS_RUN) as executor:
        results = executor.map(develop_model, MhaConfig.models_elm)
    print(f"MHA-ELM DONE: {time.perf_counter() - time_start} seconds")
