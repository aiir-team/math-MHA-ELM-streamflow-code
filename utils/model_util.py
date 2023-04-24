# !/usr/bin/env python
# Created by "Thieu" at 18:03, 29/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import pandas as pd
from pandas import read_csv, Series
import numpy as np
from config_kfold import ConfigKfold
import pickle


def is_min_operator_fitness(list_metrics):
    flag_min = False
    for metric_tune in list_metrics:
        for metric_min in ["MAE", "RMSE", "MAPE", "KLD", "VAR"]:
            if metric_min in metric_tune:
                flag_min = True
                break
    return flag_min


def save_system(system, pathfile=None, framework="keras", hybrid_model=True):
    if framework == "keras":
        system.model.save(f'{pathfile}.h5')
        del system.model
    elif framework == "tf":
        system.model.save(pathfile, save_format='tf')
        del system.model

    if hybrid_model:
        del system.optimizer.history.list_population
    name_obj = open(f'{pathfile}.pkl', 'wb')
    pickle.dump(system, name_obj)
    name_obj.close()


def load_system(pathfile=None, framework="keras"):
    from keras.models import load_model as keras_load_model

    system = pickle.load(open(f"{pathfile}.pkl", 'rb'))
    if framework == "keras":
        model = keras_load_model(f"{pathfile}.h5", compile=False)
        system.model = model
    elif framework == "tf":
        model = keras_load_model(pathfile, compile=False)
        system.model = model
    return system


def get_best_model(dataframe, path_results, metrics, weights):
    def fitness(cols):
        return Series([np.sum(cols * weights)])

    dataframe['fitness'] = dataframe[metrics].apply(fitness, axis=1)
    ## Get the best model based on the fitness
    if is_min_operator_fitness(metrics):
        minvalueIndexLabel = dataframe['fitness'].idxmin()
        best_data = dataframe.iloc[[minvalueIndexLabel]]
    else:
        maxvalueIndexLabel = dataframe['fitness'].idxmax()
        best_data = dataframe.iloc[[maxvalueIndexLabel]]
    best_dict = best_data.T.to_dict()
    best_model = best_dict[list(best_dict.keys())[0]]
    paras = eval(best_model["model_paras"])
    best_model["filename"] = "".join([f"{paras[key]}-" for key in paras.keys()])[:-1]
    name_folder = f"{path_results}/{best_model['path_paras']}/trial-{best_model['trial']}/{best_model['model_name']}"
    name_file = f"{best_model['filename']}-model"
    best_model["folder"] = name_folder
    best_model["file"] = name_file
    return best_model


def get_best_parameter_kfold(dataframe):
    df = dataframe.copy(deep=True)
    def fitness(cols):
        return Series([np.sum(cols * ConfigKfold.BEST_MODEL_WEIGHT_TUNE_PARAS)])

    df['fitness'] = df[ConfigKfold.BEST_MODEL_METRIC_TUNE_PARAS].apply(fitness, axis=1)
    if is_min_operator_fitness(ConfigKfold.BEST_MODEL_METRIC_TUNE_PARAS):
        minvalueIndexLabel = df['fitness'].idxmin()
        best_data = df.iloc[[minvalueIndexLabel]]
    else:
        maxvalueIndexLabel = df['fitness'].idxmax()
        best_data = df.iloc[[maxvalueIndexLabel]]
    return best_data


def get_best_parameter_retrain(pathload, model_name, paras):
    df = read_csv(pathload, usecols=["model_name", "model_paras"])
    paras_series = df[df['model_name'] == model_name]["model_paras"]
    paras_string = paras_series.values.tolist()[0]
    paras_list = []
    for para in paras_string.split('-'):
        if para.isalpha():
            paras_list.append(para)
        else:
            paras_list.append(eval(para))
    # paras_list = list(map(eval, paras_string.split('-')))
    paras_new = {}
    for idx, key in enumerate(paras.keys()):
        paras_new[key] = paras_list[idx]
    return paras_new


def update_best_paras_for_models_cv(models, pathread=None):
    for idx, model in enumerate(models):
        df = pd.read_excel(f"{pathread}/{model['name']}/metric-final.xlsx", sheet_name="best")
        paras_dict = eval(df["model_paras"][0])
        paras_dict_new = {}
        for para_name, para_value in paras_dict.items():
            paras_dict_new[para_name] = [para_value]
        models[idx]["param_grid"] = paras_dict_new
    return models


def save_model_phase2(obj, pathfile=None):
    del obj.app.model
    del obj.optimizer.history.list_population
    model_filename = open(f'{pathfile}.pkl', 'wb')
    pickle.dump(obj, model_filename)
    model_filename.close()
    return 0


def load_model_phase2(pathfile=None):
    obj = pickle.load(open(f"{pathfile}.pkl", 'rb'))
    return obj
