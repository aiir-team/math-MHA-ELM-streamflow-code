# !/usr/bin/env python
# Created by "Thieu" at 20:36, 15/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import time
import pandas as pd
from models.mha import dict_optimizer_classes
from utils.data_util import get_dataset_kfold
from utils.io_util import save_results_to_csv, save_to_csv_dict
from utils.model_util import save_system
from utils.visual.line import draw_predict_line_with_error, draw_multiple_lines, draw_predict_and_error_distribution
from utils import math_util
from permetrics.regression import RegressionMetric
from config import Config
import numpy as np


class BaseClass:

    def __init__(self, base_paras=None):
        self.obj = base_paras["obj"]
        self.size_hidden = base_paras["size_hidden"]
        self.act_name = base_paras["act_name"]
        self.act_func = getattr(math_util, self.act_name)
        self.verbose = base_paras["verbose"]
        self.mode_train = base_paras["mode_train"]
        self.pathsave = base_paras["pathsave"]
        self.name_model = base_paras["name_model"]

        ## Below variables will be created by child class of this class
        self.framework = "numpy"  # keras, tf, numpy
        self.hybrid_model = True
        self.model, self.scaler, self.weights = None, {}, None
        self.fold_id, self.trial_id = None, None
        self.model_paras, self.filename = None, None
        self.solution, self.best_fit, self.loss_train = None, None, None
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None

    def get_weights(self):
        return self.weights

    def create_network(self, X_train, y_train):
        model = {
            "weights": {"w1": None, "b": None, "w2": None}
        }
        return model

    def decode_solution(self, solution):
        pass

    def predict(self, X=None):
        hidd = self.act_func(np.add(np.matmul(X, self.model["weights"]["w1"]), self.model["weights"]["b"]))
        return np.matmul(hidd, self.model["weights"]["w2"])

    def transform_X(self, X):
        return X 

    def predict_with_raw(self, X=None):
        X = self.transform_X(X)
        return self.predict(X)

    def prediction(self, X=None):
        self.decode_solution(self.solution)
        return self.model.predict(X)

    def predict_with_solution(self, X=None, solution=None):
        self.decode_solution(solution)
        X = self.transform_X(X)
        return self.predict(X)

    def fit(self, X_train, y_train, validation_data=None):
        # Depend the child class of this class. They will implement their training function
        pass

    def evaluate(self, x_scaled, y_scaled, name="experiment"):
        y_unscaled = self.scaler["label"].inverse_transform(y_scaled)
        ## Get the prediction for training and testing set
        y_pred_scaled = self.predict(x_scaled)
        ## Unscaling the predictions to calculate the errors
        y_pred_unscaled = self.scaler["label"].inverse_transform(y_pred_scaled)
        
        ## Calculate metrics
        RM1 = RegressionMetric(y_unscaled, y_pred_unscaled, decimal=5)
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)

        ## Save prediction results of training set and testing set to csv file
        data = {
            f"Y_{name}_true_scaled": y_scaled.flatten(),
            f"Y_{name}_true_unscaled": y_unscaled.flatten(),
            f"Y_{name}_pred_scaled": y_pred_scaled.flatten(),
            f"Y_{name}_pred_unscaled": y_pred_unscaled.flatten(),
        }
        final_metrics = {}
        for metric_name, value in mm1.items():
            final_metrics[f"{metric_name}_{name}"] = value
        return data, final_metrics

    def score(self, X, y, metric="R2"):
        y_pred = self.predict(X)
        y_pred_unscaled = self.scaler["label"].inverse_transform(y_pred)
        y_unscaled = self.scaler["label"].inverse_transform(y)
        RM1 = RegressionMetric(y_unscaled, y_pred_unscaled, decimal=5)
        return RM1.get_metric_by_name(metric)[metric]

    def save_results(self, results: dict):
        ## Save prediction results of training set and testing set to csv file
        data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_TRAIN_FILE}
        save_to_csv_dict(data, f"{self.filename}-{Config.FILENAME_PRED_TRAIN}", self.pathsave)

        if Config.VALIDATION_USED:
            data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_VALID_FILE}
            save_to_csv_dict(data, f"{self.filename}-{Config.FILENAME_PRED_VALID}", self.pathsave)

        data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_TEST_FILE}
        save_to_csv_dict(data, f"{self.filename}-{Config.FILENAME_PRED_TEST}", self.pathsave)

        ## Save loss train to csv file
        epoch = list(range(1, len(self.loss_train['loss']) + 1))
        data = {"epoch": epoch, "loss": self.loss_train['loss'], "val_loss": self.loss_train['val_loss']}
        save_to_csv_dict(data, f"{self.filename}-{Config.FILENAME_LOSS_TRAIN}", self.pathsave)

        ## Calculate performance metrics and save it to csv file
        RM1 = RegressionMetric(results[Config.Y_TRAIN_TRUE_UNSCALED], results[Config.Y_TRAIN_PRED_UNSCALED], decimal=5)
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        mm2 = {}
        if Config.VALIDATION_USED:
            RM2 = RegressionMetric(results[Config.Y_VALID_TRUE_UNSCALED], results[Config.Y_VALID_PRED_UNSCALED], decimal=5)
            mm2 = RM2.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        RM3 = RegressionMetric(results[Config.Y_TEST_TRUE_UNSCALED], results[Config.Y_TEST_PRED_UNSCALED], decimal=5)
        mm3 = RM3.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        item = {"model_name": self.name_model, "model_paras": self.model_paras, "time_train": self.time_train, "time_total": self.time_total}
        for metric_name, value in mm1.items():
            item[metric_name + "_train"] = value
        if Config.VALIDATION_USED:
            for metric_name, value in mm2.items():
                item[metric_name + "_valid"] = value
        for metric_name, value in mm3.items():
            item[metric_name + "_test"] = value
        save_results_to_csv(item, f"{self.filename}-{Config.FILENAME_METRICS}", self.pathsave)

        ## Visualization
        #### Draw prediction accuracy
        draw_predict_line_with_error([results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten()],
                                     [item["MAE_test"], item["RMSE_test"]], f"{self.filename}-{Config.FILENAME_PERFORMANCE}",
                                     self.pathsave, Config.FILE_FIGURE_TYPES)
        #### Draw convergence
        epoch = np.array(list(range(1, len(self.loss_train['loss']) + 1)))
        list_lines = [[epoch, self.loss_train['loss']], [epoch, self.loss_train['val_loss']]]
        list_legends = ["Training loss", "Validation loss"] if Config.VALIDATION_USED else ["Training loss", "Testing loss"]
        draw_multiple_lines(list_lines, list_legends, None, None, ["Iterations", f"{self.obj}"],
                            title=None, filename=f"{self.filename}-{Config.FILENAME_CONVERGENCE}",
                            pathsave=self.pathsave, exts=[".png", ".pdf"], verbose=False)

        ## Save models
        if Config.SAVE_MODEL:
            save_system(self, pathfile=f"{self.pathsave}/{self.filename}-{Config.FILENAME_MODEL}", framework=self.framework, hybrid_model=self.hybrid_model)

    ## Processing all tasks
    def processing(self, data=None, validation=False, timeseries=None):
        self.time_total = time.perf_counter()
        self.validation, self.timeseries = validation, timeseries

        ## Pre-processing dataset
        self.X_train, self.X_valid, self.X_test = data["X_train"], data["X_valid"], data["X_test"]
        self.Y_train, self.Y_valid, self.Y_test = data['Y_train'], data['Y_valid'], data['Y_test']
        ## Build the network for hybrid model
        if self.model is None:
            self.model = self.create_network(self.X_train, self.Y_train)

        ## Training
        self.time_train = time.perf_counter()
        if self.validation:
            self.fit(self.X_train, self.Y_train, (self.X_valid, self.Y_valid))
        else:
            self.fit(self.X_train, self.Y_train, (self.X_test, self.Y_test))
        self.time_train = round(time.perf_counter() - self.time_train, 3)
        self.decode_solution(self.solution)

        ## Get the prediction for training and testing set
        Y_train_pred = self.predict(self.X_train)
        Y_valid_pred = self.predict(self.X_valid) if self.validation else None
        Y_test_pred = self.predict(self.X_test)

        ## Unscaling the predictions to calculate the errors
        results = {
            Config.Y_TRAIN_TRUE_SCALED: self.Y_train,
            Config.Y_TRAIN_TRUE_UNSCALED: self.timeseries.inverse_scale(self.Y_train),
            Config.Y_TRAIN_PRED_SCALED: Y_train_pred,
            Config.Y_TRAIN_PRED_UNSCALED: self.timeseries.inverse_scale(Y_train_pred),

            Config.Y_TEST_TRUE_SCALED: self.Y_test,
            Config.Y_TEST_TRUE_UNSCALED: self.timeseries.inverse_scale(self.Y_test),
            Config.Y_TEST_PRED_SCALED: Y_test_pred,
            Config.Y_TEST_PRED_UNSCALED: self.timeseries.inverse_scale(Y_test_pred)
        }
        if validation:
            results[Config.Y_VALID_TRUE_SCALED] = self.Y_valid
            results[Config.Y_VALID_TRUE_UNSCALED] = self.timeseries.inverse_scale(self.Y_valid)
            results[Config.Y_VALID_PRED_SCALED] = Y_valid_pred
            results[Config.Y_VALID_PRED_UNSCALED] = self.timeseries.inverse_scale(Y_valid_pred)

        self.time_total = round(time.perf_counter() - self.time_total, 3)
        self.save_results(results)

    ## Processing all tasks
    def run_normal(self, data:dict, scaler:dict, validation=False):
        self.time_total = time.perf_counter()
        self.scaler = scaler
        self.validation = validation

        ## Pre-processing features
        self.X_train, self.X_valid, self.X_test = data["X_train"], data["X_valid"], data["X_test"]
        self.Y_train, self.Y_valid, self.Y_test = data['Y_train'], data['Y_valid'], data['Y_test']
        if self.model is None:
            self.model = self.create_network(self.X_train, data[1])

        ## Training
        self.time_train = time.perf_counter()
        if self.validation:
            self.fit(self.X_train, self.Y_train, (self.X_valid, self.Y_valid))
        else:
            self.fit(self.X_train, self.Y_train, (self.X_test, self.Y_test))
        self.time_train = round(time.perf_counter() - self.time_train, 3)
        self.decode_solution(self.solution)

        ## Get the prediction for training and testing set
        Y_train_pred = self.predict(self.X_train)
        Y_test_pred = self.predict(self.X_test)
        Y_valid_pred = self.predict(self.X_valid) if self.validation else None

        results = {
            Config.Y_TRAIN_TRUE_SCALED: self.Y_train,
            Config.Y_TRAIN_TRUE_UNSCALED: self.scaler["label"].inverse_transform(self.Y_train),
            Config.Y_TRAIN_PRED_SCALED: Y_train_pred,
            Config.Y_TRAIN_PRED_UNSCALED: self.scaler["label"].inverse_transform(Y_train_pred) ,

            Config.Y_TEST_TRUE_SCALED: self.Y_test,
            Config.Y_TEST_TRUE_UNSCALED: self.scaler["label"].inverse_transform(self.Y_test),
            Config.Y_TEST_PRED_SCALED: Y_test_pred,
            Config.Y_TEST_PRED_UNSCALED: self.scaler["label"].inverse_transform(Y_test_pred),
        }
        if self.validation:
            results[Config.Y_VALID_TRUE_SCALED] = self.Y_valid
            results[Config.Y_VALID_TRUE_UNSCALED] = self.scaler["label"].inverse_transform(self.Y_valid)
            results[Config.Y_VALID_PRED_SCALED] = Y_valid_pred,
            results[Config.Y_VALID_PRED_UNSCALED] = self.scaler["label"].inverse_transform(Y_valid_pred)

        self.time_total = round(time.perf_counter() - self.time_total, 3)
        self.save_results(results)
    
    def run_train_test(self, train_data:list, test_data:list, scaler: dict, model:any):
        time_run_total = time.perf_counter()
        x_train, y_train = self.transform_X(train_data[0]), train_data[1].copy()
        x_test, y_test = self.transform_X(test_data[0]), test_data[1].copy()
        self.scaler = scaler
        if model is None:
            self.model = self.create_network(x_train, y_train)

        ## Training
        time_run_train = time.perf_counter()
        self.fit(x_train, y_train, (x_test, y_test))
        self.decode_solution(self.solution)
        time_run_train = time.perf_counter() - time_run_train

        data_train, metrics_train = self.evaluate(x_train, train_data[1], name="train")
        data_test, metrics_test = self.evaluate(x_test, test_data[1], name="test")

        results = {**data_train, **data_test}
        metrics = {**metrics_train, **metrics_test}
        time_run_total = time.perf_counter() - time_run_total
        return results, metrics, time_run_train, time_run_total, self.model, self.scaler
    
    def run_kfold(self, X=None, y=None, scale_type="minmax", kfold=5, n_repeated=1, save_models=False):
        self.models = []
        self.scalers = []
        self.metrics_final = []
        _, data_scaled, scaler_list, self.kfold = get_dataset_kfold(X, y, kfold=kfold, shuffle=True, scale_type=scale_type, save_original=False)
        for data_idx, data_list in enumerate(data_scaled):
            _, metrics, time_run_train, time_run_total, model, scaler = self.run_train_test(data_list[:2], data_list[2:], scaler_list[data_idx], None)
            self.models.append(model)
            self.scalers.append(scaler)
            metrics_dict = {**{'model_name': self.name_model, 'model_paras': self.model_paras,
                               'fold_id': data_idx, "repeat_id": 0,
                               "time_train": time_run_train, "time_total": time_run_total,
                               }, **metrics}
            self.metrics_final.append(metrics_dict)
            save_results_to_csv(metrics_dict, f"{self.filename}-metrics_all", self.pathsave)

        for repeat_idx in range(1, n_repeated):
            for data_idx, data_list in enumerate(data_scaled):
                _, metrics, time_run_train, time_run_total, model, scaler = self.run_train_test(data_list[:2], data_list[2:], self.scalers[data_idx], self.models[data_idx])
                metrics_dict = {**{'model_name': self.name_model, 'model_paras': self.model_paras,
                                   'fold_id': data_idx, "repeat_id": repeat_idx,
                                   "time_train": time_run_train, "time_total": time_run_total}, **metrics}
                self.metrics_final.append(metrics_dict)
                save_results_to_csv(metrics_dict, f"{self.filename}-metrics_all", self.pathsave)


class GradientElm(BaseClass):
    def __init__(self, base_paras=None, gradient_paras=None):
        super().__init__(base_paras)
        self.para = gradient_paras["para"]
        self.filename = f"{self.para}"
        self.model_paras = str(gradient_paras)
        self.hybrid_model = False

    def fit(self, X_train, y_train, validation_data=None):
        """
            1. Random weights between input and hidden layer
            2. Calculate output of hidden layer
            3. Calculate weights between hidden and output layer based on matrix multiplication
        """
        w1 = np.random.uniform(size=[X_train.shape[1], self.size_hidden])
        b = np.random.uniform(size=[1, self.size_hidden])
        H = self.act_func(np.add(np.matmul(X_train, w1), b))
        w2 = np.dot(np.linalg.pinv(H), y_train)
        self.model["weights"] = {"w1": w1, "b": b, "w2": w2}
        self.loss_train = {'loss': [], 'val_loss': []}
        

class MhaHybridElm(BaseClass):
    def __init__(self, base_paras=None, hybrid_paras=None, mha_paras=None, name_opt=None):
        super().__init__(base_paras)
        self.hybrid_paras = hybrid_paras
        self.optimizer = dict_optimizer_classes[name_opt](**mha_paras)
        self.filename = "-".join([str(mha_paras[key]) for key in mha_paras.keys()])
        self.model_paras = str(mha_paras)
        self.term = {
            "max_fe": 50000
        }

    def decode_solution(self, solution):
        w1 = np.reshape(solution[:self.size_w1], (self.train_x.shape[1], self.size_hidden))
        b = np.reshape(solution[self.size_w1:self.size_w1 + self.size_b], (-1, self.size_hidden))
        H = self.act_func(np.add(np.matmul(self.train_x, w1), b))
        w2 = np.dot(np.linalg.pinv(H), self.train_y)  # calculate weights between hidden and output layer
        self.model["weights"] = {"w1": w1, "b": b, "w2": w2}

    # Evaluates the fitness function
    def fitness_function(self, solution=None):
        self.decode_solution(solution)
        ## Get the prediction for training and testing set
        Y_train_pred = self.predict(self.train_x)
        Y_test_pred = self.predict(self.test_x)
        loss_train = RegressionMetric(self.train_y, Y_train_pred, decimal=8).get_metric_by_name(self.obj)[self.obj]
        loss_test = RegressionMetric(self.test_y, Y_test_pred, decimal=8).get_metric_by_name(self.obj)[self.obj]
        return [loss_train, loss_test]

    def fit(self, X_train, y_train, validation_data=None):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, y_train, validation_data[0], validation_data[1]
        self.size_w1 = X_train.shape[1] * self.size_hidden
        self.size_b = self.size_hidden
        self.size_w2 = self.size_hidden * 1
        self.problem_size = self.size_w1 + self.size_b
        self.lb = self.hybrid_paras["lb"] * self.problem_size
        self.ub = self.hybrid_paras["ub"] * self.problem_size
        log_to = "console" if self.verbose else "None"
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": self.lb,
            "ub": self.ub,
            "minmax": "min",
            "log_to": log_to,
            "save_population": False,
            "obj_weights": [1., 0.]
        }

        self.solution, self.best_fit = self.optimizer.solve(self.problem, termination=self.term)
        self.loss_train = self.get_history_loss(self.optimizer.history.list_global_best)

    def get_history_loss(self, list_global_best=None):
        # 2D array / matrix 2D
        global_obj_list = np.array([agent[1][-1] for agent in list_global_best])
        global_obj_list = global_obj_list[1:]
        # Make each obj_list as a element in array for drawing
        return {
            "loss": global_obj_list[:, 0],
            "val_loss": global_obj_list[:, 1]
        }
