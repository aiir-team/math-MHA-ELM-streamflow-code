#!/usr/bin/env python
# Created by "Thieu" at 13:00, 13/09/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
# for chapter 6.3   https://michael-fuchs-python.netlify.app/2019/10/31/introduction-to-logistic-regression/#model-evaluation
import pandas as pd
from permetrics.classification import ClassificationMetric
from sklearn.metrics import roc_auc_score
from config import Config
from config_kfold import ConfigKfold


# def classify_metrics(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None,
#                      n_labels=2, name_labels=None, positive_label=None, n_decimal=4):
#     accuracy = round(accuracy_score(y_true, y_pred), n_decimal)
#     error_rate = round(1 - accuracy_score(y_true, y_pred), n_decimal)
#     if n_labels == 2:
#         precision = round(precision_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         recall = round(recall_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         f1score = round(f1_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         auc = round(roc_auc_score(y_true, y_pred_scaled), n_decimal)
#     else:
#         precision = round(precision_score(y_true, y_pred, average='micro'), n_decimal)
#         recall = round(recall_score(y_true, y_pred, average='micro'), n_decimal)
#         f1score = round(f1_score(y_true, y_pred, average='micro'), n_decimal)
#         auc = round(roc_auc_score(y_true, y_pred_scaled, multi_class="ovr"), n_decimal)
#
#     ## Calculate metrics
#     metric_normal = {
#         "accuracy": accuracy,
#         "error": error_rate,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1score,
#         "roc_auc": auc
#     }
#
#     # Confusion matrix
#     matrix_conf = confusion_matrix(y_true, y_pred, labels=name_labels)
#
#     # For figure purpose only
#     # logit_roc_auc = roc_auc_score(y_true, y_pred)
#     # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#
#     return matrix_conf, metric_normal
from utils.model_util import get_best_parameter_kfold


def class_metrics(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None, n_labels=2, labels=None, n_decimal=5):
    evaluator = ClassificationMetric(y_true, y_pred, decimal=n_decimal)
    if n_labels == 2:
        paras = [{"average": "micro"}, ] * len(Config.METRICS_FOR_TESTING_PHASE)
        metrics = evaluator.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE, paras)
        # metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_pred_scaled), n_decimal)
    else:
        paras = [{"average": "macro"}, ] * len(Config.METRICS_FOR_TESTING_PHASE)
        metrics = evaluator.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE, paras)
        # metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_pred_scaled, multi_class="ovr"), n_decimal)
    # Confusion matrix

    matrix_conf = evaluator.confusion_matrix(y_true, y_pred, labels)

    # For figure purpose only
    # logit_roc_auc = roc_auc_score(y_true, y_pred)
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return matrix_conf, metrics


def get_best_parameter_set_from_kfold(metrics_list, pathsave):
    df = pd.DataFrame(metrics_list)
    tmin = df.groupby(by=["model_name", "model_paras"])[ConfigKfold.HEADER_METRIC_STATISTIC_CALCULATE].agg("min").reset_index()
    tmean = df.groupby(by=["model_name", 'model_paras'])[ConfigKfold.HEADER_METRIC_STATISTIC_CALCULATE].agg("mean").reset_index()
    tmax = df.groupby(by=["model_name", 'model_paras'])[ConfigKfold.HEADER_METRIC_STATISTIC_CALCULATE].agg("max").reset_index()
    tbest = get_best_parameter_kfold(tmean)

    with pd.ExcelWriter(f"{pathsave}/metric-final.xlsx") as writer:
        tmin.to_excel(writer, sheet_name='min', index=False)
        tmean.to_excel(writer, sheet_name='mean', index=False)
        tmax.to_excel(writer, sheet_name='max', index=False)
        tbest.to_excel(writer, sheet_name='best', index=False)

    return eval(tbest.iloc[0]["model_paras"])
