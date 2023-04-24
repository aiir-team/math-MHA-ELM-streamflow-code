#!/usr/bin/env python
# Created by "Thieu" at 14:43, 13/09/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model-9e13838dd3de
# https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
# https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79


import platform
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

COLOURS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


def draw_roc_auc_curve(name_model, y_true, y_pred, title: str, filename: str, pathsave: str, exts: list, verbose=False):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # plot the roc curve for the model

    # roc curve for models
    fpr1, tpr1, _ = roc_curve(y_true, y_pred, pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_true))]
    p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)

    # auc scores
    auc_score1 = roc_auc_score(y_true, y_pred)

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label=f'{name_model}: ROC AUC={auc_score1:.3f}')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='Random classifier (worse)')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.title(title)

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_precision_recall_curve(name_model, y_true, y_pred, title: str, filename: str, pathsave: str, exts: list, verbose=False):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # plot the roc curve for the model

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color="blue", label='Random classifier (worse)')

    pr, rc, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    plt.plot(pr, rc, linestyle='--', color="orange", label=name_model)

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title(title)

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_loss_accuracy_curve(data: dict, list_legends: list, title: str, filename: str, pathsave: str, exts: list, verbose=True):
    plt.gcf().clear()
    fig, (ax1, ax2) = plt.subplots(2)
    epoch = np.arange(1, len(data['loss']) + 1)
    ax1.plot(epoch, np.array(data['loss']), label=list_legends[0])
    ax1.plot(epoch, np.array(data['val_loss']), label=list_legends[1])
    ax1.set_ylabel("Loss")
    ax2.plot(epoch, np.array(data['accuracy']))
    ax2.plot(epoch, np.array(data['val_accuracy']))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    fig.suptitle(title)
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.67, 0.1, 0.2, 0.87), bbox_transform = plt.gcf().transFigure )

    for idx, ext in enumerate(exts):
        fig.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        fig.show()
    plt.close()


def draw_roc_auc_multiclass(name_labels, y_true, y_pred_prob, title="Multiclass ROC curve",
                            filename="", pathsave="", exts=(), verbose=False):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # plot the roc curve for the model

    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(len(name_labels)):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_true, y_pred_prob[:, i], pos_label=i)

    for i in range(len(name_labels)):
        plt.plot(fpr[i], tpr[i], linestyle='--', color=COLOURS[i], label=f'Class {name_labels[i]} vs Rest')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.title(title)

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_precision_recall_multiclass(name_labels, y_true, y_pred_prob, title="Multiclass Precision Recall curve",
                                     filename="", pathsave="", exts=(), verbose=False):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # plot the roc curve for the model

    pr = {}
    rc = {}

    for i in range(len(name_labels)):
        pr[i], rc[i], _ = precision_recall_curve(y_true, y_pred_prob[:, i], pos_label=i)

    for i in range(len(name_labels)):
        plt.plot(pr[i], rc[i], linestyle='--', color=COLOURS[i], label=f'Class {name_labels[i]} vs Rest')

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title(title)

    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()
