#!/usr/bin/env python
# Created by "Thieu" at 22:31, 05/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
import platform


def slope_intercept(x_test, y_test):
    x = np.array(x_test)
    y = np.array(y_test)
    m = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) / ((np.mean(x) * np.mean(x)) - np.mean(x * x)))
    m = round(m, 2)
    b = (np.mean(y) - np.mean(x) * m)
    b = round(b, 2)
    return m, b


def draw_correlation(list_data: list, dict_metrics: dict, list_colors: list, xy_labels: list,
                        title: str, filename: str, pathsave: str, exts: list):
    m, b = slope_intercept(list_data[0], list_data[1])
    reg_line = [(m * x) + b for x in list_data[0]]
    plt.figure(figsize=(6, 6))
    plt.scatter(list_data[0], list_data[1], color='green')
    plt.plot(list_data[0], reg_line)
    plt.title(title, size=13)
    plt.xlabel(xy_labels[0], size=12)
    plt.ylabel(xy_labels[1], size=12)
    if dict_metrics is not None:
        space = (max(list_data[1]) - min(list_data[1])) / 6
        for idx, (key, value) in enumerate(dict_metrics.items()):
            plt.text(min(list_data[0]), max(list_data[1])-0.5-0.35*idx*space, f'{key} = {value:.3f}',
                     size=10, ha="left", va="top", wrap=True)

    for idx, ext in enumerate(exts):
        plt.savefig(pathsave + filename + ext, bbox_inches='tight')
    if platform.system() != "Linux":
        plt.show()
    plt.close()

#
# x = choice(list(range(0, 100)), 100, replace=False)
# y = choice(list(range(0, 5000)), 100, replace=False)
# print(x)
# print(y)
#
# list_data = [x, y]
# dict_metrics = {
#     "R2": 0.87,
#     "RMSE": 0.69
# }
# title = "Testing X, Y"
# xy_labels = ["My X", "My X + 0.1"]
# draw_correlation(list_data, dict_metrics, None, xy_labels, title, "hello-world2", ".", [".pdf", ".png"])