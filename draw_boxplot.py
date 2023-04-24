#!/usr/bin/env python
# Created by "Thieu" at 04:40, 14/04/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
# https://www.projectpro.io/recipes/make-boxplot-and-interpret-it
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html
# https://stackoverflow.com/questions/35160956/pandas-boxplot-set-color-and-properties-for-box-median-mean
# https://python-charts.com/distribution/box-plot-matplotlib/
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from config import Config, MhaConfig
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(10, 4.5))
print(plt.style.available)


models = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10']
color_boxes = ["#43A047", "#3949AB", "#FB8C00",  "#E53935", "#8E24AA", "#FFC107"]
color = "b"
exts = [".pdf", ".png"]
np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10, 10), columns=models)
metrics = ["MAE", "RMSE", "MAPE", "R", "NSE", "KGE"]
metrics_test = [f"{item}_test" for item in metrics]

df_list = []
for act_name in Config.ELM_ACT_NAMES:
    for size_hidden in Config.ELM_HIDDEN_SIZES:
        for obj in Config.OBJ_FUNCS:
            for id_metric, metric in enumerate(metrics_test):
                df_dict = {}
                models = []
                for algorithm in MhaConfig.models_elm:
                    results = []
                    models.append(algorithm['name'])
                    for mha_paras in list(ParameterGrid(algorithm["param_grid"])):
                        for trial in range(0, Config.N_TRIALS):
                            pathread = f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}/trial-{trial}/{algorithm['name']}"
                            # Load metrics
                            filename = "".join([f"-{mha_paras[key]}" for key in algorithm['param_grid'].keys()])
                            df = pd.read_csv(f"{pathread}/{filename[1:]}-metrics.csv", usecols=["model_name", metric])
                            results.append(df[metric][0])
                    df_dict[algorithm['name']] = results
                df = pd.DataFrame(df_dict)

                # boxplot = df.boxplot(grid=True, rot=45, column=models, showfliers=True, showmeans=True,
                #     # notch=True,
                #     # patch_artist=True,
                #      color=dict(boxes=color_box, whiskers=color, medians=color, caps=color),)

                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax, props = df.plot.box(grid=True, rot=45, column=models, showfliers=True, showmeans=True,
                                        # patch_artist=True,
                                        return_type='both',
                                        color=dict(boxes=color_boxes[id_metric],
                                                   whiskers=color_boxes[id_metric],
                                                   medians=color_boxes[id_metric],
                                                   caps=color_boxes[id_metric]),
                                        ax=ax)
                # for patch in props['boxes']:
                #     patch.set_facecolor("red")

                # plt.xlabel("Models")
                plt.ylabel(f"{metrics[id_metric]}")
                # plt.title("The distribution of compared models in CI metric")
                plt.tight_layout()

                pathsave = f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}/{Config.FOLDER_VISUALIZE}/{Config.FOLDER_BOXPLOT}"
                Path(pathsave).mkdir(parents=True, exist_ok=True)
                for idx, ext in enumerate(exts):
                    plt.savefig(f"{pathsave}/{metric}{ext}", bbox_inches='tight')
                if platform.system() != "Linux":
                    plt.show()
                plt.close()
