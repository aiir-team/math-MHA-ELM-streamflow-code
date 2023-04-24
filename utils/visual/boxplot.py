#!/usr/bin/env python
# Created by "Thieu" at 05:31, 14/04/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# https://www.projectpro.io/recipes/make-boxplot-and-interpret-it
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html
# https://stackoverflow.com/questions/35160956/pandas-boxplot-set-color-and-properties-for-box-median-mean
# https://python-charts.com/distribution/box-plot-matplotlib/

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('seaborn-darkgrid')
print(plt.style.available)


models = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10']
color_box = "#4285F4"
color = "b"

np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10, 10), columns=models)



boxplot = df.boxplot(grid=True, rot=45, column=models, showfliers=True, showmeans=True,
                    # notch=True,
                    # patch_artist=True,
                     color=dict(boxes=color_box, whiskers=color, medians=color, caps=color),)


# fig, ax = plt.subplots(figsize=(9,6))
# ax, props = df.plot.box(grid=True, rot=45, column=models, showfliers=True, showmeans=True,
#                         patch_artist=True,
#                         return_type='both',
#                         # color=dict(boxes=color_box, whiskers=color, medians=color, caps=color),
#                         ax=ax)
# for patch in props['boxes']:
#     patch.set_facecolor("red")



plt.xlabel("Models")
plt.ylabel("Confidence Index")
plt.title("The distribution of compared models in CI metric")
plt.tight_layout()
plt.show()

