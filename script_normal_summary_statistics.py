# !/usr/bin/env python
# Created by "Thieu" at 16:11, 03/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from config import Config, MhaConfig
from utils.statistic_util import get_all_performance_metrics, calculate_statistics


for act_name in Config.ELM_ACT_NAMES:
    for size_hidden in Config.ELM_HIDDEN_SIZES:
        for obj in Config.OBJ_FUNCS:
            list_pathread = {
                f"{act_name}-{size_hidden}-{obj}": f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}"
            }
            df_results = get_all_performance_metrics(list_pathread, Config.N_TRIALS, MhaConfig.models_elm, Config.FILENAME_METRICS,
                                                     Config.FILENAME_METRICS_ALL_MODELS, Config.DATA_RESULTS)
            print(df_results.info())
            calculate_statistics(df_results, Config.HEADER_METRIC_STATISTIC_CALCULATE,
                                 f"{Config.DATA_RESULTS}/{act_name}-{size_hidden}-{obj}/{Config.FILENAME_STATISTICS}")
