#!/usr/bin/env python
# Created by "Thieu" at 18:22, 21/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from sklearn.model_selection import ParameterGrid


def get_all_performance_metrics(list_pathread, trials, models, suf_fileread, filesave, pathsave):
    df_list = []
    for path_paras, pathread in list_pathread.items():
        for trial in range(trials):
            for model in models:
                path_model = f"{pathread}/trial-{trial}/{model['name']}"
                keys = model["param_grid"].keys()
                for mha_paras in list(ParameterGrid(model["param_grid"])):
                    # Load metrics
                    filename = "".join([f"-{mha_paras[key]}" for key in keys])
                    df = pd.read_csv(f"{path_model}/{filename[1:]}-{suf_fileread}.csv")
                    df.insert(0, 'path_paras', [path_paras], allow_duplicates=False)
                    df.insert(1, 'trial', [trial], allow_duplicates=False)
                    df_list.append(df)
    merged_df = pd.concat(df_list)
    merged_df.to_csv(f"{pathsave}/{filesave}.csv", index=False)
    # # savetxt(f"{Config.DATA_RESULTS}/statistics_final.csv", matrix_results, delimiter=",")
    # df = pd.read_csv(f"{pathsave}/{filesave}.csv", usecols=cols_header_save)
    return merged_df


def calculate_statistics(df_results, cols_header_read, pathsave):
    df = df_results.copy(deep=True)
    tmin = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].agg("min").reset_index()
    tmean = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].agg("mean").reset_index()
    tmax = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].agg("max").reset_index()
    tstd = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].agg("std").reset_index()

    # # group by 'group' column
    # tmean2 = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].mean()
    # tstd2 = df.groupby(by=["path_paras", "model_name", "model_paras"])[cols_header_read].std()
    # tcv = (tstd2 / tmean2) * 100
    # tcv.reset_index(inplace=True)

    # calculate the CV for each column by group
    tcv = df.groupby(["path_paras", "model_name", "model_paras"])[cols_header_read].apply(lambda x: x.std() / x.mean() * 100)
    tcv.reset_index(inplace=True)

    with pd.ExcelWriter(f"{pathsave}.xlsx") as writer:
        tmin.to_excel(writer, sheet_name='min', index=False)
        tmean.to_excel(writer, sheet_name='mean', index=False)
        tmax.to_excel(writer, sheet_name='max', index=False)
        tstd.to_excel(writer, sheet_name='std', index=False)
        tcv.to_excel(writer, sheet_name='cv', index=False)
