import glob
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

attr_names = ['pre', 'rec', 'f1', 'acc', 'auc']


def collect_best_results(metric="auc", dir=None, name=None):
    root_dir = f"."

    results, kernel_li = [], []
    file_li = glob.glob(osp.join(root_dir, "*", "Eval_*.tsv"))
    criterion = re.compile(".*K[0-9]*.tsv$")
    file_li = list(filter(criterion.match, file_li))

    for file in file_li:
        kernel = int(file.split('_K')[2].split('.tsv')[0])
        kernel_li += [kernel]
    assert len(file_li) == len(kernel_li)
    assert len(kernel_li) == len(set(kernel_li))


    kernel_li = []

    attr_data = [[] for _ in range(len(attr_names))]

    results_df_d = {}

    trend_d = dict(zip(attr_names, attr_data))

    for file in file_li:
        kernel = int(file.split('_K')[2].split('.tsv')[0])
        kernel_li += [kernel]
    print(f"Kernel sizes: {kernel_li}")

    kernel_li, file_li = zip(*sorted(zip(kernel_li, file_li), key=lambda x: x[0]))
    kernel_li, file_li = list(kernel_li), list(file_li)

    for file, kernel in zip(file_li, kernel_li):
        df = pd.read_csv(file, sep='\t')

        max_f1_index = df[metric].argmax()
        best_row = df.iloc[max_f1_index][attr_names]
        best_row = pd.DataFrame(best_row).T
        best_row['kernel'] = kernel
        results += [best_row]

        for attr_name in attr_names:
            trend_d[attr_name] += [pd.DataFrame(df[attr_name]).T]

    best_results_df = pd.concat(results)
    best_results_df = best_results_df.rename_axis('best_epoch').reset_index().set_index("kernel")

    for attr_name in attr_names:
        results_df_d[attr_name] = pd.concat(trend_d[attr_name]).reset_index()
        results_df_d[attr_name].index = kernel_li
        results_df_d[attr_name].rename_axis("kernel", inplace=True)

    results_merged_df = pd.concat(results_df_d.values())


def analyze_results(labs_tmp, preds_tmp, counters, filenames, epoch, args):
    preds_tmp = np.array(preds_tmp)
    labs_tmp = np.array(labs_tmp)
    tp = np.where((preds_tmp == labs_tmp) & (preds_tmp == 1))[0]
    tn = np.where((preds_tmp == labs_tmp) & (preds_tmp == 0))[0]
    fp = np.where((preds_tmp != labs_tmp) & (preds_tmp == 1))[0]
    fn = np.where((preds_tmp != labs_tmp) & (preds_tmp == 0))[0]

    filenames = np.array(filenames)
    for ids, counter in zip([tp, tn, fp, fn], counters):
        counter.update(filenames[ids])
    return counters


if __name__ == "__main__":
    sensitivity_analysis()
