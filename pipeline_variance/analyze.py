import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


alpha = 0.1
z_star = 1.64

pipeline_data = pd.read_csv('./pipeline_seed_runs.csv', index_col=0)
pipeline_data['runs'] = pipeline_data['runs'].apply(pd.eval)

for group_id, group_data in pipeline_data.groupby('dataset_id'):
    group_data.sort_values(['f1_macro_mean_over_runs', 'f1_macro_std_dev_over_runs'], ascending=[False, True], inplace=True)

    best_pipeline_id = group_data['pipeline_id'].iloc[0]
    reject_indices = []
    for runs in group_data['runs']:
        res = stats.ttest_ind(group_data['runs'].iloc[0], runs, equal_var=False)
        reject_indices.append((res.pvalue * pipeline_data.shape[0]) < alpha)  # Bonferonni adjustment to p-value
    reject_indices = pd.Series(reject_indices)

    x = pd.Series(np.arange(group_data.shape[0]))
    y = group_data['f1_macro_mean_over_runs'].reset_index(drop=True)
    yerr = z_star*group_data['f1_macro_std_dev_over_runs'].reset_index(drop=True)

    marekers, caps, bars = plt.errorbar(x[~reject_indices], y[~reject_indices], yerr=yerr[~reject_indices], color='blue', fmt='o', markersize=1, label='No Significant Difference')
    [bar.set_alpha(0.25) for bar in bars]
    marekers, caps, bars = plt.errorbar(x[reject_indices], y[reject_indices], yerr=yerr[reject_indices], color='red', fmt='o', markersize=1, label='Significant Difference')
    [bar.set_alpha(0.25) for bar in bars]

    plt.title(group_id)
    plt.xlabel('Pipeline Index (Sorted by Average F1 Score)')
    plt.ylabel('F1 Macro Score')
    plt.tick_params(labelleft=True, labelright=True)
    plt.legend(loc='lower left')
    plt.savefig('{}_pipeline_variance_at.pdf'.format(group_id))
    plt.close()
