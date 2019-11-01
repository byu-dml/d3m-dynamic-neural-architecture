import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


pipeline_data = pd.read_csv('./pipeline_seed_runs.csv', index_col=0)
pipeline_data['runs'] = pipeline_data['runs'].apply(pd.eval)

for group_id, group_data in pipeline_data.groupby('dataset_id'):
    group_data.sort_values(['f1_macro_mean_over_runs', 'f1_macro_std_dev_over_runs'], ascending=[False, True], inplace=True)
    response = []
    response_label = []
    for i, (runs, pipeline_id) in enumerate(zip(group_data['runs'], group_data['pipeline_id'])):
        if i % 50 == 0:
            response.extend(runs)
            response_label.extend([pipeline_id]*len(runs))
    assert len(response) == len(response_label)

    best_pipeline_id = group_data['pipeline_id'].iloc[0]
    reject_indices = []
    for runs in group_data['runs']:
        res = stats.ttest_ind(group_data['runs'].iloc[0], runs, equal_var=False)
        reject_indices.append((res.pvalue * pipeline_data.shape[0]) < 0.05)  # Bonferonni adjustment to p-value
    reject_indices = pd.Series(reject_indices)
    
    x = pd.Series(np.arange(group_data.shape[0]))#.reset_index()
    y = group_data['f1_macro_mean_over_runs'].reset_index(drop=True)
    yerr = 1.96*group_data['f1_macro_std_dev_over_runs'].reset_index(drop=True)

    # plt.plot(x[~reject_indices], y[~reject_indices], color='blue')
    # plt.fill_between(x[~reject_indices], (y-yerr)[~reject_indices], (y+yerr)[~reject_indices], alpha=.25, color='blue')
    # plt.plot(x[reject_indices], y[reject_indices], color='red')
    # plt.fill_between(x[reject_indices], (y-yerr)[reject_indices], (y+yerr)[reject_indices], alpha=.25, color='red')

    marekers, caps, bars = plt.errorbar(x[~reject_indices], y[~reject_indices], yerr=yerr[~reject_indices], color='blue', fmt='o', markersize=1, label='No Significant Difference')
    [bar.set_alpha(0.25) for bar in bars]
    marekers, caps, bars = plt.errorbar(x[reject_indices], y[reject_indices], yerr=yerr[reject_indices], color='red', fmt='o', markersize=1, label='Significant Difference')
    [bar.set_alpha(0.25) for bar in bars]

    plt.title(group_id)
    plt.xlabel('Pipeline Index (Sorted by Average F1 Score)')
    plt.ylabel('F1 Macro Score')
    plt.legend(loc='lower left')
    plt.show()
    plt.close()
