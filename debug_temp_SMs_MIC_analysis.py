import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

def trim_iqr(x, k=5):
    x = np.asarray(x)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return x[(x >= lo) & (x <= hi)]


data_path = '/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3197.csv'

mic_data = pd.read_csv(data_path)['BAA-3197'].values
original_length = len(mic_data)
mic_data = trim_iqr(mic_data)
print(f'{len(mic_data)} / {original_length}')

# x = np.random.randn(500)

fig, ax = plt.subplots()
ax.violinplot(mic_data, showmeans=True, showmedians=True)
ax.set_xticks([1])
ax.set_xticklabels(["BAA-3197"])
ax.set_ylabel("MIC")
plt.show()