import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

threshold = 15
data_path = '/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3197.csv'
filtered_save_path = f'/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3197_filtered_below_{threshold}.csv'

mic_df = pd.read_csv(data_path)
mic_df_filtered = mic_df[mic_df['BAA-3197'] <= 15]

print(f'filtered active number of molecules: {len(mic_df_filtered)}')
mic_df_filtered.to_csv(filtered_save_path)