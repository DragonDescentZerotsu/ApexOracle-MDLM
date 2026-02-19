import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

threshold = 15
mic_data_path_3197 = '/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3197.csv'
mic_data_path_3170 = '/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3170.csv'
# filtered_save_path_3197 = f'/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3197_filtered_below_{threshold}.csv'
# filtered_save_path_3170 = f'/data2/tianang/projects/mdlm/temp_data/SMs_mic_predictions_BAA-3170_filtered_below_{threshold}.csv'

collins_SMs_path = '/data2/tianang/projects/Synergy/DataPrepare/Data/small_molecule/processed/small_molecule_Evo_binary_data.csv'

mic_df_3197 = pd.read_csv(mic_data_path_3197)
mic_df_3197_filtered = mic_df_3197[mic_df_3197['BAA-3197'] <= 15]

mic_df_3170 = pd.read_csv(mic_data_path_3170)
mic_df_3170_filtered = mic_df_3170[mic_df_3170['BAA-3170'] <= 15]

mic_df_3197_active_smiles = set(mic_df_3197_filtered['SMILES_Sequence'].values.tolist())
mic_df_3170_active_smiles = set(mic_df_3170_filtered['SMILES_Sequence'].values.tolist())

collins_SMs_data = pd.read_csv(collins_SMs_path)
collins_SMs_active = set(collins_SMs_data[collins_SMs_data['MIC']>0.5]['SMILES'].values.tolist())

# canonicalise the SMILES
canonical_collins_SMs_active = set()
for smiles in tqdm(collins_SMs_active, desc='canonicalising SMILES from collins'):
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolToSmiles(mol, canonical=True)
    canonical_collins_SMs_active.add(canon)

canonical_3197_active = set()
for smiles in tqdm(mic_df_3197_active_smiles, desc='canonicalising SMILES from BAA-3197'):
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolToSmiles(mol, canonical=True)
    canonical_3197_active.add(canon)

canonical_3170_active = set()
for smiles in tqdm(mic_df_3170_active_smiles, desc='canonicalising SMILES from BAA-3170'):
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolToSmiles(mol, canonical=True)
    canonical_3170_active.add(canon)

venn2([canonical_collins_SMs_active, canonical_3170_active], set_labels=("canonical_collins_SMs_active", "canonical_3170_active"))
plt.title("Venn Diagram (2 sets)")
plt.show()