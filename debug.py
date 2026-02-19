from rdkit import Chem
import pandas as pd
file_path = 'temp_data/Molport_SMILES/iis_smiles-000-000-000--000-499-999.txt'
df = pd.read_csv(file_path, sep='\s+', on_bad_lines='skip', quoting=3)
print(df.head())