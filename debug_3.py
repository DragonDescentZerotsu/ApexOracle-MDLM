import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import sys

def check_canonical():
    file_path = 'temp_data/Molport_SMILES/iis_smiles-000-000-000--000-499-999.txt'
    print(f"Reading {file_path}...")
    
    # Try reading as whitespace separated
    try:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Columns found: {df.columns.tolist()}")
    if 'SMILES' not in df.columns or 'SMILES_CANONICAL' not in df.columns:
        print("Required columns 'SMILES' or 'SMILES_CANONICAL' not found.")
        return

    mismatches = 0
    errors = 0
    total = len(df)
    
    print(f"Checking {total} molecules...")
    
    for idx, row in tqdm(df.iterrows(), total=total):
        orig_smi = row['SMILES']
        file_canon_smi = row['SMILES_CANONICAL']
        mol_id = row.get('MOLPORTID', f'Row_{idx}')
        
        mol = Chem.MolFromSmiles(orig_smi)
        if mol is None:
            print(f"[ERROR] Invalid SMILES at {mol_id}: {orig_smi}")
            errors += 1
            continue
            
        # Generates canonical SMILES with RDKit default settings
        rdkit_canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        
        if rdkit_canon_smi != file_canon_smi:
            # Check if it matches without isomeric info, just in case
            rdkit_canon_smi_no_iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            
            if rdkit_canon_smi_no_iso == file_canon_smi:
                 print(f"[MISMATCH - ISOMERIC] {mol_id}")
                 print(f"  File Canon: {file_canon_smi}")
                 print(f"  RDKit Canon (Iso): {rdkit_canon_smi}")
                 print(f"  RDKit Canon (NoIso): {rdkit_canon_smi_no_iso}")
            else:
                 if mismatches < 10: # Print first few
                     print(f"[MISMATCH] {mol_id}")
                     print(f"  File Orig:  {orig_smi}")
                     print(f"  File Canon: {file_canon_smi}")
                     print(f"  RDKit Canon: {rdkit_canon_smi}")
            
            mismatches += 1

    print(f"\nSummary:")
    print(f"Total processed: {total}")
    print(f"Parse Errors: {errors}")
    print(f"Mismatches: {mismatches}")
    if mismatches == 0:
        print("SUCCESS: All file canonical SMILES match RDKit output.")
    else:
        print(f"FAILURE: {mismatches} mismatches found.")

if __name__ == "__main__":
    check_canonical()
