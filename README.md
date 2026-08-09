# MDLM Project

> **ApexOracle downstream MDLM 重构中。** 本 checkout 正在从历史 upstream-MDLM + ApexOracle
> 单文件实验集合，整理为可作为 super-repo submodule 使用的 downstream MDLM 模块。重构前源码已由
> annotated tag `legacy-code-snapshot-2026-08-09` 保护；当前不会删除数据、checkpoint 或尚未验证的
> legacy source。

- 重构阶段与验收：[`REFACTOR_PLAN.md`](REFACTOR_PLAN.md)
- 代码功能与处置审计：[`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md)
- Core/MDLM/Generation 跨仓库契约：[`docs/CROSS_REPO_CONTRACTS.md`](docs/CROSS_REPO_CONTRACTS.md)
- Legacy 恢复方法：[`docs/LEGACY_SNAPSHOT.md`](docs/LEGACY_SNAPSHOT.md)

在 canonical embedding/scoring CLI 完成前，下面的历史入口不能解释为最终 public quickstart。

[中文版](README_zh.md)

This repository contains tools for molecular generation and MIC (Minimum Inhibitory Concentration) prediction using MDLM (Masked Diffusion Language Model).

## MIC Prediction Tools

### 1. `temp_judge_generated_mols_MIC.py`

**Purpose**: Batch MIC prediction for generated molecules with statistical visualization.

This script evaluates the antimicrobial activity of generated molecules by predicting their MIC values against specific bacterial strains. It provides statistical analysis through violin plots and exports results to CSV format.

**Features:**
- Batch MIC prediction for molecules in SELFIES format
- Violin plot visualization of MIC distributions
- CSV export of prediction results
- Support for multiple bacterial strains
- Filtering and peptide sequence conversion

**Key Configuration Variables:**

```python
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # Select GPU

# Model checkpoint path
ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/...'

# Bacterial strains to evaluate
strains = ['11775']  # Add strain IDs (e.g., 'BAA-999', '15700', '15697', etc.)

# Path to SELFIES files containing generated molecules
generate_mol_save_dir = Path('/path/to/selfies/files')

# Output directory for violin plots
fig_save_dir = Path('/path/to/save/figures')

# Output directory for CSV results
csv_save_dir = Path('/path/to/save/csv')

# CSV file name
csv_save_path = csv_save_dir / 'mic_predictions.csv'
```

**Input Format:**
- Input files should contain molecules in SELFIES format (one per line)
- Files should be named following pattern: `strain_{strain_id}_..._noise.txt`

**Output:**
1. **Violin plots**: Visual distribution of predicted MIC values (saved as PDF)
2. **CSV file**: Structured data with SELFIES/SMILES and corresponding MIC values

**Usage:**

```bash
python temp_judge_generated_mols_MIC.py
```

**Requirements:**
- PyTorch with CUDA support
- RDKit (for molecule handling)
- SELFIES library
- scikit-learn
- matplotlib, seaborn, pandas

---

### 2. `temp_judge_mol_mic_with_fig.py`

**Purpose**: Generate molecular structure images with annotated MIC predictions.

This script creates visual representations of molecules where each structure image is overlaid with its predicted MIC value and peptide sequence (if applicable). It filters high-activity molecules and generates organized output.

**Features:**
- 2D molecular structure visualization
- Direct MIC value annotation on images
- Peptide sequence identification and display
- Automatic filtering (MIC < 15 µmol)
- Canonical peptide validation
- Batch processing for multiple bacterial strains

**Key Configuration Variables:**

```python
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Model checkpoint path (same MIC regression model)
ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/...'

# Bacterial strains to evaluate
strains = ['BAA-999', '15700', '15697', '23272', '4356']
strain_show_names = ['BAA-999', '15700', '15697', '23272', '4356']

# Path to SELFIES files
generate_mol_save_dir = Path('/path/to/selfies/files')

# Output directory for images
img_save_dir = Path('/path/to/save/images')

# Output directory for filtered SELFIES
selfies_save_dir = Path('/path/to/save/filtered_selfies')
```

**Filtering Criteria:**
- Only molecules with MIC < 15 µmol are processed
- Must be valid canonical peptides (no 'X' in sequence)
- Successfully convertible from SELFIES to SMILES

**Output:**
1. **Molecular images**: PNG files with structure and MIC annotations
   - Naming: `mol_{index}_mic_{value}.png`
   - Size: 1500x1500 pixels
   - Includes MIC value and peptide sequence overlay

2. **Filtered SELFIES**: Text file per strain with qualified molecules
   - Location: `selfies_save_dir/f'strain_{strain_id}.txt'`

**Usage:**

```bash
python temp_judge_mol_mic_with_fig.py
```

**Output Structure:**
```
/path/to/save/images/
└── strain_{strain_id}/
    ├── mol_0_mic_1.23.png
    ├── mol_1_mic_2.45.png
    └── ...

/path/to/save/filtered_selfies/
├── strain_BAA-999.txt
├── strain_15700.txt
└── ...
```

**Requirements:**
- PyTorch with CUDA support
- RDKit (for structure drawing)
- PIL/Pillow (for image processing)
- SELFIES library
- matplotlib

---

## Workflow

Typical usage workflow:

1. **Generate molecules** using MDLM (output in SELFIES format)
2. **Predict MIC values** using `temp_judge_generated_mols_MIC.py`
   - Get statistical overview
   - Export CSV with all predictions
3. **Visualize high-activity molecules** using `temp_judge_mol_mic_with_fig.py`
   - Get annotated structure images
   - Filter for promising candidates

## Dependencies

```bash
pip install torch torchvision
pip install rdkit-pypi
pip install selfies
pip install transformers
pip install scikit-learn
pip install matplotlib seaborn pandas
pip install pillow
pip install biopython
pip install hydra-core
pip install tqdm
```

## Notes

- Both scripts require pre-trained MIC regression model checkpoints
- Genome and text embeddings must be prepared beforehand
- GPU memory requirements depend on batch size
- SELFIES format is preferred for robust molecule representation
