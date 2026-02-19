# MDLM 项目

本仓库包含用于分子生成和 MIC（最低抑菌浓度）预测的工具，基于 MDLM（掩码扩散语言模型）。

## MIC 预测工具

### 1. `temp_judge_generated_mols_MIC.py`

**功能**：对生成的分子进行批量 MIC 预测，并提供统计可视化。

该脚本通过预测分子针对特定细菌菌株的 MIC 值来评估其抗菌活性。它通过小提琴图提供统计分析，并将结果导出为 CSV 格式。

**特性：**
- 对 SELFIES 格式的分子进行批量 MIC 预测
- MIC 分布的小提琴图可视化
- CSV 格式导出预测结果
- 支持多种细菌菌株
- 过滤和肽序列转换

**关键配置变量：**

```python
# 设备配置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # 选择 GPU

# 模型检查点路径
ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/...'

# 要评估的细菌菌株
strains = ['11775']  # 添加菌株 ID（如 'BAA-999'、'15700'、'15697' 等）

# 包含生成分子的 SELFIES 文件路径
generate_mol_save_dir = Path('/path/to/selfies/files')

# 小提琴图输出目录
fig_save_dir = Path('/path/to/save/figures')

# CSV 结果输出目录
csv_save_dir = Path('/path/to/save/csv')

# CSV 文件名
csv_save_path = csv_save_dir / 'mic_predictions.csv'
```

**输入格式：**
- 输入文件应包含 SELFIES 格式的分子（每行一个）
- 文件命名应遵循模式：`strain_{strain_id}_..._noise.txt`

**输出：**
1. **小提琴图**：预测 MIC 值的视觉分布（保存为 PDF）
2. **CSV 文件**：包含 SELFIES/SMILES 和对应 MIC 值的结构化数据

**使用方法：**

```bash
python temp_judge_generated_mols_MIC.py
```

**依赖要求：**
- 支持 CUDA 的 PyTorch
- RDKit（用于分子处理）
- SELFIES 库
- scikit-learn
- matplotlib、seaborn、pandas

---

### 2. `temp_judge_mol_mic_with_fig.py`

**功能**：生成带有 MIC 预测标注的分子结构图像。

该脚本创建分子的视觉表示，其中每个结构图像都叠加了其预测的 MIC 值和肽序列（如适用）。它过滤高活性分子并生成有组织的输出。

**特性：**
- 2D 分子结构可视化
- 图像上直接标注 MIC 值
- 肽序列识别和显示
- 自动过滤（MIC < 15 µmol）
- 标准肽验证
- 多细菌菌株的批量处理

**关键配置变量：**

```python
# 设备配置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 模型检查点路径（相同的 MIC 回归模型）
ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/...'

# 要评估的细菌菌株
strains = ['BAA-999', '15700', '15697', '23272', '4356']
strain_show_names = ['BAA-999', '15700', '15697', '23272', '4356']

# SELFIES 文件路径
generate_mol_save_dir = Path('/path/to/selfies/files')

# 图像输出目录
img_save_dir = Path('/path/to/save/images')

# 过滤后 SELFIES 的输出目录
selfies_save_dir = Path('/path/to/save/filtered_selfies')
```

**过滤条件：**
- 仅处理 MIC < 15 µmol 的分子
- 必须是有效的标准肽（序列中不含 'X'）
- 能成功从 SELFIES 转换为 SMILES

**输出：**
1. **分子图像**：带有结构和 MIC 标注的 PNG 文件
   - 命名：`mol_{index}_mic_{value}.png`
   - 尺寸：1500x1500 像素
   - 包含 MIC 值和肽序列叠加层

2. **过滤后的 SELFIES**：每个菌株一个文本文件，包含符合条件的分子
   - 位置：`selfies_save_dir/f'strain_{strain_id}.txt'`

**使用方法：**

```bash
python temp_judge_mol_mic_with_fig.py
```

**输出结构：**
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

**依赖要求：**
- 支持 CUDA 的 PyTorch
- RDKit（用于结构绘制）
- PIL/Pillow（用于图像处理）
- SELFIES 库
- matplotlib

---

## 工作流程

典型使用流程：

1. 使用 MDLM **生成分子**（输出为 SELFIES 格式）
2. 使用 `temp_judge_generated_mols_MIC.py` **预测 MIC 值**
   - 获取统计概览
   - 导出包含所有预测的 CSV 文件
3. 使用 `temp_judge_mol_mic_with_fig.py` **可视化高活性分子**
   - 获取带标注的结构图像
   - 筛选有前景的候选分子

## 依赖库

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

## 注意事项

- 两个脚本都需要预训练的 MIC 回归模型检查点
- 必须事先准备好基因组和文本嵌入
- GPU 内存需求取决于批大小
- 推荐使用 SELFIES 格式以获得稳健的分子表示
