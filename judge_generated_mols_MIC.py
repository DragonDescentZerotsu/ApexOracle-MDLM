from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
from Bio import Phylo
from triton.language import bfloat16
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import json
import itertools
import logging

import hydra
from hydra import compose, initialize
import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter, FuncFormatter
from matplotlib.patches import Patch, PathPatch
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import ast
import seaborn as sns
from guaidance_regressor_all_data import FirstTokenAttention_genome, RegressionHead, load_all_genome_embeddings, load_text_wo_genome_embeddings

current_directory = Path('/data2/tianang/projects/Synergy')
CACHE_DIR = Path(__file__).resolve().parent / 'temp_data' / 'temp_precomputed_MIC_for_figs'

GROUP_ORDER = ['Unconditional', 'Guided']
GROUP_OFFSETS = [-0.15, 0.15]
GROUP_STYLE = {
    'Unconditional': {
        'facecolor': "#000000",
        'edgecolor': "#000000",
    },
    'Guided': {
        'facecolor': "#F279AB",
        'edgecolor': "#F279AB",
    },
}

with initialize(config_path="configs"):
    config = compose(config_name="config")

class mol_emb_mdlm(nn.Module):
    def __init__(self, config, vocab_size, ckpt_path):
        super(mol_emb_mdlm, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.ckpt_path = ckpt_path
        self.parameterization = self.config.parameterization
        self.time_conditioning = self.config.time_conditioning
        self.backbone = self.load_DIT()  # hidden_size = 768
        # print(self.bert.config.max_position_embeddings)
        self.noise = noise_schedule.get_noise(self.config)

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, device):
        sampling_eps = 1e-3
        _eps_t = torch.rand(n, device=device)
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        return t * 0

    def _forward(self, x, sigma):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.backbone.vocab_embed(x)
            c = F.silu(self.backbone.sigma_map(sigma))
            rotary_cos_sin = self.backbone.rotary_emb(x)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x, rotary_cos_sin, c, seqlens=None)

        return x


    def forward(self, input_ids, attention_mask=None):
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        outputs = self._forward(input_ids, unet_conditioning)
        return outputs

    def load_DIT(self):
        backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        lightning_ckpt = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = lightning_ckpt['state_dict']

        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
            else:
                new_key = k
            new_sd[new_key] = v

        backbone.load_state_dict(new_sd, strict=False)

        return backbone

class mol_emb_mdlm_no_weights(mol_emb_mdlm):
    """
    重写 load_DIT 的部分消去对路径的依赖
    """
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size, ckpt_path=None)

    def load_DIT(self):
        backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        return backbone


class MIC_regressor(nn.Module):
    """
    融合了 mdlm 和所有的 genome text emb 的部分
    """

    def __init__(self, config, ckpt_path, device):
        super(MIC_regressor, self).__init__()
        self.config = config
        # self.strain_cond = strain_cond
        self.ckpt_path = ckpt_path
        self.device = device
        self.mdlm_model: nn.Module = mol_emb_mdlm_no_weights(config, len(tokenizer.get_vocab()))
        self.co_cross_attn_genome = FirstTokenAttention_genome(self.mdlm_model.config.model.hidden_size, 8192, 4, 0.1)
        self.co_cross_attn_text = FirstTokenAttention_genome(self.mdlm_model.config.model.hidden_size, 4096, 4, 0.1)
        self.reg_head = RegressionHead(8192 + 4096, (8192 + 4096) // 4, 128, 1, 0.2)
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192, device=device))
        self.load_pretrained_weight()

        self.ATCC_genome_emb_dict, self.ATCC_text_emb_dict, self.text_only_emb_dict = self.load_genome_test_embedding()

    def load_genome_test_embedding(self):
        ATCC_genome_emb_dict = load_all_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Genome_embs'),
            1e14,
            'cpu',
            'ATCC genome embedding')
        ATCC_text_emb_dict = load_all_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Text_Description'/'ATCC'/'embeddings'),
            1,
            'cpu',
            'ATCC text embedding')
        text_only_emb_dict = load_text_wo_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Text_Description'/'wo_ATCC'/'embeddings'),
            1,
            'cpu',
            'text only embedding')
        return ATCC_genome_emb_dict, ATCC_text_emb_dict, text_only_emb_dict

    def load_pretrained_weight(self):
        """
        在使用 forward 之前进行，load 自己要用的权重
        """
        regressor_ckpt_path = self.ckpt_path
        checkpoint = torch.load(regressor_ckpt_path, map_location=self.device)
        # new_sd = OrderedDict()
        # for k, v in checkpoint['mdlm_model_state_dict'].items():
        #     if k.startswith('backbone.'):
        #         new_key = k[len('backbone.'):]
        #     else:
        #         new_key = k
        #     new_sd[new_key] = v
        self.mdlm_model.load_state_dict(checkpoint['mdlm_model_state_dict'])
        self.reg_head.load_state_dict(checkpoint['re_head_state_dict'])
        self.co_cross_attn_genome.load_state_dict(checkpoint['co_cross_attn_genome'])
        self.co_cross_attn_text.load_state_dict(checkpoint['co_cross_attn_text'])
        self.learnable_embedding_weight = checkpoint['learnable_embedding_weight']

    def forward(self, input_ids: torch.Tensor, strain_cond):
        output = self.mdlm_model(input_ids)
        mol_cls_embedding = output[:, 0, :]
        if strain_cond in self.ATCC_genome_emb_dict.keys():
            padded_genome_embeddings = self.ATCC_genome_emb_dict[strain_cond].to(device)
            padded_text_embeddings = self.ATCC_text_emb_dict[strain_cond].to(device)
            padded_genome_embeddings = padded_genome_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
            genome_attn_masks = torch.ones(padded_genome_embeddings.shape[0], padded_genome_embeddings.shape[1]).to(device)
        else:
            padded_genome_embeddings = None
            padded_text_embeddings = self.text_only_emb_dict[strain_cond].to(device)
        padded_text_embeddings = padded_text_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
        text_attn_masks = torch.ones(padded_text_embeddings.shape[0], padded_text_embeddings.shape[1]).to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if padded_genome_embeddings is not None:
                mol_cls_embedding_genome, _ = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            else:
                padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
                mol_cls_embedding_genome, _ = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            mol_cls_embedding_text, _ = self.co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
            reg_logits = self.reg_head(mol_cls_embedding)

        return reg_logits

def find_matching_generated_file(file_names, strain, target_MIC, guidance_method, target_length):
    for file_name in file_names:
        file_strain = file_name.split('.txt')[0].split('_')[1]
        file_target_MIC = file_name.split('.txt')[0].split('_')[3]
        try:
            file_guidance_method = file_name.split('.txt')[0].split('_')[-1]
            file_target_length = file_name.split('.txt')[0].split('_')[5]
        except:
            continue
        if strain == file_strain and file_target_MIC == target_MIC and file_guidance_method == guidance_method and file_target_length == target_length:
            return file_name
    return None


def get_mic_cache_path(generate_mol_save_dir, strain, target_MIC, guidance_method, target_length, ckpt_path):
    cache_key = (
        f"{generate_mol_save_dir.resolve()}::"
        f"{strain}::{target_MIC}::{guidance_method}::{target_length}::{Path(ckpt_path).resolve()}"
    )
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_name = f"{strain}_MIC_{target_MIC}_length_{target_length}_{guidance_method}_{cache_hash}.pt"
    return CACHE_DIR / cache_name


def get_mic(generate_mol_save_dir, file_names, mic_regressor, tokenizer, strain, target_MIC, guidance_method, target_length, ckpt_path):
    matched_file_name = find_matching_generated_file(file_names, strain, target_MIC, guidance_method, target_length)
    if matched_file_name is None:
        raise FileNotFoundError(
            f"Could not find generated molecules for strain={strain}, target_MIC={target_MIC}, "
            f"guidance_method={guidance_method}, target_length={target_length} in {generate_mol_save_dir}"
        )

    cache_path = get_mic_cache_path(generate_mol_save_dir, strain, target_MIC, guidance_method, target_length, ckpt_path)
    if cache_path.exists():
        cached_payload = torch.load(cache_path, map_location='cpu')
        print(f'Loaded cached MICs from {cache_path}')
        return cached_payload['mics'].to(torch.float32)

    if mic_regressor is None or tokenizer is None:
        raise RuntimeError(f'MIC cache miss for {strain} / {target_MIC} / {target_length}, but model is not initialized.')

    with open(generate_mol_save_dir / matched_file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    SELFIES_strs = [line.rstrip("\n") for line in lines]

    input_ids = tokenizer(
        [s.replace('][', '] [') for s in SELFIES_strs],
        return_tensors='pt',
        padding=True,
        truncation=False,
        add_special_tokens=True
    )['input_ids'].squeeze().to(device)

    mics = []
    i=0
    step_size = 1  # 1 / 100
    while i<len(input_ids):

        batch_input_ids = input_ids[i:i+step_size]
        if step_size == 1:
            batch_input_ids = batch_input_ids[batch_input_ids!=tokenizer.pad_token_id].unsqueeze(0)
        mic_logits = mic_regressor(batch_input_ids, strain)
        mic_logits = mic_logits.squeeze().detach()
        mic = 10 ** (-mic_logits) * 10

        if step_size == 1:
            mic = mic.cpu().to(torch.float)
        mics.append(mic)

        i += step_size

    if step_size != 1:
        mics = torch.concat(mics, dim = 0)
    else:
        mics = torch.from_numpy(np.array(mics))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'mics': mics.cpu().to(torch.float32),
            'strain': strain,
            'target_MIC': target_MIC,
            'guidance_method': guidance_method,
            'target_length': target_length,
            'generate_mol_save_dir': str(generate_mol_save_dir),
            'source_file': matched_file_name,
            'ckpt_path': ckpt_path,
        },
        cache_path,
    )
    print(f'Saved MIC cache to {cache_path}')

    return mics


def draw_significance(ax, x1, x2, y_top, left_drop, right_drop, text, text_y):
    ax.plot(
        [x1, x1, x2, x2],
        [y_top - left_drop, y_top, y_top, y_top - right_drop],
        color="#222222",
        linewidth=1.3,
        solid_capstyle="butt",
        clip_on=False,
        zorder=4,
    )
    ax.text(
        (x1 + x2) / 2,
        text_y,
        text,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def format_p_value(p_value):
    if p_value < 1e-4:
        return "p < 1e-4"
    return f"p = {p_value:.4f}"


def add_p_value_annotation(ax, x1, x2, left_values, right_values):
    p_value = mannwhitneyu(left_values, right_values, alternative="two-sided").pvalue

    combined_values = np.concatenate((left_values, right_values))
    y_min = float(np.min(combined_values))
    y_max = float(np.max(combined_values))
    y_span = max(y_max - y_min, 1.0)
    y_top = y_max + 0.08 * y_span
    drop = 0.04 * y_span
    text_y = y_top + 0.015 * y_span

    draw_significance(
        ax,
        x1,
        x2,
        y_top,
        drop,
        drop,
        format_p_value(p_value),
        text_y,
    )

    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(current_bottom, max(current_top, text_y + 0.08 * y_span))


def style_violin_body(body, label):
    style = GROUP_STYLE[label]
    body.set_facecolor(mcolors.to_rgba(style["facecolor"], alpha=0.55))
    body.set_edgecolor('none')
    body.set_linewidth(0.0)
    body.set_antialiased(False)


def add_violin_outline(ax, body, label):
    outline = PathPatch(
        body.get_paths()[0],
        facecolor='none',
        edgecolor=GROUP_STYLE[label]["edgecolor"],
        linewidth=1.4,
        joinstyle='round',
        capstyle='round',
        antialiased=True,
        zorder=body.get_zorder() + 0.2,
    )
    ax.add_patch(outline)


def violin_span_at_y(body, y_value):
    path = body.get_paths()[0]
    vertices = path.vertices
    x_intersections = []

    for (x1, y1), (x2, y2) in zip(vertices[:-1], vertices[1:]):
        if y1 == y2:
            if y_value == y1:
                x_intersections.extend([x1, x2])
            continue
        if min(y1, y2) <= y_value <= max(y1, y2):
            ratio = (y_value - y1) / (y2 - y1)
            x_intersections.append(x1 + ratio * (x2 - x1))

    if len(x_intersections) < 2:
        center_x = np.mean(vertices[:, 0])
        return center_x, center_x

    return min(x_intersections), max(x_intersections)


def add_distribution_summary(ax, body, values, label):
    edgecolor = GROUP_STYLE[label]["edgecolor"]
    median = np.percentile(values, 50)
    median_x_min, median_x_max = violin_span_at_y(body, median)
    ax.hlines(
        median,
        median_x_min,
        median_x_max,
        colors=edgecolor,
        linestyles=(0, (1, 1)),
        linewidth=1.2,
        zorder=3,
    )

if __name__ == '__main__':

    show_log2 = True

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/guidance_best_R2_all_peptide_epoch_12.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/noise_guidance_all_peptide_epoch_100_of_100.pth'

    guidance_method = 'noise'  # clean / noise
    target_MICs = ['1', '1000']  # '1000' 表示 unconditional generation
    strain_plot_configs = [
        {
            'strain': 'BAA-3170',
            'label': r'$\it{E.\ coli}$ AR-0349',
            'center': 0.0,
            'length': '368',
            'generate_mol_save_dir': Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES-new-test'),
        },
        {
            'strain': 'BAA-3197',
            'label': r'$\it{P.\ aeruginosa}$ PA5257',
            'center': 0.8,
            'length': '232',
            'generate_mol_save_dir': Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES'),
        },
    ]

    target_label_map = {
        '1': 'Guided',
        '1000': 'Unconditional',
    }

    all_cache_ready = True
    for strain_config in strain_plot_configs:
        generate_mol_save_dir = strain_config['generate_mol_save_dir']
        file_names = [file.name for file in generate_mol_save_dir.iterdir()]
        for target_MIC in target_MICs:
            cache_path = get_mic_cache_path(
                generate_mol_save_dir,
                strain_config['strain'],
                target_MIC,
                guidance_method,
                strain_config['length'],
                ckpt_path,
            )
            matched_file_name = find_matching_generated_file(
                file_names,
                strain_config['strain'],
                target_MIC,
                guidance_method,
                strain_config['length'],
            )
            if matched_file_name is None or not cache_path.exists():
                all_cache_ready = False
                break
        if not all_cache_ready:
            break

    tokenizer = None
    mic_regressor = None
    if all_cache_ready:
        print(f'Using precomputed MIC caches from {CACHE_DIR}')
    else:
        model_name = "ibm-research/materials.selfies-ted"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mic_regressor = MIC_regressor(config, ckpt_path, device)
        mic_regressor.to(device)
        mic_regressor.eval()

    fig, ax = plt.subplots(figsize=(5, 5))

    for strain_config in strain_plot_configs:
        generate_mol_save_dir = strain_config['generate_mol_save_dir']
        file_names = [file.name for file in generate_mol_save_dir.iterdir()]
        grouped_plot_values = {}

        for target_MIC in target_MICs:
            mic = get_mic(
                generate_mol_save_dir,
                file_names,
                mic_regressor,
                tokenizer,
                strain_config['strain'],
                target_MIC,
                guidance_method,
                strain_config['length'],
                ckpt_path,
            )

            print(f"strain: {strain_config['strain']}")
            print(mic)
            print(f' target MIC: {target_MIC}')
            print(f' predicted mean MIC: {mic.mean()}')
            print(f' predicted median MIC: {torch.median(mic)}')

            top_k = 1
            min_values, _ = torch.topk(mic, top_k, largest=False)
            print(f'{top_k} min mics: {min_values}')
            print(f' mean: {min_values.mean()}')
            print(f' median: {min_values.median()}')

            label = target_label_map[target_MIC]
            if show_log2:
                grouped_plot_values[label] = torch.log2(mic).detach().cpu().to(torch.float32).numpy()
            else:
                grouped_plot_values[label] = mic.detach().cpu().to(torch.float32).numpy()

        x_positions = {
            label: strain_config['center'] + offset
            for label, offset in zip(GROUP_ORDER, GROUP_OFFSETS)
        }

        for label in GROUP_ORDER:
            values = grouped_plot_values[label]
            parts = ax.violinplot(
                [values],
                positions=[x_positions[label]],
                widths=0.22,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.35,
                points=300,
            )
            violin_body = parts["bodies"][0]
            style_violin_body(violin_body, label)
            add_violin_outline(ax, violin_body, label)
            add_distribution_summary(ax, violin_body, values, label)

        add_p_value_annotation(
            ax,
            x_positions['Unconditional'],
            x_positions['Guided'],
            grouped_plot_values['Unconditional'],
            grouped_plot_values['Guided'],
        )

    def inv_log2(x, pos):
        return int(2 ** x)

    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.3)
    ax.set_axisbelow(True)

    if show_log2:
        ax.yaxis.set_major_formatter(FuncFormatter(inv_log2))

    centers = [config['center'] for config in strain_plot_configs]
    ax.set_xlim(min(centers) + min(GROUP_OFFSETS) - 0.18, max(centers) + max(GROUP_OFFSETS) + 0.18)
    ax.set_xticks(centers)
    ax.set_xticklabels([config['label'] for config in strain_plot_configs], fontsize=13)
    ax.set_xlabel("")
    y_label = ax.set_ylabel("log 2 scale MIC value (µmol)")
    y_label.set_fontsize(14)
    ax.set_title("Generated Molecule MIC Distribution", fontsize=14)

    legend_handles = [
        Patch(
            facecolor=GROUP_STYLE[label]["facecolor"],
            edgecolor=GROUP_STYLE[label]["edgecolor"],
            linewidth=2.8,
            alpha=0.65,
            label=label,
        )
        for label in GROUP_ORDER
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="center left", bbox_to_anchor=(0.5, 0.1), fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.tick_params(axis='y', labelsize=10, width=1.6, length=6)
    ax.tick_params(axis='x', labelsize=13, width=1.6, length=6)

    plt.tight_layout()
    plt.savefig("/data2/tianang/projects/Synergy/paper_figs/3170-3197-guidance-MIC.pdf", format="pdf", bbox_inches="tight")
    plt.show()

        # fig, ax = plt.subplots()
        # ax.violinplot(mic_list)
        # ax.set_xticks(list(range(1, len(mic_list) + 1)))
        # ax.set_xticklabels(ax_labels[:len(mic_list)])
        # ax.set_title(f'{strain_show_name} generated molecule MIC, {guidance_method} guidance')
        # ax.set_ylabel('log 2 scale MIC value (µmol)')
        #
        # # 4. 如果你绘的是 log2 转换后的值，就不需要再转；否则可以用对数坐标轴
        #
        #
        # # ax.set_yscale('log', base=2)
        # # ax.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
        # # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        #
        # def inv_log2(x, pos):
        #     orig = 2 ** x
        #     # 如果数值足够大，改成科学记数法或保留整数
        #     # if orig >= 1000:
        #     #     return f"{orig:.1e}"
        #     # else:
        #     #     return f"{orig:.0f}"
        #     return orig
        #
        #
        # # 3) 应用到 y 轴主刻度
        # if show_log2:
        #     ax.yaxis.set_major_formatter(FuncFormatter(inv_log2))

        # plt.show()
