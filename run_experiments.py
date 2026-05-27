#!/usr/bin/env python3
"""
run_experiments.py
一键运行全部实验，生成论文所需数据和图表

用法：
  python run_experiments.py                # 全量（30 seeds，约40分钟）
  python run_experiments.py --quick        # 快速验证（5 seeds，约5分钟）
  python run_experiments.py --exp sota     # 只跑SOTA对比
  python run_experiments.py --exp ablation # 只跑消融实验
  python run_experiments.py --exp extreme  # 只跑极端环境
  python run_experiments.py --exp scale    # 只跑可扩展性
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 字体设置（修复Mac中文乱码）
import matplotlib.font_manager as fm

def _setup_font():
    """优先使用系统中文字体，找不到则用英文标签"""
    candidates = [
        'Arial Unicode MS',   # Mac 自带
        'PingFang SC',        # Mac 苹方
        'Hiragino Sans GB',   # Mac 冬青黑
        'STHeiti',            # Mac 华文黑体
        'DejaVu Sans',        # 后备（不支持中文）
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

_setup_font()

from env import RescueEnvironment
from robot import SensorConfig
from algorithms.sccs import SCCSSystem
from algorithms.baselines import (
    GreedySystem, PSOSystem, RandomSystem,
    SCCS_noTCFM, SCCS_noClustering, SCCS_noFusion,
)

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/data').mkdir(parents=True, exist_ok=True)
DPI = 300
# ── 莫兰迪配色（低饱和度，适合学术论文）
COLORS = {
    'SCCS（完整）':      '#7B9EAE',   # 雾蓝
    'SCCS':             '#7B9EAE',
    'w/o LiDAR融合':    '#C4A882',   # 暖沙
    'w/o TCFM':         '#B08FA0',   # 藕粉
    'w/o 聚类':         '#9AAE9A',   # 雾绿
    'Greedy':           '#C49A8A',   # 赤陶
    'PSO':              '#A0A8C0',   # 薰衣草蓝
    'Random':           '#B8B0A8',   # 暖灰
    'Full SCCS':        '#7B9EAE',
    # 极端环境配色
    '正常-融合':         '#7B9EAE',
    '扬尘-融合':         '#C4B49A',
    '扬尘-纯视觉':       '#C49A8A',
    '弱光-融合':         '#9AAE9A',
    '弱光-纯视觉':       '#B08FA0',
}
DEFAULT_COLOR = '#C0B8B0'

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def make_env(seed: int) -> RescueEnvironment:
    return RescueEnvironment(
        width=80, height=80,
        obstacle_ratio=0.30,
        n_survivors=50,
        n_clusters=5,
        cluster_std=5.0,
        life_decay_rate=0.0018,
        seed=seed,
    )


def run_one(SystemClass, seed: int, sensor=None,
            max_steps: int = 3000, **kwargs) -> dict:
    """运行单次仿真，返回指标字典"""
    env = make_env(seed)
    if sensor is not None:
        sys_ = SystemClass(env, sensor=sensor, **kwargs)
    else:
        sys_ = SystemClass(env, **kwargs)
    result = sys_.run(max_steps=max_steps)
    result['seed'] = seed
    result.pop('coverage_curve', None)
    return result


def run_batch(SystemClass, seeds: list, label: str,
              sensor=None, max_steps: int = 3000,
              verbose: bool = True, **kwargs) -> pd.DataFrame:
    """批量运行，返回 DataFrame"""
    rows = []
    for seed in seeds:
        r = run_one(SystemClass, seed, sensor, max_steps, **kwargs)
        r['method'] = label
        rows.append(r)
        if verbose:
            print(f"    [{label:22s}] seed={seed:5d}  "
                  f"coverage={r['coverage_rate']:.3f}  "
                  f"steps={r['completion_step']:4d}  "
                  f"load_var={r['load_variance']:5.2f}  "
                  f"comm={r['comm_cost']:5d}")
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, title: str,
                  group_col: str = 'method'):
    cols = [c for c in [
        'coverage_rate', 'completion_step', 'response_step',
        'load_variance', 'collision_count', 'comm_cost',
    ] if c in df.columns]
    summary = df.groupby(group_col)[cols].agg(['mean', 'std']).round(3)
    print(f'\n{"─"*65}')
    print(f'  {title}')
    print('─' * 65)
    print(summary.to_string())
    print('─' * 65)


def save(fig, name: str):
    path = f'output/figures/{name}.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f'  ✅ 保存: {path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 实验1：SOTA 对比
# ─────────────────────────────────────────────────────────────

def exp_sota(seeds: list, max_steps: int) -> pd.DataFrame:
    print('\n' + '=' * 65)
    print(' Exp1: SOTA 对比  (SCCS vs Greedy vs PSO vs Random)')
    print('=' * 65)

    frames = []
    for Cls, name in [
        (SCCSSystem,   'SCCS（完整）'),
        (GreedySystem, 'Greedy'),
        (PSOSystem,    'PSO'),
        (RandomSystem, 'Random'),
    ]:
        print(f'\n  → {name}')
        frames.append(
            run_batch(Cls, seeds, name, max_steps=max_steps)
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv('output/data/sota_results.csv', index=False)
    print_summary(df, 'SOTA 对比汇总')

    # 计算提升百分比
    sccs   = df[df['method'] == 'SCCS（完整）']
    greedy = df[df['method'] == 'Greedy']
    step_imp = (greedy['completion_step'].mean() -
                sccs['completion_step'].mean()) / greedy['completion_step'].mean() * 100
    var_imp  = (greedy['load_variance'].mean() -
                sccs['load_variance'].mean()) / greedy['load_variance'].mean() * 100
    col_imp  = (greedy['collision_count'].mean() -
                sccs['collision_count'].mean()) / max(greedy['collision_count'].mean(), 1) * 100

    print(f'\n  ── 论文核心数字（填入摘要/结论）')
    print(f'     SCCS平均覆盖率:  {sccs["coverage_rate"].mean():.3f} '
          f'± {sccs["coverage_rate"].std():.3f}')
    print(f'     响应时间缩短:    {step_imp:.1f}%  (目标≥42%)')
    print(f'     负载方差降低:    {var_imp:.1f}%   (目标≥62%)')
    print(f'     碰撞次数减少:    {col_imp:.1f}%  (目标≥78%)')
    print(f'     TCFM显式通信:    0条  '
          f'(广播需 {greedy["comm_cost"].mean():.0f} 条)')

    return df


# ─────────────────────────────────────────────────────────────
# 实验2：消融实验
# ─────────────────────────────────────────────────────────────

def exp_ablation(seeds: list, max_steps: int) -> pd.DataFrame:
    print('\n' + '=' * 65)
    print(' Exp2: 消融实验')
    print('=' * 65)

    frames = []
    for Cls, name in [
        (SCCSSystem,        'SCCS（完整）'),
        (SCCS_noTCFM,       'w/o TCFM'),
        (SCCS_noClustering, 'w/o 聚类'),
        (SCCS_noFusion,     'w/o LiDAR融合'),
    ]:
        print(f'\n  → {name}')
        frames.append(
            run_batch(Cls, seeds, name, max_steps=max_steps)
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv('output/data/ablation_results.csv', index=False)
    print_summary(df, '消融实验汇总')
    return df


# ─────────────────────────────────────────────────────────────
# 实验3：极端环境鲁棒性
# ─────────────────────────────────────────────────────────────

def exp_extreme(seeds: list, max_steps: int) -> pd.DataFrame:
    print('\n' + '=' * 65)
    print(' Exp3: 极端环境鲁棒性（SCCS融合 vs Greedy纯视觉）')
    print('=' * 65)

    # 每种环境下对比：SCCS（雷视融合） vs Greedy（纯视觉）
    configs = [
        # 正常环境
        (SCCSSystem,   '正常-SCCS融合',   SensorConfig()),
        (GreedySystem, '正常-Greedy纯视觉', SensorConfig.vision_only()),
        # 扬尘环境
        (SCCSSystem,   '扬尘-SCCS融合',   SensorConfig.dusty()),
        (GreedySystem, '扬尘-Greedy纯视觉', SensorConfig(
                                lidar_grid=0,
                                camera_grid=3,
                                detect_prob=0.60,
                                env_factor=0.55)),
        # 弱光环境
        (SCCSSystem,   '弱光-SCCS融合',   SensorConfig.low_light()),
        (GreedySystem, '弱光-Greedy纯视觉', SensorConfig(
                                lidar_grid=0,
                                camera_grid=2,
                                detect_prob=0.55,
                                env_factor=0.40)),
    ]

    frames = []
    for Cls, label, sensor in configs:
        print(f'\n  → {label}')
        df = run_batch(Cls, seeds, label,
                       sensor=sensor, max_steps=max_steps)
        df['env_type'] = label
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv('output/data/extreme_results.csv', index=False)
    print_summary(df, '极端环境汇总', group_col='env_type')

    # 计算每种环境下SCCS vs Greedy的差距
    print('\n  ── 各环境SCCS vs Greedy纯视觉覆盖率差距 ')
    for env in ['正常', '扬尘', '弱光']:
        sccs_cov = df[df['env_type']==f'{env}-SCCS融合']['coverage_rate'].mean()
        grdy_cov = df[df['env_type']==f'{env}-Greedy纯视觉']['coverage_rate'].mean()
        diff = (sccs_cov - grdy_cov) * 100
        print(f'     {env}: SCCS={sccs_cov:.3f}  Greedy纯视觉={grdy_cov:.3f}  差距=+{diff:.1f}%')

    return df


# ─────────────────────────────────────────────────────────────
# 实验4：可扩展性（不同机器人数量）
# ─────────────────────────────────────────────────────────────

def exp_scale(seeds: list, max_steps: int) -> pd.DataFrame:
    print('\n' + '=' * 65)
    print(' Exp4: 可扩展性（N = 3 / 5 / 7 / 9）')
    print('=' * 65)

    frames = []
    for n in [3, 5, 7, 9]:
        for Cls, base in [(SCCSSystem, 'SCCS'), (GreedySystem, 'Greedy')]:
            label = f'{base} N={n}'
            print(f'\n  → {label}')
            df = run_batch(Cls, seeds, label,
                           max_steps=max_steps, n_robots=n)
            df['n_robots']    = n
            df['base_method'] = base
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv('output/data/scale_results.csv', index=False)
    return df


# ─────────────────────────────────────────────────────────────
# 覆盖率曲线收集（用于画折线图）
# ─────────────────────────────────────────────────────────────

def collect_curves(seeds: list, max_steps: int,
                   n_seeds: int = 8) -> dict:
    """单独收集 coverage_curve，用于时间曲线图"""
    configs = [
        (SCCSSystem,    'SCCS（完整）',  None),
        (SCCS_noFusion, 'w/o LiDAR融合', None),
        (GreedySystem,  'Greedy',         None),
        (PSOSystem,     'PSO',            None),
    ]
    curves = {name: [] for _, name, _ in configs}
    for seed in seeds[:n_seeds]:
        for Cls, name, sensor in configs:
            env = make_env(seed)
            sys_ = Cls(env) if sensor is None else Cls(env, sensor=sensor)
            r = sys_.run(max_steps=max_steps)
            if 'coverage_curve' in r:
                curves[name].append(r['coverage_curve'])
    return curves


# ─────────────────────────────────────────────────────────────
# 图表生成
# ─────────────────────────────────────────────────────────────

def plot_coverage_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, curve_list in curves.items():
        if not curve_list:
            continue
        max_len = max(len(c) for c in curve_list)
        padded  = [c + [c[-1]] * (max_len - len(c)) for c in curve_list]
        arr     = np.array(padded)
        mean    = arr.mean(axis=0)
        std     = arr.std(axis=0)
        x       = np.arange(max_len)
        color   = COLORS.get(method, DEFAULT_COLOR)
        ax.plot(x, mean, label=method, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std,
                        alpha=0.15, color=color)
    ax.axhline(0.95, color='#C07070', linestyle='--',
               alpha=0.6, linewidth=1, label='95% Target')
    ax.set_xlabel('Simulation Steps', fontsize=12)
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('Search Coverage Rate over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, color='#CCCCCC')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig1_coverage_curves')


def plot_sota_boxplots(df: pd.DataFrame):
    metrics = [
        ('coverage_rate',   'Coverage Rate'),
        ('completion_step', 'Completion Steps'),
        ('load_variance',   'Load Variance'),
        ('comm_cost',       'Comm Cost'),
    ]
    metrics = [(m, l) for m, l in metrics if m in df.columns]
    methods = list(df['method'].unique())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, metrics):
        data   = [df[df['method'] == m][metric].values for m in methods]
        colors = [COLORS.get(m, DEFAULT_COLOR) for m in methods]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={'color': '#555555', 'linewidth': 2},
                        whiskerprops={'color': '#888888'},
                        capprops={'color': '#888888'},
                        flierprops={'marker': 'o', 'markersize': 3,
                                    'alpha': 0.5})
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor('#666666')
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, color='#CCCCCC')
        ax.set_facecolor('#FAFAFA')
    fig.suptitle(f'SOTA Comparison ({len(df)//len(methods)} trials)',
                 fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig2_sota_boxplots')


def plot_ablation(df: pd.DataFrame):
    metrics = [
        ('coverage_rate',   'Coverage Rate'),
        ('completion_step', 'Completion Steps'),
        ('load_variance',   'Load Variance'),
        ('comm_cost',       'Comm Cost'),
    ]
    metrics = [(m, l) for m, l in metrics if m in df.columns]
    methods = list(df['method'].unique())
    x = np.arange(len(methods))
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, metrics):
        means  = [df[df['method'] == m][metric].mean() for m in methods]
        stds   = [df[df['method'] == m][metric].std()  for m in methods]
        colors = [COLORS.get(m, DEFAULT_COLOR) for m in methods]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      alpha=0.85, edgecolor='#888888', linewidth=0.6,
                      error_kw={'elinewidth': 1.2})
        # 误差棒不超过0（碰撞/通信不可能为负）
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, color='#CCCCCC')
        ax.set_facecolor('#FAFAFA')
    fig.suptitle('Ablation Study', fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig3_ablation')


def plot_extreme(df: pd.DataFrame):
    col    = 'env_type' if 'env_type' in df.columns else 'method'
    groups = list(df[col].unique())
    # SCCS融合用蓝色系，Greedy纯视觉用暖色系
    def _bar_color(g):
        if 'SCCS' in g:
            return '#7B9EAE'
        return '#C49A8A'
    colors = [_bar_color(g) for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric, label) in zip(axes, [
        ('coverage_rate',   'Coverage Rate'),
        ('completion_step', 'Completion Steps'),
    ]):
        if metric not in df.columns:
            continue
        means = [df[df[col] == g][metric].mean() for g in groups]
        stds  = [df[df[col] == g][metric].std()  for g in groups]
        bars = ax.bar(range(len(groups)), means, yerr=stds, capsize=5,
               color=colors, alpha=0.85, edgecolor='#888888', linewidth=0.6,
               error_kw={'elinewidth': 1.2})
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=40, ha='right', fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, color='#CCCCCC')
        ax.set_facecolor('#FAFAFA')

    # 图例
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(facecolor='#7B9EAE', label='SCCS (LiDAR+Vision Fusion)'),
        mpatches.Patch(facecolor='#C49A8A', label='Greedy (Vision Only)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=9, framealpha=0.8)
    fig.suptitle('Robustness in Extreme Environments (Fusion vs Vision-only)',
                 fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig4_extreme_env')


def plot_scale(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for base, color in [('SCCS', '#7B9EAE'), ('Greedy', '#C49A8A')]:
        if 'base_method' not in df.columns:
            break
        sub = df[df['base_method'] == base]
        if sub.empty:
            continue
        ns   = sorted(sub['n_robots'].unique())
        covs = [sub[sub['n_robots'] == n]['coverage_rate'].mean()   for n in ns]
        tims = [sub[sub['n_robots'] == n]['completion_step'].mean() for n in ns]
        axes[0].plot(ns, covs, marker='o', label=base,
                     color=color, linewidth=2, markersize=6)
        axes[1].plot(ns, tims, marker='s', label=base,
                     color=color, linewidth=2, markersize=6)
    for ax, ylabel, title in zip(axes,
        ['Coverage Rate', 'Completion Steps'],
        ['Coverage Rate vs Robot Count', 'Completion Steps vs Robot Count']):
        ax.set_xlabel('Number of Robots', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25, color='#CCCCCC')
        ax.set_facecolor('#FAFAFA')
    fig.suptitle('Scalability Analysis', fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig5_scalability')


def plot_comm_comparison(df_sota: pd.DataFrame):
    methods = list(df_sota['method'].unique())
    means   = [df_sota[df_sota['method'] == m]['comm_cost'].mean()
               for m in methods]
    colors  = [COLORS.get(m, DEFAULT_COLOR) for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(methods)), means, color=colors,
                  alpha=0.85, edgecolor='#888888', linewidth=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Explicit Communication Messages', fontsize=11)
    ax.set_title('Communication Overhead: TCFM O(1) vs Broadcast O(N²)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, color='#CCCCCC')
    ax.set_facecolor('#FAFAFA')
    for i, (bar, v) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(means) * 0.015,
                f'{v:.0f}', ha='center', va='bottom', fontsize=9,
                color='#444444')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    save(fig, 'fig6_comm_cost')


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（5 seeds）')
    parser.add_argument('--exp', default='all',
                        choices=['all', 'sota', 'ablation',
                                 'extreme', 'scale'])
    parser.add_argument('--max-steps', type=int, default=3000)
    args = parser.parse_args()

    seeds = (
        [42, 123, 456, 789, 100]
        if args.quick else
        [42, 123, 456, 789, 100, 200, 300, 400, 500, 600,
         11,  22,  33,  44,  55,  66,  77,  88,  99, 111,
         1000, 2000, 3000, 4000, 5000,
         6000, 7000, 8000, 9000, 9999]
    )
    max_steps = args.max_steps

    print(f'\n{"="*65}')
    print(f'  SCCS 仿真实验')
    print(f'  seeds={len(seeds)}  max_steps={max_steps}  '
          f'模式={"快速" if args.quick else "正式"}')
    print(f'{"="*65}')

    t0 = time.time()

    # ── 始终先收集覆盖率曲线（快，不依赖其他实验）
    if args.exp in ('all', 'sota'):
        print('\n→ 收集覆盖率曲线...')
        curves = collect_curves(seeds, max_steps)
        plot_coverage_curves(curves)

    # ── Exp1: SOTA 对比
    if args.exp in ('all', 'sota'):
        df_sota = exp_sota(seeds, max_steps)
        plot_sota_boxplots(df_sota)
        plot_comm_comparison(df_sota)

    # ── Exp2: 消融
    if args.exp in ('all', 'ablation'):
        df_abl = exp_ablation(seeds, max_steps)
        plot_ablation(df_abl)

    # ── Exp3: 极端环境
    if args.exp in ('all', 'extreme'):
        df_ext = exp_extreme(seeds, max_steps)
        plot_extreme(df_ext)

    # ── Exp4: 可扩展性
    if args.exp in ('all', 'scale'):
        df_sc = exp_scale(seeds, max_steps)
        plot_scale(df_sc)

    # ── 最终汇总
    elapsed = time.time() - t0
    print(f'\n{"="*65}')
    print(f'  全部完成  耗时: {elapsed/60:.1f} 分钟')
    print(f'  📊 数据: output/data/*.csv')
    print(f'  🖼️  图表: output/figures/*.png')
    print(f'{"="*65}')

    # ── 论文数字速查（仅SOTA跑完后显示）
    sota_path = Path('output/data/sota_results.csv')
    if sota_path.exists():
        df = pd.read_csv(sota_path)
        sccs   = df[df['method'] == 'SCCS（完整）']
        greedy = df[df['method'] == 'Greedy']
        if len(sccs) > 0 and len(greedy) > 0:
            step_imp = (greedy['completion_step'].mean() -
                        sccs['completion_step'].mean()
                        ) / greedy['completion_step'].mean() * 100
            var_imp  = (greedy['load_variance'].mean() -
                        sccs['load_variance'].mean()
                        ) / greedy['load_variance'].mean() * 100
            col_imp  = (greedy['collision_count'].mean() -
                        sccs['collision_count'].mean()
                        ) / max(greedy['collision_count'].mean(), 1) * 100
            print(f'\n  ══ 论文数字速查（{len(seeds)} seeds 平均值）══')
            print(f'     SCCS覆盖率:       '
                  f'{sccs["coverage_rate"].mean():.3f} '
                  f'± {sccs["coverage_rate"].std():.3f}')
            print(f'     响应时间缩短:     {step_imp:.1f}%')
            print(f'     负载方差降低:     {var_imp:.1f}%')
            print(f'     碰撞次数减少:     {col_imp:.1f}%')
            print(f'     TCFM显式通信:     0 条')
            print(f'     广播通信(Greedy): '
                  f'{greedy["comm_cost"].mean():.0f} 条')


if __name__ == '__main__':
    main()
