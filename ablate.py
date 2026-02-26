"""
ablate.py — Causal Ablation Study CLI
======================================
Zeros out neurons during forward passes to test causal importance.
Distinguishes neurons that are merely correlated with physics variables
from those that are causally necessary for correct outputs.

Two modes:
  single  — ablate a specific set of neurons together, report output change
  rank    — ablate each neuron individually, rank all by causal impact

Usage:
    python ablate.py --layer act_0 --mode rank
    python ablate.py --layer act_0 --neurons 2 5 --mode single
    python ablate.py --layer act_0 --mode rank --samples 10000
    python ablate.py --layer act_0 --mode rank --no-plots
    python ablate.py --layer act_0 --neurons 2 5 --mode single --experiment ebm_0d_v1

Tip: Run analyze.py first to identify candidate neurons, then use
     'rank' mode to confirm which ones are causally important.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import get_experiment, FIGURES_DIR
from climate_nn import load_model_from_checkpoint, load_and_split_data
from interpretability import InterpretabilityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation study: zero out neurons and measure causal impact on outputs."
    )
    parser.add_argument(
        '--layer', type=str, required=True,
        help="Activation layer to ablate, e.g. 'act_0'"
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['single', 'rank'],
        help=(
            "'single': ablate --neurons together and report the combined effect. "
            "'rank': ablate each neuron individually and rank by impact."
        )
    )
    parser.add_argument(
        '--neurons', type=int, nargs='+', default=None, metavar='N',
        help="Neuron indices to ablate (required for 'single' mode)"
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help="Experiment name (default: ACTIVE_EXPERIMENT in config.py)"
    )
    parser.add_argument(
        '--samples', type=int, default=5000,
        help="Number of samples to evaluate over (default: 5000)"
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Print text summary only, skip figure generation"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Directory for output files (default: figures/ from config)"
    )
    return parser.parse_args()


def print_single_result(result):
    """Print a detailed report for a single ablation experiment."""
    neurons = result['neuron_indices']
    layer   = result['layer_name']

    print(f"\nAblation Result — {layer}: neuron(s) {neurons}")
    print("-" * 55)
    print(f"  {'Target':<20s}  {'Baseline':>10s}  {'Ablated':>10s}  {'Δ':>10s}  {'Δ%':>8s}")
    print("-" * 55)

    for i, name in enumerate(result['output_names']):
        print(
            f"  {name:<20s}"
            f"  {result['baseline_output'][i]:>10.4f}"
            f"  {result['ablated_output'][i]:>10.4f}"
            f"  {result['delta'][i]:>+10.4f}"
            f"  {result['delta_pct'][i]:>+7.1f}%"
        )
    print("-" * 55)

    mean_impact = float(np.abs(result['delta_pct']).mean())
    print(f"  Mean |Δ%| across targets: {mean_impact:.2f}%")

    if mean_impact < 1.0:
        verdict = "Minimal causal role — output barely changed."
    elif mean_impact < 10.0:
        verdict = "Moderate causal role — noticeable output shift."
    else:
        verdict = "Strong causal role — substantial output disruption."
    print(f"  Verdict: {verdict}")


def print_rank_result(df_rank):
    """Print the neuron importance ranking table."""
    print(f"\nNeuron Importance Ranking (by mean |Δ%| across targets):")
    print("-" * 60)

    delta_pct_cols = [c for c in df_rank.columns if c.startswith('delta_pct_')]
    target_names   = [c.replace('delta_pct_', '') for c in delta_pct_cols]

    header = f"  {'Rank':>4s}  {'Neuron':>6s}  {'Mean|Δ%|':>10s}"
    for t in target_names:
        header += f"  {('Δ%_' + t):>12s}"
    print(header)
    print("-" * 60)

    for rank, (_, row) in enumerate(df_rank.iterrows(), 1):
        line = f"  {rank:>4d}  {int(row['neuron']):>6d}  {row['mean_abs_delta_pct']:>9.2f}%"
        for t in target_names:
            line += f"  {row[f'delta_pct_{t}']:>+11.1f}%"
        print(line)

    print("-" * 60)


def plot_single(result, output_dir, experiment_name):
    """Bar chart comparing baseline vs ablated output per target."""
    output_names    = result['output_names']
    baseline        = result['baseline_output']
    ablated         = result['ablated_output']
    neurons         = result['neuron_indices']
    layer           = result['layer_name']

    x = np.arange(len(output_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(output_names)), 5))

    bars_b = ax.bar(x - width/2, baseline, width, label='Baseline',  color='steelblue',  alpha=0.85)
    bars_a = ax.bar(x + width/2, ablated,  width, label='Ablated',   color='tomato',     alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(output_names)
    ax.set_ylabel('Mean Output Value (original units)')
    ax.set_title(
        f'{experiment_name}\n'
        f'Ablation: {layer} neuron(s) {neurons}'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotate delta % above bars
    for xi, (b, a) in enumerate(zip(baseline, ablated)):
        pct = 100 * (a - b) / (abs(b) + 1e-8)
        ax.text(xi + width/2, a, f'{pct:+.1f}%',
                ha='center', va='bottom', fontsize=9, color='tomato')

    plt.tight_layout()
    neuron_str = '_'.join(str(n) for n in neurons)
    fname = f"ablate_{layer}_n{neuron_str}.png"
    savepath = os.path.join(output_dir, fname)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved ablation figure: {savepath}")


def plot_rank(df_rank, output_dir, experiment_name, layer_name):
    """Horizontal bar chart of neuron importance ranking."""
    neurons    = df_rank['neuron'].astype(int).tolist()
    importance = df_rank['mean_abs_delta_pct'].tolist()

    # Color by impact magnitude
    colors = ['tomato' if v > 10 else 'steelblue' if v > 1 else 'lightgray'
              for v in importance]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(neurons))))

    y_pos = range(len(neurons))
    ax.barh(y_pos, importance, color=colors, edgecolor='white', alpha=0.85)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f'n{n}' for n in neurons])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |Δ%| across targets')
    ax.set_title(f'{experiment_name}\nNeuron Importance Ranking — {layer_name}')
    ax.axvline(1,  color='steelblue', linestyle='--', alpha=0.5, label='1% threshold')
    ax.axvline(10, color='tomato',    linestyle='--', alpha=0.5, label='10% threshold')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fname = f"ablate_rank_{layer_name}.png"
    savepath = os.path.join(output_dir, fname)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved ranking figure: {savepath}")


def main():
    args = parse_args()

    # Validate mode/neuron combination
    if args.mode == 'single' and not args.neurons:
        print("ERROR: --mode single requires --neurons (e.g. --neurons 2 5)")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Load experiment
    # -------------------------------------------------------------------------
    experiment = get_experiment(args.experiment)
    paths      = experiment['paths']
    data_schema     = experiment['data_schema']
    training_config = experiment['training_config']

    output_dir = args.output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ablate.py  |  {experiment['name']}  |  {args.layer}  |  mode={args.mode}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 2. Load model and data
    # -------------------------------------------------------------------------
    if not os.path.exists(paths['model']):
        print(f"\nERROR: No trained model found at {paths['model']}")
        print("Run `python climate_nn.py` first.")
        sys.exit(1)

    model, _ = load_model_from_checkpoint(paths['model'])
    train_ds, _ = load_and_split_data(
        paths['data'],
        data_schema,
        val_size=training_config['val_size'],
        seed=training_config['seed']
    )

    # -------------------------------------------------------------------------
    # 3. Validate layer
    # -------------------------------------------------------------------------
    analyzer = InterpretabilityAnalyzer(model, train_ds)

    if args.layer not in analyzer.activation_layers:
        print(f"\nERROR: Layer '{args.layer}' not found.")
        print(f"Available layers: {analyzer.activation_layers}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Run ablation
    # -------------------------------------------------------------------------
    if args.mode == 'single':
        print(f"\nAblating neuron(s) {args.neurons} in '{args.layer}'...")
        result = analyzer.ablate_neurons(
            args.layer, args.neurons, sample_size=args.samples
        )
        print_single_result(result)

        if not args.no_plots:
            plot_single(result, output_dir, experiment['name'])

    else:  # rank
        print(f"\nRanking all neurons in '{args.layer}' by ablation impact...")
        df_rank = analyzer.rank_neurons_by_ablation(
            args.layer, sample_size=args.samples
        )
        print_rank_result(df_rank)

        # Save CSV
        csv_path = os.path.join(output_dir, f"ablate_rank_{args.layer}.csv")
        df_rank.to_csv(csv_path, index=False)
        print(f"\nSaved ranking data: {csv_path}")

        if not args.no_plots:
            plot_rank(df_rank, output_dir, experiment['name'], args.layer)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
