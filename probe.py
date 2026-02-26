"""
probe.py — Single Neuron Deep Dive CLI
=======================================
Examines one neuron in detail: activation distribution, scatter plots
against all physics variables, and incoming weight profile.

Usage:
    python probe.py --layer act_0 --neuron 3
    python probe.py --layer act_0 --neuron 3 --experiment ebm_0d_v1
    python probe.py --layer act_0 --neuron 3 --samples 5000
    python probe.py --layer act_0 --neuron 3 --no-plots

Tip: Run analyze.py first to identify neurons of interest.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from config import get_experiment, FIGURES_DIR
from climate_nn import load_model_from_checkpoint, load_and_split_data
from interpretability import InterpretabilityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-neuron deep dive: distribution, physics scatter, weight profile."
    )
    parser.add_argument(
        '--layer', type=str, required=True,
        help="Activation layer to probe, e.g. 'act_0'"
    )
    parser.add_argument(
        '--neuron', type=int, required=True,
        help="Neuron index within that layer"
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help="Experiment name (default: ACTIVE_EXPERIMENT in config.py)"
    )
    parser.add_argument(
        '--samples', type=int, default=5000,
        help="Number of samples to use (default: 5000)"
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Print text summary only, skip figure generation"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Directory for output figure (default: figures/ from config)"
    )
    return parser.parse_args()


def print_weight_profile(df_weights, neuron_idx, layer_name, top_n=10):
    """Print the top incoming weights for a neuron."""
    print(f"\nIncoming Weight Profile — {layer_name}:neuron {neuron_idx}")
    print("-" * 45)
    n = min(top_n, len(df_weights))
    for _, row in df_weights.head(n).iterrows():
        bar_len = int(abs(row['weight']) / df_weights['weight'].abs().max() * 20)
        direction = '+' if row['weight'] > 0 else '-'
        bar = direction * bar_len
        print(f"  {row['input_name']:15s}  {row['weight']:+.4f}  |{bar}")


def plot_probe(analyzer, layer_name, neuron_idx, samples, output_dir, experiment_name):
    """
    Generate a multi-panel probe figure for a single neuron:
      - Top-left:  activation distribution (histogram)
      - Remaining: scatter vs each physics variable
    """
    physics_vars = analyzer.physics_variables
    n_physics = len(physics_vars)

    # Total panels: 1 histogram + n_physics scatters
    n_panels = 1 + n_physics
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Get activations once
    acts, indices = analyzer.get_activations(layer_name, sample_size=samples)
    neuron_acts = acts[:, neuron_idx]

    # Panel 0: activation distribution
    ax = axes_flat[0]
    ax.hist(neuron_acts, bins=40, edgecolor='black', alpha=0.75, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.6, label='zero')
    ax.set_xlabel('Activation')
    ax.set_ylabel('Count')
    ax.set_title(f'{layer_name}:n{neuron_idx} — Distribution')
    ax.legend(fontsize=8)

    dead_frac = (neuron_acts == 0).mean()
    ax.text(0.97, 0.95, f'dead: {dead_frac:.1%}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)

    # Panels 1+: scatter vs each physics variable
    for i, var in enumerate(physics_vars):
        ax = axes_flat[i + 1]
        phys_vals = analyzer.dataset.physics_meta[var][indices]
        corr = np.corrcoef(neuron_acts, phys_vals)[0, 1]

        ax.scatter(phys_vals, neuron_acts, alpha=0.2, s=4, color='steelblue')
        ax.set_xlabel(var)
        ax.set_ylabel(f'n{neuron_idx} activation')
        ax.set_title(f'vs {var}  (r={corr:+.3f})')

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f'{experiment_name} — {layer_name}:neuron {neuron_idx}',
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    fname = f"probe_{layer_name}_n{neuron_idx}.png"
    savepath = os.path.join(output_dir, fname)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved probe figure: {savepath}")


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load experiment
    # -------------------------------------------------------------------------
    experiment = get_experiment(args.experiment)
    paths = experiment['paths']
    data_schema = experiment['data_schema']
    training_config = experiment['training_config']

    output_dir = args.output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"probe.py  |  {experiment['name']}  |  {args.layer}:neuron {args.neuron}")
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
    # 3. Validate layer and neuron
    # -------------------------------------------------------------------------
    analyzer = InterpretabilityAnalyzer(model, train_ds)

    if args.layer not in analyzer.activation_layers:
        print(f"\nERROR: Layer '{args.layer}' not found.")
        print(f"Available layers: {analyzer.activation_layers}")
        sys.exit(1)

    # Check neuron index is in range by peeking at one activation
    acts_check, _ = analyzer.get_activations(args.layer, sample_size=1)
    n_neurons = acts_check.shape[1]
    if args.neuron >= n_neurons or args.neuron < 0:
        print(f"\nERROR: Neuron {args.neuron} out of range. Layer has {n_neurons} neurons (0–{n_neurons-1}).")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Weight profile (text always shown)
    # -------------------------------------------------------------------------
    df_weights = analyzer.get_weight_profile(args.layer, args.neuron)
    print_weight_profile(df_weights, args.neuron, args.layer)

    # -------------------------------------------------------------------------
    # 5. Correlation summary for this neuron (text always shown)
    # -------------------------------------------------------------------------
    acts, indices = analyzer.get_activations(args.layer, sample_size=args.samples)
    neuron_acts = acts[:, args.neuron]

    print(f"\nCorrelations with physics variables ({args.samples} samples):")
    print("-" * 45)
    for var in analyzer.physics_variables:
        phys_vals = train_ds.physics_meta[var][indices]
        corr = np.corrcoef(neuron_acts, phys_vals)[0, 1]
        bar_len = int(abs(corr) * 20)
        direction = '+' if corr > 0 else '-'
        bar = direction * bar_len
        print(f"  {var:20s}  r={corr:+.3f}  |{bar}")

    # -------------------------------------------------------------------------
    # 6. Plot
    # -------------------------------------------------------------------------
    if not args.no_plots:
        plot_probe(analyzer, args.layer, args.neuron,
                   args.samples, output_dir, experiment['name'])

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
