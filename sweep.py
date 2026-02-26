"""
sweep.py — Physics Counterfactual Sweep CLI
============================================
Sweeps one input feature across its data range while holding all others
fixed at baseline. Tests whether the network has learned physically correct
input-output relationships, or is merely interpolating training patterns.

Usage:
    python sweep.py --feature S0                        # sweep solar constant
    python sweep.py --feature S0 --track act_0 act_1   # also watch neuron activations
    python sweep.py --feature co2_ppm --baseline median
    python sweep.py --feature S0 --steps 500 --experiment ebm_0d_v1
    python sweep.py --feature S0 --no-plots             # print table only

Tip: Use feature names exactly as they appear in your data schema.
     Run `python analyze.py` to remind yourself of available features.
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
        description="Sweep one input feature across its range; observe network response."
    )
    parser.add_argument(
        '--feature', type=str, required=True,
        help="Input feature to sweep (e.g. 'S0', 'co2_ppm')"
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help="Experiment name (default: ACTIVE_EXPERIMENT in config.py)"
    )
    parser.add_argument(
        '--steps', type=int, default=200,
        help="Number of points along the sweep (default: 200)"
    )
    parser.add_argument(
        '--baseline', type=str, default='mean', choices=['mean', 'median'],
        help="How to set non-swept inputs: 'mean' or 'median' (default: mean)"
    )
    parser.add_argument(
        '--track', type=str, nargs='+', default=None, metavar='LAYER',
        help="Activation layers to track during sweep, e.g. --track act_0 act_1"
    )
    parser.add_argument(
        '--track-neurons', type=int, nargs='+', default=None, metavar='N',
        help="Specific neuron indices to plot from tracked layers (default: all, max 8)"
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Print summary table only, skip figure generation"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Directory for output figures (default: figures/ from config)"
    )
    return parser.parse_args()


def print_sweep_table(result, n_rows=10):
    """Print a condensed table of sweep values vs outputs."""
    feature_vals = result['feature_values']
    outputs = result['outputs']
    output_names = result['output_names']

    # Sample n_rows evenly spaced points
    indices = np.linspace(0, len(feature_vals) - 1, n_rows, dtype=int)

    header = f"  {'Input':>12s}" + "".join(f"  {n:>12s}" for n in output_names)
    print(f"\nSweep Table ({n_rows} sample points):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i in indices:
        row = f"  {feature_vals[i]:>12.4f}" + "".join(
            f"  {outputs[i, j]:>12.4f}" for j in range(outputs.shape[1])
        )
        print(row)
    print("-" * len(header))


def plot_sweep(result, output_dir, experiment_name, track_neurons=None):
    """
    Generate sweep figure:
      - Top panel(s): network output vs swept feature (one line per target)
      - Bottom panel(s): tracked neuron activations vs swept feature (if requested)
    """
    feature_name = result['feature_name']
    feature_vals = result['feature_values']
    outputs = result['outputs']
    output_names = result['output_names']
    layer_acts = result.get('layer_acts', {})

    n_output_panels = len(output_names)
    n_layer_panels = len(layer_acts)
    n_panels = n_output_panels + n_layer_panels

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(9, 3.5 * n_panels),
                             squeeze=False)
    axes = axes.flatten()

    # Output panels
    for i, name in enumerate(output_names):
        ax = axes[i]
        ax.plot(feature_vals, outputs[:, i], color='steelblue', linewidth=2)
        ax.set_xlabel(feature_name)
        ax.set_ylabel(name)
        ax.set_title(f'Network output: {name}')
        ax.grid(True, alpha=0.3)

        # Annotate slope direction
        slope = outputs[-1, i] - outputs[0, i]
        direction = '↑ increases' if slope > 0 else '↓ decreases'
        ax.text(0.02, 0.95, direction,
                transform=ax.transAxes, va='top', fontsize=9,
                color='steelblue')

    # Neuron activation panels (one per tracked layer)
    for panel_idx, (layer_name, acts) in enumerate(layer_acts.items()):
        ax = axes[n_output_panels + panel_idx]
        n_neurons = acts.shape[1]

        # Determine which neurons to plot
        if track_neurons is not None:
            neuron_indices = [n for n in track_neurons if n < n_neurons]
        else:
            neuron_indices = list(range(min(n_neurons, 8)))

        cmap = plt.get_cmap('tab10')
        for k, ni in enumerate(neuron_indices):
            ax.plot(feature_vals, acts[:, ni],
                    label=f'n{ni}', color=cmap(k % 10), linewidth=1.5)

        ax.set_xlabel(feature_name)
        ax.set_ylabel('Activation')
        ax.set_title(f'{layer_name} neuron activations during sweep')
        ax.legend(loc='upper right', fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{experiment_name} — sweep: {feature_name}',
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    fname = f"sweep_{feature_name}.png"
    savepath = os.path.join(output_dir, fname)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved sweep figure: {savepath}")


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
    print(f"sweep.py  |  {experiment['name']}  |  feature: {args.feature}")
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
    # 3. Validate inputs
    # -------------------------------------------------------------------------
    analyzer = InterpretabilityAnalyzer(model, train_ds)

    if args.feature not in train_ds.feature_names:
        print(f"\nERROR: Feature '{args.feature}' not found.")
        print(f"Available features: {train_ds.feature_names}")
        sys.exit(1)

    if args.track:
        bad = [l for l in args.track if l not in analyzer.activation_layers]
        if bad:
            print(f"\nERROR: Unknown layer(s): {bad}")
            print(f"Available layers: {analyzer.activation_layers}")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Run sweep
    # -------------------------------------------------------------------------
    print(f"\nRunning sweep over '{args.feature}' ({args.steps} steps, baseline={args.baseline})")
    if args.track:
        print(f"Tracking activations: {args.track}")

    result = analyzer.sweep_input(
        feature_name=args.feature,
        n_steps=args.steps,
        baseline=args.baseline,
        track_layers=args.track,
    )

    # -------------------------------------------------------------------------
    # 5. Print summary
    # -------------------------------------------------------------------------
    feature_vals = result['feature_values']
    outputs = result['outputs']

    print(f"\nFeature range: {feature_vals[0]:.4f} → {feature_vals[-1]:.4f}")
    for i, name in enumerate(result['output_names']):
        out_min, out_max = outputs[:, i].min(), outputs[:, i].max()
        slope = outputs[-1, i] - outputs[0, i]
        direction = 'positive' if slope > 0 else 'negative'
        print(f"  {name}: {out_min:.4f} → {out_max:.4f}  ({direction} relationship)")

    print_sweep_table(result)

    # -------------------------------------------------------------------------
    # 6. Plot
    # -------------------------------------------------------------------------
    if not args.no_plots:
        plot_sweep(result, output_dir, experiment['name'],
                   track_neurons=args.track_neurons)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
