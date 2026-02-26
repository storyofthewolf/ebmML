"""
analyze.py — Correlation Spectroscopy CLI
==========================================
Correlates neuron activations with physics variables and reports
which neurons specialize in which physical quantities.

Usage:
    python analyze.py                                   # use active experiment, all layers
    python analyze.py --experiment ebm_0d_v1            # specific experiment
    python analyze.py --layer act_0                     # single layer only
    python analyze.py --threshold 0.6 --samples 10000  # custom settings
    python analyze.py --no-plots                        # text output only
"""

import argparse
import os
import sys

from config import get_experiment, FIGURES_DIR
from climate_nn import load_model_from_checkpoint, load_and_split_data
from interpretability import (
    InterpretabilityAnalyzer,
    plot_correlations,
    plot_correlation_distributions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlation spectroscopy: correlate neurons with physics variables."
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help="Experiment name (default: ACTIVE_EXPERIMENT in config.py)"
    )
    parser.add_argument(
        '--layer', type=str, default=None,
        help="Activation layer to analyze, e.g. 'act_0' (default: all layers)"
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help="Minimum |r| to count as specialized (default: 0.5)"
    )
    parser.add_argument(
        '--samples', type=int, default=20000,
        help="Number of samples for correlation computation (default: 20000)"
    )
    parser.add_argument(
        '--sort-by', type=str, default=None,
        help="Physics variable to sort heatmap rows by (default: first variable)"
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Skip plot generation, print summary only"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="Directory for output files (default: figures/ from config)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load experiment configuration
    # -------------------------------------------------------------------------
    experiment = get_experiment(args.experiment)
    paths = experiment['paths']
    data_schema = experiment['data_schema']
    training_config = experiment['training_config']

    output_dir = args.output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"analyze.py  |  {experiment['name']}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 2. Load model and data
    # -------------------------------------------------------------------------
    if not os.path.exists(paths['model']):
        print(f"\nERROR: No trained model found at {paths['model']}")
        print("Run `python climate_nn.py` first to train a model.")
        sys.exit(1)

    print(f"\nLoading model: {paths['model']}")
    model, checkpoint = load_model_from_checkpoint(paths['model'])

    print(f"Loading data:  {paths['data']}")
    train_ds, _ = load_and_split_data(
        paths['data'],
        data_schema,
        val_size=training_config['val_size'],
        seed=training_config['seed']
    )

    # -------------------------------------------------------------------------
    # 3. Run correlation spectroscopy
    # -------------------------------------------------------------------------
    analyzer = InterpretabilityAnalyzer(model, train_ds)

    if args.layer:
        if args.layer not in analyzer.activation_layers:
            print(f"\nERROR: Layer '{args.layer}' not found.")
            print(f"Available layers: {analyzer.activation_layers}")
            sys.exit(1)
        df_corr = analyzer.correlate_neurons_with_physics(
            layer_name=args.layer,
            sample_size=args.samples
        )
    else:
        df_corr = analyzer.correlate_all_layers(sample_size=args.samples)

    # -------------------------------------------------------------------------
    # 4. Print specialization summary
    # -------------------------------------------------------------------------
    analyzer.summarize_specializations(df_corr, threshold=args.threshold)

    # -------------------------------------------------------------------------
    # 5. Save CSV
    # -------------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "neuron_correlations.csv")
    df_corr.to_csv(csv_path, index=False)
    print(f"\nSaved correlation data: {csv_path}")

    # -------------------------------------------------------------------------
    # 6. Generate plots (unless suppressed)
    # -------------------------------------------------------------------------
    if not args.no_plots:
        print("\nGenerating plots...")

        heatmap_path = os.path.join(output_dir, "neuron_correlations.png")
        plot_correlations(
            df_corr,
            heatmap_path,
            sort_by=args.sort_by,
            title=f"{experiment['name']} — Neuron Specialization"
        )

        dist_path = os.path.join(output_dir, "correlation_distributions.png")
        plot_correlation_distributions(df_corr, dist_path)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
