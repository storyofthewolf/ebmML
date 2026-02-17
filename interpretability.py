"""
Interpretability Analysis Module
================================
Schema-driven mechanistic interpretability tools for Climate NN.

This module provides:
1. InterpretabilityAnalyzer: Correlate neuron activations with physics variables
2. Plotting utilities: Heatmaps, neuron specialization visualizations
3. Ablation study utilities (extensible)

Usage:
    from interpretability import InterpretabilityAnalyzer, plot_correlations
    
    # Load trained model and data
    model, checkpoint = load_model_from_checkpoint(MODEL_PATH)
    train_ds, val_ds = load_and_split_data(DATA_PATH, data_schema)
    
    # Analyze
    analyzer = InterpretabilityAnalyzer(model, train_ds)
    df_corr = analyzer.correlate_neurons_with_physics()
    
    # Plot
    plot_correlations(df_corr, 'neuron_correlations.png')
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from config import get_experiment, get_paths, FIGURES_DIR
from climate_nn import load_model_from_checkpoint, load_and_split_data


# =============================================================================
# 1. INTERPRETABILITY ANALYZER
# =============================================================================

class InterpretabilityAnalyzer:
    """
    Analyze neuron activations and their correlations with physics variables.
    
    Schema-driven: automatically uses whatever physics_meta variables are
    available in the dataset, enabling analysis across different EBM complexities.
    
    Args:
        model: Trained ClimateMLP instance
        dataset: ClimateDataset with physics_meta
        device: Computation device ('cpu' or 'cuda')
    """
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()
        
        # Get available physics variables from dataset
        self.physics_variables = dataset.get_physics_meta_names()
        
        # Get available activation layers from model
        self.activation_layers = model.get_activation_layer_names()
        
        print(f"InterpretabilityAnalyzer initialized:")
        print(f"  Physics variables: {self.physics_variables}")
        print(f"  Activation layers: {self.activation_layers}")
        
    def get_activations(self, layer_name, sample_size=None, indices=None):
        """
        Get activations from a specific layer for a subset of data.
        
        Args:
            layer_name: Name of activation layer (e.g., 'act_0')
            sample_size: Number of random samples (ignored if indices provided)
            indices: Specific indices to use
            
        Returns:
            tuple: (activations_array, indices_used)
        """
        if layer_name not in self.activation_layers:
            raise ValueError(
                f"Unknown layer '{layer_name}'. "
                f"Available: {self.activation_layers}"
            )
        
        # Determine which samples to use
        if indices is None:
            if sample_size is None:
                sample_size = min(20000, len(self.dataset))
            indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        X_subset = self.dataset.X[indices].to(self.device)
        
        with torch.no_grad():
            activations = self.model.get_hidden_activations(X_subset)
            acts = activations[layer_name]
        
        return acts.cpu().numpy(), indices
    
    def correlate_neurons_with_physics(self, layer_name=None, sample_size=20000):
        """
        Compute correlations between all neurons and all physics variables.
        
        Schema-driven: automatically correlates with whatever physics_meta
        variables are available in the dataset.
        
        Args:
            layer_name: Which activation layer to analyze (default: first layer)
            sample_size: Number of samples for correlation computation
            
        Returns:
            pd.DataFrame: Columns are 'neuron', 'layer', and 'corr_{variable}'
                          for each physics variable
        """
        if layer_name is None:
            layer_name = self.activation_layers[0]
            
        print(f"\nRunning Neuron Spectroscopy on '{layer_name}'...")
        
        acts, indices = self.get_activations(layer_name, sample_size)
        n_neurons = acts.shape[1]
        
        # Get physics variables for sampled indices
        physics_values = {
            var: self.dataset.physics_meta[var][indices]
            for var in self.physics_variables
        }
        
        results = []
        for i in range(n_neurons):
            neuron_acts = acts[:, i]
            
            # Skip dead neurons
            if np.std(neuron_acts) < 1e-6:
                continue
            
            row = {
                'neuron': i,
                'layer': layer_name,
            }
            
            # Compute correlation with each physics variable
            for var_name, var_values in physics_values.items():
                corr = np.corrcoef(neuron_acts, var_values)[0, 1]
                row[f'corr_{var_name}'] = corr
            
            results.append(row)
        
        df = pd.DataFrame(results)
        print(f"  Analyzed {len(df)} active neurons (of {n_neurons} total)")
        
        return df
    
    def correlate_all_layers(self, sample_size=20000):
        """
        Run correlation analysis on all activation layers.
        
        Returns:
            pd.DataFrame: Combined results from all layers
        """
        all_results = []
        
        for layer_name in self.activation_layers:
            df = self.correlate_neurons_with_physics(layer_name, sample_size)
            all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True)
    
    def find_specialized_neurons(self, df_corr, variable, threshold=0.5):
        """
        Find neurons that are highly correlated with a specific variable.
        
        Args:
            df_corr: DataFrame from correlate_neurons_with_physics()
            variable: Physics variable name (without 'corr_' prefix)
            threshold: Minimum absolute correlation
            
        Returns:
            pd.DataFrame: Filtered and sorted by correlation strength
        """
        col = f'corr_{variable}'
        if col not in df_corr.columns:
            available = [c.replace('corr_', '') for c in df_corr.columns if c.startswith('corr_')]
            raise ValueError(f"Variable '{variable}' not found. Available: {available}")
        
        mask = df_corr[col].abs() >= threshold
        return df_corr[mask].sort_values(col, key=abs, ascending=False)
    
    def summarize_specializations(self, df_corr, threshold=0.5):
        """
        Print a summary of which neurons specialize in which variables.
        
        Args:
            df_corr: DataFrame from correlate_neurons_with_physics()
            threshold: Minimum absolute correlation to count as specialized
        """
        corr_cols = [c for c in df_corr.columns if c.startswith('corr_')]
        
        print(f"\nNeuron Specialization Summary (|r| >= {threshold}):")
        print("-" * 50)
        
        for col in corr_cols:
            var_name = col.replace('corr_', '')
            specialized = (df_corr[col].abs() >= threshold).sum()
            
            if specialized > 0:
                best_idx = df_corr[col].abs().idxmax()
                best_neuron = df_corr.loc[best_idx, 'neuron']
                best_corr = df_corr.loc[best_idx, col]
                print(f"  {var_name:15s}: {specialized:2d} neurons, "
                      f"strongest = neuron {best_neuron} (r={best_corr:+.3f})")
            else:
                print(f"  {var_name:15s}: No specialized neurons")


# =============================================================================
# 2. PLOTTING UTILITIES
# =============================================================================

def plot_correlations(df_corr, savepath, sort_by=None, title=None, figsize=(10, 12)):
    """
    Create a heatmap of neuron correlations with physics variables.
    
    Schema-driven: automatically handles whatever correlation columns exist.
    
    Args:
        df_corr: DataFrame from InterpretabilityAnalyzer.correlate_neurons_with_physics()
        savepath: Path to save the figure
        sort_by: Variable name to sort neurons by (default: first variable)
        title: Plot title (default: auto-generated)
        figsize: Figure size tuple
    """
    # Extract correlation columns
    corr_cols = [c for c in df_corr.columns if c.startswith('corr_')]
    if not corr_cols:
        raise ValueError("No correlation columns found in DataFrame")
    
    # Variable names for labels
    var_names = [c.replace('corr_', '') for c in corr_cols]
    
    # Determine sort column
    if sort_by is None:
        sort_col = corr_cols[0]
        sort_var = var_names[0]
    else:
        sort_col = f'corr_{sort_by}'
        sort_var = sort_by
        if sort_col not in corr_cols:
            raise ValueError(f"Sort variable '{sort_by}' not found. Available: {var_names}")
    
    # Sort and extract data
    df_sorted = df_corr.sort_values(sort_col, ascending=False)
    data = df_sorted[corr_cols].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Correlation')
    
    # Labels
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    
    # Y-axis: neuron labels
    if 'layer' in df_sorted.columns:
        y_labels = [f"{row['layer']}:n{row['neuron']}" 
                    for _, row in df_sorted.iterrows()]
    else:
        y_labels = df_sorted['neuron'].values
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Physics Variable')
    ax.set_ylabel('Neuron')
    
    if title is None:
        title = f'Neuron Specialization (Sorted by {sort_var})'
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation heatmap to {savepath}")


def plot_correlation_distributions(df_corr, savepath, figsize=(12, 4)):
    """
    Plot distribution of correlations for each physics variable.
    
    Args:
        df_corr: DataFrame from InterpretabilityAnalyzer.correlate_neurons_with_physics()
        savepath: Path to save the figure
        figsize: Figure size tuple
    """
    corr_cols = [c for c in df_corr.columns if c.startswith('corr_')]
    var_names = [c.replace('corr_', '') for c in corr_cols]
    
    n_vars = len(corr_cols)
    fig, axes = plt.subplots(1, n_vars, figsize=figsize, sharey=True)
    
    if n_vars == 1:
        axes = [axes]
    
    for ax, col, var in zip(axes, corr_cols, var_names):
        values = df_corr[col].dropna()
        
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'Correlation with {var}')
        ax.set_xlim(-1, 1)
        
        # Annotate with stats
        mean_corr = values.mean()
        max_corr = values.abs().max()
        ax.text(0.05, 0.95, f'mean: {mean_corr:.3f}\nmax |r|: {max_corr:.3f}',
                transform=ax.transAxes, verticalalignment='top', fontsize=9)
    
    axes[0].set_ylabel('Count')
    fig.suptitle('Distribution of Neuron-Physics Correlations', y=1.02)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation distributions to {savepath}")


def plot_neuron_response(analyzer, neuron_idx, layer_name, physics_var,
                         savepath, sample_size=5000, figsize=(6, 5)):
    """
    Scatter plot of a single neuron's activation vs a physics variable.
    
    Args:
        analyzer: InterpretabilityAnalyzer instance
        neuron_idx: Index of neuron to plot
        layer_name: Activation layer name
        physics_var: Physics variable name
        savepath: Path to save the figure
        sample_size: Number of points to plot
        figsize: Figure size tuple
    """
    acts, indices = analyzer.get_activations(layer_name, sample_size)
    
    neuron_acts = acts[:, neuron_idx]
    physics_values = analyzer.dataset.physics_meta[physics_var][indices]
    
    corr = np.corrcoef(neuron_acts, physics_values)[0, 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(physics_values, neuron_acts, alpha=0.3, s=5)
    ax.set_xlabel(physics_var)
    ax.set_ylabel(f'Neuron {neuron_idx} Activation')
    ax.set_title(f'{layer_name}:Neuron {neuron_idx} vs {physics_var}\nr = {corr:.3f}')
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    
    print(f"Saved neuron response plot to {savepath}")


# =============================================================================
# 3. CONVENIENCE FUNCTIONS
# =============================================================================

def quick_analysis(experiment_name=None, layer_name=None):
    """
    One-liner to run full interpretability analysis on a trained model.
    
    Args:
        experiment_name: Experiment to analyze (default: ACTIVE_EXPERIMENT)
        layer_name: Specific layer to analyze (default: all layers)
        
    Returns:
        tuple: (analyzer, df_correlations)
    """
    from climate_nn import load_model_from_checkpoint, load_and_split_data
    
    experiment = get_experiment(experiment_name)
    paths = experiment['paths']
    data_schema = experiment['data_schema']
    
    print(f"\n{'='*60}")
    print(f"Quick Analysis: {experiment['name']}")
    print(f"{'='*60}")
    
    # Load model
    model, checkpoint = load_model_from_checkpoint(paths['model'])
    
    # Load data
    train_ds, val_ds = load_and_split_data(
        paths['data'], 
        data_schema,
        val_size=experiment['training_config']['val_size'],
        seed=experiment['training_config']['seed']
    )
    
    # Analyze
    analyzer = InterpretabilityAnalyzer(model, train_ds)
    
    if layer_name:
        df_corr = analyzer.correlate_neurons_with_physics(layer_name)
    else:
        df_corr = analyzer.correlate_all_layers()
    
    # Summary
    analyzer.summarize_specializations(df_corr)
    
    return analyzer, df_corr


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run analysis on the active experiment
    experiment = get_experiment()
    paths = experiment['paths']
    data_schema = experiment['data_schema']
    
    print(f"\n{'='*60}")
    print(f"Interpretability Analysis: {experiment['name']}")
    print(f"{'='*60}")
    
    # 1. Load trained model
    print(f"\nLoading model from {paths['model']}...")
    model, checkpoint = load_model_from_checkpoint(paths['model'])
    
    # 2. Load data (need physics_meta for correlations)
    print(f"\nLoading data from {paths['data']}...")
    train_ds, _ = load_and_split_data(
        paths['data'],
        data_schema,
        val_size=experiment['training_config']['val_size'],
        seed=experiment['training_config']['seed']
    )
    
    # 3. Run analysis
    analyzer = InterpretabilityAnalyzer(model, train_ds)
    
    # Analyze all layers
    df_corr = analyzer.correlate_all_layers()
    
    # Print summary
    analyzer.summarize_specializations(df_corr)
    
    # 4. Generate plots
    print("\nGenerating plots...")
    
    # Main correlation heatmap
    heatmap_path = os.path.join(FIGURES_DIR, "neuron_correlations.png")
    
    # Determine first physics variable for sorting
    physics_vars = train_ds.get_physics_meta_names()
    sort_var = physics_vars[0] if physics_vars else None
    
    plot_correlations(df_corr, heatmap_path, sort_by=sort_var)
    
    # Distribution plot
    dist_path = os.path.join(FIGURES_DIR, "correlation_distributions.png")
    plot_correlation_distributions(df_corr, dist_path)
    
    # 5. Save results
    csv_path = os.path.join(FIGURES_DIR, "neuron_correlations.csv")
    df_corr.to_csv(csv_path, index=False)
    print(f"Saved correlation data to {csv_path}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {FIGURES_DIR}/")
