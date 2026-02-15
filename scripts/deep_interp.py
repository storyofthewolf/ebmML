import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from climate_nn import load_model_from_checkpoint

def deep_interpretability_analysis():
    checkpoint_path = '../networks/climate_model.pt'
    if not os.path.exists(checkpoint_path):
        print("Model file not found.")
        return

    csv_path = '../training_sets/ebm_0d_model_v1_climate_data_1M.csv'
    if not os.path.exists(csv_path):
        print("Model file not found.")
        return

    # 1. Load Model and Data
    try:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    
    df = pd.read_csv(csv_path).sample(10000) # Sample for speed
    scaler_X = checkpoint['scaler_X']
    X_scaled = torch.FloatTensor(scaler_X.transform(df[['Ts', 'log_pCO2', 'S0']].values))

    # 2. Get Activations for Layer 0 (Senses) and Layer 3 (Logic)
    with torch.no_grad():
        acts = model.get_hidden_activations(X_scaled)
    
    # --- PART 1: SCATTER PLOTS (The Shape of a Law) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Neuron 3 (Your Ice Detector) vs Albedo
    ax1.scatter(df['Albedo'], acts['relu_0'][:, 3], alpha=0.4, s=2, color='blue')
    ax1.set_title("Neuron 3 Activation vs. Physical Albedo")
    ax1.set_xlabel("True Albedo (EBM Law)")
    ax1.set_ylabel("Neuron 3 'Firing' Strength")
    
    # Plot Neuron 0 (Your Greenhouse Sensor) vs Emissivity
    ax2.scatter(df['Emissivity'], acts['relu_0'][:, 0], alpha=0.4, s=2, color='red')
    ax2.set_title("Neuron 0 Activation vs. Physical Emissivity")
    ax2.set_xlabel("True Emissivity (EBM Law)")
    ax2.set_ylabel("Neuron 0 'Firing' Strength")
    
    plt.tight_layout()
    plt.savefig('figures/neuron_scatter_shapes.png')
    print("Saved scatter plots to figures/neuron_scatter_shapes.png")

    # --- PART 2: THE DEEP LAYER CORRELATION ---
    # We repeat your heatmap logic but for 'relu_3'
    deep_acts = acts['relu_3'].numpy()
    results = []
    for i in range(8):
        results.append({
            'neuron': i,
            'corr_Ts': np.corrcoef(deep_acts[:, i], df['Ts'])[0,1],
            'corr_Albedo': np.corrcoef(deep_acts[:, i], df['Albedo'])[0,1],
            'corr_Emissivity': np.corrcoef(deep_acts[:, i], df['Emissivity'])[0,1],
            'corr_N_toa': np.corrcoef(deep_acts[:, i], df['N_toa'])[0,1]
        })
    
    df_deep = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    plt.imshow(df_deep[['corr_Ts', 'corr_Albedo', 'corr_Emissivity', 'corr_N_toa']].values, 
               cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(4), ['Ts', 'Albedo', 'Emissivity', 'N_toa'])
    plt.title('DEEP LAYER (relu_3) Specialization\n(Note the increased complexity)')
    plt.savefig('figures/deep_layer_correlations.png')
    print("Saved deep correlation map to figures/deep_layer_correlations.png")

if __name__ == "__main__":
    deep_interpretability_analysis()
