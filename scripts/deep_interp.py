import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- STANDARD HEADER START ---
# Add parent directory to path to find config.py and climate_nn.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- STANDARD HEADER END ---

# [NEW] Import paths and loaders from the central config/utility
from config import MODEL_PATH, DATA_PATH, FIGURES_DIR
from climate_nn import load_model_from_checkpoint

def deep_interpretability_analysis():
    # 1. Validation of Paths
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at: {DATA_PATH}")
        return

    # 2. Load Model
    try:
        model, checkpoint = load_model_from_checkpoint(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load Data
    # Use pandas directly as in original script, but use DATA_PATH
    try:
        # Sample for speed, just like original
        df = pd.read_csv(DATA_PATH).sample(10000, random_state=42) 
    except Exception as e:
        print(f"Error loading data CSV: {e}")
        return

    # Prepare inputs using the scaler saved with the model
    scaler_X = checkpoint['scaler_X']
    X_input = df[['Ts', 'log_pCO2', 'S0']].values
    X_scaled = torch.FloatTensor(scaler_X.transform(X_input))

    # 4. Get Activations
    # We need 'relu_0' (Layer 1 sensors) and 'relu_3' (Deep Logic)
    with torch.no_grad():
        acts = model.get_hidden_activations(X_scaled)
    
    # --- PART 1: SCATTER PLOTS (The Shape of a Law) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Retrieve activations (converting to numpy)
    layer0_acts = acts['relu_0'].numpy()
    
    # Plot Neuron 3 (Ice Detector) vs Albedo
    ax1.scatter(df['Albedo'], layer0_acts[:, 3], alpha=0.4, s=2, color='blue')
    ax1.set_title("Neuron 3 Activation vs. Physical Albedo")
    ax1.set_xlabel("True Albedo (EBM Law)")
    ax1.set_ylabel("Neuron 3 'Firing' Strength")
    
    # Plot Neuron 0 (Greenhouse Sensor) vs Emissivity
    ax2.scatter(df['Emissivity'], layer0_acts[:, 0], alpha=0.4, s=2, color='red')
    ax2.set_title("Neuron 0 Activation vs. Physical Emissivity")
    ax2.set_xlabel("True Emissivity (EBM Law)")
    ax2.set_ylabel("Neuron 0 'Firing' Strength")
    
    plt.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, 'neuron_scatter_shapes.png')
    plt.savefig(scatter_path)
    print(f"Saved scatter plots to {scatter_path}")

    # --- PART 2: THE DEEP LAYER CORRELATION ---
    # We repeat your heatmap logic but for 'relu_3' (Layer 4)
    if 'relu_3' not in acts:
        print("Warning: 'relu_3' not found in activations. Your model might be shallower than expected.")
        return

    deep_acts = acts['relu_3'].numpy()
    n_neurons = deep_acts.shape[1]
    
    results = []
    for i in range(n_neurons):
        # Calculate correlations with ground truth physics
        a = deep_acts[:, i]
        
        # Handle dead neurons (constant output) to avoid division by zero
        if np.std(a) < 1e-9:
            c_ts, c_alb, c_emiss, c_ntoa = 0,0,0,0
        else:
            c_ts = np.corrcoef(a, df['Ts'])[0,1]
            c_alb = np.corrcoef(a, df['Albedo'])[0,1]
            c_emiss = np.corrcoef(a, df['Emissivity'])[0,1]
            c_ntoa = np.corrcoef(a, df['N_toa'])[0,1]
            
        results.append({
            'neuron': i,
            'corr_Ts': c_ts,
            'corr_Albedo': c_alb,
            'corr_Emissivity': c_emiss,
            'corr_N_toa': c_ntoa
        })
    
    df_deep = pd.DataFrame(results)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(df_deep[['corr_Ts', 'corr_Albedo', 'corr_Emissivity', 'corr_N_toa']].values, 
               cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(4), ['Ts', 'Albedo', 'Emissivity', 'N_toa'])
    plt.yticks(range(n_neurons))
    plt.xlabel("Physical Variable")
    plt.ylabel("Neuron Index")
    plt.title('DEEP LAYER (relu_3) Specialization\n(Note the increased complexity)')
    
    heatmap_path = os.path.join(FIGURES_DIR, 'deep_layer_correlations.png')
    plt.savefig(heatmap_path)
    print(f"Saved deep correlation map to {heatmap_path}")

if __name__ == "__main__":
    deep_interpretability_analysis()
