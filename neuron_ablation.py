import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from climate_nn import ClimateMLP, load_and_split_data

def run_dual_ablation():
    # 1. Setup
    checkpoint_path = 'networks/climate_model.pt'
    device = 'cpu'
    
    # Load Model & Scalers
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_base = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model_base.load_state_dict(checkpoint['model_state_dict'])
    model_base.eval()

    # 2. Perform Surgeries
    # Surgery A: Remove Ice Feedback (Neurons 3 and 5)
    model_ice_free = copy.deepcopy(model_base)
    # Surgery B: Remove Greenhouse Feedback (Neurons 0, 6, and 7)
    model_gh_free = copy.deepcopy(model_base)
    
    with torch.no_grad():
        # ICE ABLATION
        for n in [3, 5]:
            model_ice_free.network[0].weight[n, :] = 0
            model_ice_free.network[0].bias[n] = 0
            
        # GREENHOUSE ABLATION
        for n in [0, 6, 7]:
            model_gh_free.network[0].weight[n, :] = 0
            model_gh_free.network[0].bias[n] = 0

    # 3. Physical Experiment: Temperature Sweep
    T_range = np.linspace(220, 360, 100)
    log_co2 = np.full_like(T_range, 3.0) 
    S0 = np.full_like(T_range, 1361.0)
    
    scaler_X = checkpoint['scaler_X']
    X_raw = np.stack([T_range, log_co2, S0], axis=1)
    X_scaled = torch.FloatTensor(scaler_X.transform(X_raw))

    with torch.no_grad():
        p_h = model_base(X_scaled).numpy()
        p_ice = model_ice_free(X_scaled).numpy()
        p_gh = model_gh_free(X_scaled).numpy()

    # Inverse scale (Index 0 is Ts_next)
    scaler_Y = checkpoint['scaler_Y']
    dT_h = scaler_Y.inverse_transform(p_h)[:, 0] - T_range
    dT_ice = scaler_Y.inverse_transform(p_ice)[:, 0] - T_range
    dT_gh = scaler_Y.inverse_transform(p_gh)[:, 0] - T_range

    # 4. Visualization
    plt.figure(figsize=(12, 7))
    plt.plot(T_range, dT_h, 'k-', label='Healthy (All Feedbacks)', linewidth=2.5)
    plt.plot(T_range, dT_ice, 'r--', label='Ablated: No Ice Feedback (3,5)', linewidth=2)
    plt.plot(T_range, dT_gh, 'g--', label='Ablated: No Greenhouse Feedback (0,6,7)', linewidth=2)
    
    plt.axhline(0, color='black', alpha=0.3)
    plt.axvline(280, color='red', alpha=0.2, label='Ice Melt Threshold')
    
    plt.xlabel('Surface Temperature (K)')
    plt.ylabel('Net Heating Rate (K/step)')
    plt.title('Mechanistic Interpretability: Impact of Targetted Neuron Removal')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('figures/dual_ablation_results.png')
    plt.show()

if __name__ == "__main__":
    run_dual_surgery()
