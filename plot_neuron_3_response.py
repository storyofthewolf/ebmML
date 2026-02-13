import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from climate_nn_toy import ClimateMLP

def plot_neuron_response():
    checkpoint_path = 'toy_outputs/climate_model.pt'
    if not os.path.exists(checkpoint_path):
        print("Model file not found. Ensure you have trained the [8,8,8,8] model.")
        return

    # 1. Load the Model & Scalers
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = checkpoint['scaler_X']
    
    # 2. Create a Temperature Sweep (240K to 320K)
    Ts_sweep = np.linspace(240, 320, 200)
    
    # Keep CO2 and Solar constant (at typical values)
    log_co2_val = 3.0  # ~1000 ppm
    S0_val = 1361.0
    
    X_raw = np.zeros((len(Ts_sweep), 3))
    X_raw[:, 0] = Ts_sweep
    X_raw[:, 1] = log_co2_val
    X_raw[:, 2] = S0_val
    
    # Scale inputs using the same scaler used during training
    X_scaled = torch.FloatTensor(scaler_X.transform(X_raw))
    
    # 3. Get Activations
    with torch.no_grad():
        activations = model.get_hidden_activations(X_scaled)
    
    # Extract Neuron 3 from the first hidden layer (relu_0)
    neuron_3_acts = activations['relu_0'][:, 3].numpy()
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(Ts_sweep, neuron_3_acts, color='blue', linewidth=3, label='Neuron 3 Activation')
    
    # Draw the physical threshold from your EBM code
    plt.axvline(280, color='red', linestyle='--', alpha=0.5, label='Physical Ice-Melt Threshold (280K)')
    
    plt.fill_between(Ts_sweep, 0, neuron_3_acts, color='blue', alpha=0.1)
    
    plt.title("Neuron 3 Response Curve: The 'Ice Detector' Circuit", fontsize=14)
    plt.xlabel("Surface Temperature $T_s$ (K)", fontsize=12)
    plt.ylabel("Activation Intensity (ReLU Output)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Label the physical regimes
    plt.text(250, max(neuron_3_acts)*0.8, "ICE REGIME\n(Active Sensor)", color='blue', fontweight='bold')
    plt.text(285, max(neuron_3_acts)*0.1, "MELT REGIME\n(Sensor Off)", color='gray', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('toy_outputs/neuron_3_response_curve.png')
    print("Saved neuron response curve to toy_outputs/neuron_3_response_curve.png")

if __name__ == "__main__":
    plot_neuron_response()
