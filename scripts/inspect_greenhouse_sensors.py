import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, FIGURES_DIR
from climate_nn import load_model_from_checkpoint

def inspect_greenhouse_sensors():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    try:
        model, checkpoint = load_model_from_checkpoint(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # We'll check neurons 0, 6, and 7
    target_neurons = [0, 6, 7]
    
    print("="*45)
    print("THE LOGIC OF GREENHOUSE SENSORS (L1)")
    print("="*45)
    
    for n in target_neurons:
        weights = model.network[0].weight[n].detach().numpy()
        bias = model.network[0].bias[n].detach().numpy()
        print(f"NEURON {n}:")
        print(f"  Weight (Ts):      {weights[0]:.4f}")
        print(f"  Weight (logCO2):  {weights[1]:.4f}")
        print(f"  Weight (S0):      {weights[2]:.4f}")
        print(f"  Bias:             {bias:.4f}")
        print("-" * 45)

if __name__ == "__main__":
    inspect_greenhouse_sensors()
