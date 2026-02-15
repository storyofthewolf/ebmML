import sys
import torch
import os

# Add parent directory to path to find config.py and climate_nn.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_PATH
from climate_nn import load_model_from_checkpoint

def inspect_greenhouse_output():

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return

    try:
        model, checkpoint = load_model_from_checkpoint(MODEL_PATH)
        print(f"Model file not found at: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Trace from L2 Neuron 6 toward the FINAL OUTPUTS
    # Output 0: Ts_next, Output 1: OLR, Output 2: ASR
    output_weights = model.network[8].weight.detach().numpy()
    
    # We trace from L2:N6 to L3 to L4 (the outputs)
    l3_weights_from_l2_n6 = model.network[4].weight[:, 6].detach().numpy()
    
    print("="*45)
    print("HUB OUTPUT: Layer 2, Neuron 6 (The Thermostat)")
    print("="*45)
    print("Connections to Layer 3 neurons:")
    for i, w in enumerate(l3_weights_from_l2_n6):
        print(f"  To L3 Neuron {i}: {w:.4f}")
        
    print("-" * 45)
    print("FINAL OUTPUT TENDENCIES (Average Impact):")
    # This shows if the model generally treats this pathway as warming or cooling
    print(f"  Effect on OLR (Cooling): {output_weights[1, :].mean():.4f}")
    print(f"  Effect on Ts_next:       {output_weights[0, :].mean():.4f}")
    print("="*45)

if __name__ == "__main__":
    inspect_greenhouse_output()
