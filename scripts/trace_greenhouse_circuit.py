import torch
import os
from climate_nn import load_model_from_checkpoint

def trace_greenhouse_circuit():
    checkpoint_path = '../networks/climate_model.pt'
    if not os.path.exists(checkpoint_path):
        print("Model file not found.")
        return
    
    try:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Target neurons from L1 (Greenhouse Sensors)
    gh_sensors = [0, 6, 7]
    
    # Weights shape: [L2_neurons, L1_neurons] -> [8, 8]
    l2_weights = model.network[2].weight.detach().numpy()
    
    print("="*45)
    print("TRACING GREENHOUSE SIGNAL TO LAYER 2")
    print("="*45)
    
    for sensor in gh_sensors:
        # Find the strongest connection for each sensor
        weights_from_sensor = l2_weights[:, sensor]
        top_l2_idx = abs(weights_from_sensor).argmax()
        strength = weights_from_sensor[top_l2_idx]
        
        print(f"L1 Neuron {sensor} -> Primary Consumer: L2 Neuron {top_l2_idx}")
        print(f"  Connection Strength: {strength:.4f}")
    print("="*45)

if __name__ == "__main__":
    trace_greenhouse_circuit()
