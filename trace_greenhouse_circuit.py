import torch
import os
from climate_nn_toy import ClimateMLP

def trace_greenhouse_circuit():
    checkpoint_path = 'toy_outputs/climate_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
