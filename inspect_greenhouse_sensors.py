import torch
import os
from climate_nn import ClimateMLP

def inspect_greenhouse_sensors():
    checkpoint_path = 'outputs/climate_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
