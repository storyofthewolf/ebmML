import torch
import os
from climate_nn_toy import ClimateMLP

def inspect_greenhouse_output():
    checkpoint_path = 'toy_outputs/climate_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
