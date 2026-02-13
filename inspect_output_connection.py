import torch
import os
from climate_nn_toy import ClimateMLP

def inspect_output_connection():
    checkpoint_path = 'toy_outputs/climate_model.pt'
    if not os.path.exists(checkpoint_path):
        print("Model file not found.")
        return

    # 1. Load the model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Trace from Layer 2, Neuron 7 to the FINAL OUTPUTS
    # We have Layer 3 and Layer 4 still to go. For simplicity, 
    # let's see how Neuron 7 connects to the NEXT Layer (Layer 3)
    l3_weights = model.network[4].weight[:, 7].detach().numpy()
    
    print("="*40)
    print("SIGNAL DESTINATION: Layer 2, Neuron 7")
    print("="*40)
    print("Connections to Layer 3:")
    for i, w in enumerate(l3_weights):
        print(f"To Layer 3, Neuron {i}: {w:.4f}")
        
    # 3. Final Output Layer Weights (Layer 4 -> Targets)
    # Output 0: Ts_next, Output 1: OLR, Output 2: ASR
    output_weights = model.network[8].weight.detach().numpy()
    
    print("-" * 40)
    print("FINAL OUTPUT BIASES (The 'Tendency'):")
    print(f"Ts_next weight effect: {output_weights[0, :].mean():.4f}")
    print(f"ASR weight effect:      {output_weights[2, :].mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    inspect_output_connection()
