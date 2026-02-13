import torch
import os
import pandas as pd
from climate_nn_toy import ClimateMLP

def trace_circuit():
    checkpoint_path = 'toy_outputs/climate_model.pt'
    if not os.path.exists(checkpoint_path):
        print("Model file not found.")
        return

    # 1. Load the model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Extract Weights for Layer 2 (network[2])
    # Shape: [8 neurons in Layer 2, 8 neurons in Layer 1]
    # We want index 3 from the Layer 1 dimension
    l2_weights = model.network[2].weight[:, 3].detach().numpy()
    
    # 3. Analyze the connections
    print("="*40)
    print("TRACING SIGNAL: Layer 1, Neuron 3 (Ice Sensor)")
    print("="*40)
    print("Weight connections to Layer 2 neurons:")
    
    connections = []
    for i, w in enumerate(l2_weights):
        print(f"To Layer 2, Neuron {i}: {w:.4f}")
        connections.append({'neuron_l2': i, 'weight': w})
    
    df_conn = pd.DataFrame(connections)
    top_consumer = df_conn.iloc[df_conn['weight'].abs().idxmax()]
    
    print("-" * 40)
    print(f"PRIMARY CONSUMER: Layer 2, Neuron {int(top_consumer['neuron_l2'])}")
    print(f"Connection Strength: {top_consumer['weight']:.4f}")
    
    # 4. Physical Interpretation
    if top_consumer['weight'] > 0:
        print("\nInterpretation: This is an EXCITATORY connection.")
        print("When the Ice Sensor fires, it activates this Layer 2 neuron.")
    else:
        print("\nInterpretation: This is an INHIBITORY connection.")
        print("When the Ice Sensor fires, it suppresses this Layer 2 neuron.")
    print("="*40)

if __name__ == "__main__":
    trace_circuit()
