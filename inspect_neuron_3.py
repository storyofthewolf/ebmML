import torch
import os
from climate_nn import ClimateMLP

def inspect_neuron_3():
    checkpoint_path = 'networks/climate_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print("Error: Model file not found.")
        return

    # 1. Load the model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Extract Weights and Bias for Layer 1, Neuron 3
    # Weights shape is [8 neurons, 3 inputs]
    weights = model.network[0].weight[3].detach().numpy()
    bias = model.network[0].bias[3].detach().numpy()
    
    # 3. Print the Equation
    print("="*40)
    print("THE LOGIC OF NEURON 3 (Ice Detector)")
    print("="*40)
    print(f"Bias Term: {bias:.4f}")
    print(f"Weight (Ts):      {weights[0]:.4f}")
    print(f"Weight (logCO2):  {weights[1]:.4f}")
    print(f"Weight (S0):      {weights[2]:.4f}")
    print("-" * 40)
    
    # Interpret the 'Activation Formula'
    # Activation = ReLU( (W_Ts * Ts) + (W_CO2 * CO2) + (W_S0 * S0) + Bias )
    # Note: These weights apply to the SCALED inputs.
    
    print("\nPhysical Interpretation:")
    if weights[0] < 0:
        print("-> This neuron fires MORE when Temperature is COLD (Negative Weight on Ts).")
    else:
        print("-> This neuron fires MORE when Temperature is HOT (Positive Weight on Ts).")
        
    if abs(weights[1]) < 0.05:
        print("-> This neuron largely IGNORES CO2 (Near-zero Weight).")
        
    print(f"-> This neuron's 'Sensing Threshold' is roughly: {-bias/weights[0]:.2f} (in scaled units)")
    print("="*40)

if __name__ == "__main__":
    inspect_neuron_3()
