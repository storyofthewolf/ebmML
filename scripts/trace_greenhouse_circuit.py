import torch
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, FIGURES_DIR
from climate_nn import load_model_from_checkpoint


def trace_greenhouse_circuit():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at: {MODEL_PATH}")
        return
    
    try:
        model, checkpoint = load_model_from_checkpoint(MODEL_PATH)
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
