# Mechanistic Interpretability Lab: EBM Climate Emulator

This repository contains a suite of tools designed to "open the black box" of a neural network climate emulator. Using a 0-D Energy Balance Model (EBM) as a ground-truth laboratory, these scripts demonstrate how to identify, trace, and causally verify physical circuits—such as the Ice-Albedo and Greenhouse feedbacks—within a trained model.

## 🛠 Operation Overview

The laboratory operates in three phases:
1.  **Generation & Training**: Creating a physically consistent dataset and training a "bottlenecked" architecture to force feature disentanglement.
2.  **Discovery**: Using correlation spectroscopy to find neurons that have spontaneously evolved into physical sensors (e.g., Ice Detectors).
3.  **Validation**: Performing "Neuron Surgery" (ablation) and weight-tracing to prove the model is using causal physical logic rather than spurious correlations.

---

## 📄 Script Directory

### 1. Data Generation
* **`ebm_models/ebm_0d_model_v1.py -- 0D energy balance model, produces dT/dt for abitrary timestep
* **`ebm_models/ebm_0d_model_v2.py -- 0D energy balance model, run to convergence Ts

### 2. Core Model & Training
* **`climate_nn.py`**: The master script. It contains, the `ClimateMLP` architecture (`[8, 8, 8, 8]`), and the `InterpretabilityAnalyzer` which produces the initial neuron-physical correlation heatmaps.

### 3. Diagnostic & Visualization Tools
* **`plot_neuron_3_response.py`**: Generates a response curve for the "Ice Detector" neuron. It visualizes the activation "switch" as the temperature crosses the 280K melt threshold.
* **`inspect_neuron_logic.py`**: Extracts the raw synaptic weights and biases for a specific neuron. It translates the model's internal linear algebra into a human-readable "activation formula".

### 4. Circuit Tracing
* **`trace_circuit.py`**: Traces the "causal wiring" from Layer 1 sensors to Layer 2 logic hubs. This identifies "consumer" neurons that aggregate signals to form higher-level physical concepts.
* **`inspect_greenhouse_output.py`**: Maps the final leg of the circuit, showing how internal logic hubs connect to the final physical outputs like $T_{next}$ and Outgoing Longwave Radiation (OLR).

### 5. Causal Verification
* **`neuron_ablation.py`**: Performs targeted neuron ablation. It "lobotomizes" specific feedback circuits (like the Ice-Albedo loop) and plots the resulting climate sensitivity to prove the importance of those neurons to the model's physical accuracy.

---

## 🚀 How to Use

### 1. Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### 2. Generate EBM Data
```bash
python ebm_models/emacs ebm_0d_model_v1.py
```
*Creates training data ./training_sets/ebm_0d_model_v1_climate_data_1M.csv`


### 3. Generate Data and Train
```bash
python climate_nn_toy.py
```
*Trains neural network model and saves to `networks/climate_model.pt`.*

### 4. Identify Sensors
Review `figures/neuron_correlations.png` to find which neurons correlate with **Albedo** (Ice) or **Emissivity** (Greenhouse).

### 5. Verify the Logic
Run the inspector scripts to see the "mind" of the identified neurons:
```bash
python inspect_neuron_logic.py
python plot_neuron_3_response.py
```

### 6. Trace the Circuit
Map the flow from the sensor to the final temperature prediction:
```bash
python trace_circuit.py
python inspect_greenhouse_output.py
```

### 7. Prove Causality
Test what happens to the "physics" of the model when you remove its sensors:
```bash
python neuron_ablation.py
```

---

## 🔬 Research Context
This work was developed by **Eric Theodore Wolf** as a case study in **neural network mechanistic intrepretability** using elementary climate calculations.


