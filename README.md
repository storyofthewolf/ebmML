# Mechanistic Interpretability Lab: EBM Climate Emulator

This repository contains a suite of tools designed to "open the black box" of a neural network climate emulator. Using Energy Balance Models (EBMs) as a ground-truth laboratory, these scripts demonstrate how to identify, trace, and causally verify physical circuits—such as the Ice-Albedo and Greenhouse feedbacks—within a trained model.

## 🛠 Operation Overview

The laboratory operates in three phases:
1.  **Generation & Training**: Creating a physically consistent dataset and training a "bottlenecked" architecture to force feature disentanglement.
2.  **Discovery**: Using correlation spectroscopy to find neurons that have spontaneously evolved into physical sensors (e.g., Ice Detectors).
3.  **Validation**: Performing "Neuron Surgery" (ablation) and weight-tracing to prove the model is using causal physical logic rather than spurious correlations.

---

## 📁 Project Structure

```
├── config.py                 # Central configuration: paths, experiment selection
├── climate_nn.py             # Data loading, model definition, training
├── interpretability.py       # Neuron correlation analysis and plotting
│
├── experiments/              # Experiment specifications (YAML)
│   ├── ebm_0d_v1.yaml
│   ├── ebm_0d_v1_deep.yaml
│   └── ...
│
├── ebm_models/               # Energy Balance Model implementations
│   ├── ebm_0d_model_v1.py
│   └── ebm_0d_model_v2.py
│
├── scripts/                  # Analysis and diagnostic scripts
│   ├── plot_neuron_3_response.py
│   ├── inspect_neuron_logic.py
│   ├── trace_circuit.py
│   ├── inspect_greenhouse_output.py
│   └── neuron_ablation.py
│
├── training_sets/            # Generated EBM training data (CSV)
├── networks/                 # Saved model checkpoints (.pt)
└── figures/                  # Generated plots and visualizations
```

---

## ⚙️ Configuration System

The project uses a **schema-driven configuration** that separates experiment definitions from code. This enables flexible exploration of different EBM complexities and neural network architectures.

### Experiment Files

Each experiment is defined in a self-contained YAML file in `experiments/`:

```yaml
# experiments/ebm_0d_v1.yaml
description: "0D EBM with ice-albedo and linearized greenhouse"

data_schema:
  features: [Ts, log_pCO2, S0]
  targets: [Ts_next, OLR, ASR]
  physics_meta: [Albedo, Emissivity, N_toa, Ts]

model_config:
  hidden_dims: [4, 4, 4, 4]
  activation: ReLU

training_config:
  epochs: 20
  batch_size: 1024
  learning_rate: 0.001
  ...

files:
  data: ebm_0d_model_v1_climate_data.csv
  model: ebm_0d_model_v1_nn.pt
```

### Switching Experiments

To switch the active experiment, edit one line in `config.py`:

```python
ACTIVE_EXPERIMENT = 'ebm_0d_v1'  # or 'ebm_0d_v1_deep', etc.
```

All scripts automatically use the active experiment's schema, paths, and settings.

### Adding New Experiments

1. Create a new YAML file in `experiments/` (e.g., `ebm_1d_budyko.yaml`)
2. Define the complete specification (data schema, model config, file names)
3. Set `ACTIVE_EXPERIMENT` to the new experiment name

No Python code changes required—the new experiment is auto-discovered.

---

## 📄 Module Reference

### Core Modules

| Module | Purpose |
|--------|---------|
| **`config.py`** | Central configuration hub. Loads experiment YAML files, computes paths, provides `get_experiment()` and related helpers. |
| **`climate_nn.py`** | Schema-driven data loading (`ClimateDataset`), neural network definition (`ClimateMLP` with activation-agnostic hooks), and training utilities. |
| **`interpretability.py`** | `InterpretabilityAnalyzer` for neuron-physics correlations, plus plotting functions (`plot_correlations`, `plot_correlation_distributions`). |

### Data Generation

| Script | Purpose |
|--------|---------|
| **`ebm_models/ebm_0d_model_v1.py`** | 0D energy balance model, produces dT/dt for arbitrary timestep |
| **`ebm_models/ebm_0d_model_v2.py`** | 0D energy balance model, run to equilibrium Ts |

### Diagnostic & Visualization Scripts

| Script | Purpose |
|--------|---------|
| **`plot_neuron_3_response.py`** | Response curve for "Ice Detector" neuron. Visualizes the activation switch at the 280K melt threshold. |
| **`inspect_neuron_logic.py`** | Extracts weights/biases for a neuron, translates to human-readable activation formula. |
| **`trace_circuit.py`** | Traces causal wiring from Layer 1 sensors to Layer 2 logic hubs. |
| **`inspect_greenhouse_output.py`** | Maps connections from internal logic hubs to final outputs (T_next, OLR). |
| **`neuron_ablation.py`** | Targeted neuron ablation to prove causal importance of feedback circuits. |

---

## 🚀 How to Use

### 1. Prerequisites

```bash
pip install torch numpy pandas matplotlib scikit-learn pyyaml
```

### 2. Generate EBM Data

```bash
python ebm_models/ebm_0d_model_v1.py
```
*Creates training data in `training_sets/`*

### 3. Train the Model

```bash
python climate_nn.py
```
*Trains the neural network using the active experiment configuration. Saves checkpoint to `networks/`.*

### 4. Run Interpretability Analysis

```bash
python interpretability.py
```
*Computes neuron-physics correlations and generates plots in `figures/`.*

### 5. Identify Sensors

Review `figures/neuron_correlations.png` to find which neurons correlate with **Albedo** (Ice) or **Emissivity** (Greenhouse).

### 6. Verify the Logic

Run inspector scripts to examine individual neurons:
```bash
python scripts/inspect_neuron_logic.py
python scripts/plot_neuron_3_response.py
```

### 7. Trace the Circuit

Map signal flow from sensors to outputs:
```bash
python scripts/trace_circuit.py
python scripts/inspect_greenhouse_output.py
```

### 8. Prove Causality

Test model behavior with ablated neurons:
```bash
python scripts/neuron_ablation.py
```

---

## 🔧 Key Design Principles

### Schema-Driven Flexibility

The `data_schema` in each experiment YAML defines:
- **features**: Input columns for the neural network
- **targets**: Output columns for the neural network  
- **physics_meta**: Columns preserved for interpretability analysis (not used in training)

This allows the same codebase to handle 0D EBMs, 1D Budyko-Sellers models, or any future extensions without code changes.

### Activation-Agnostic Hooks

`ClimateMLP` automatically registers forward hooks on any activation layer (ReLU, Tanh, GELU, etc.). Analysis scripts can inspect activations from any layer without knowing the specific activation function used.

### Self-Contained Checkpoints

Saved model checkpoints include the full experiment configuration, enabling complete reconstruction and analysis without external dependencies:
- Model weights
- Data scalers  
- Experiment config (schema, architecture, training params)
- Activation layer metadata

---

## 🔬 Research Context

This work was developed by **Eric Theodore Wolf** as a case study in **neural network mechanistic interpretability** using elementary climate calculations.

The project explores whether neural networks trained on climate model outputs learn genuine physical understanding (compositional structure, conservation laws) versus sophisticated pattern matching. Methods include correlation spectroscopy, circuit tracing, ablation studies, and causal verification.
