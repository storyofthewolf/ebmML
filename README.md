# Mechanistic Interpretability Lab: EBM Climate Emulator

This repository contains a suite of tools designed to "open the black box" of a neural network climate emulator. Using Energy Balance Models (EBMs) as a ground-truth laboratory, these scripts demonstrate how to identify, trace, and causally verify physical circuits—such as the Ice-Albedo and Greenhouse feedbacks—within a trained model.

## 🛠 Operation Overview

The laboratory operates in three phases:
1.  **Generation & Training**: Creating a physically consistent dataset and training a "bottlenecked" architecture to force feature disentanglement.
2.  **Discovery**: Using correlation spectroscopy to find neurons that have spontaneously evolved into physical sensors (e.g., Ice Detectors).
3.  **Validation**: Performing causal ablation and physics counterfactual sweeps to prove the model is using causal physical logic rather than spurious correlations.

---

## 📁 Project Structure

```
├── config.py                 # Central configuration: paths, experiment selection
├── climate_nn.py             # Data loading, model definition, training
├── interpretability.py       # Analysis library: all interpretability logic and plotting
│
├── analyze.py                # CLI: correlation spectroscopy across all neurons/layers
├── probe.py                  # CLI: single-neuron deep dive (weights, scatter, distribution)
├── sweep.py                  # CLI: physics counterfactual sweeps over input features
├── ablate.py                 # CLI: causal ablation studies, neuron importance ranking
│
├── experiments/              # Experiment specifications (YAML)
│   ├── ebm_0d_v1.yaml
│   └── ...
│
├── ebm_models/               # Energy Balance Model implementations
│   ├── ebm_0d_model_v1.py
│   └── ebm_0d_model_v2.py
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
ACTIVE_EXPERIMENT = 'ebm_0d_v1'  # or 'ebm_0d_v2', etc.
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
| **`climate_nn.py`** | Schema-driven data loading (`ClimateDataset`), neural network definition (`ClimateMLP` with activation-agnostic hooks), and training utilities. Run directly to train a model. |
| **`interpretability.py`** | Master analysis library. Contains `InterpretabilityAnalyzer` (correlation spectroscopy, ablation, sweeps) and all plotting utilities. Extended here as new analysis methods are added. |

### CLI Analysis Scripts

These are the primary user-facing tools. All scripts accept `--experiment` to override the active experiment and `--no-plots` for text-only output.

| Script | Purpose |
|--------|---------|
| **`analyze.py`** | Correlation spectroscopy: correlates all neuron activations with all physics metadata variables across every layer. The starting point for any interpretability session. |
| **`probe.py`** | Single-neuron deep dive: activation distribution, scatter plots vs all physics variables, and incoming weight profile for one specified neuron. |
| **`sweep.py`** | Physics counterfactual sweep: holds all inputs fixed at baseline, sweeps one feature across its data range, plots network output response. Tests whether learned relationships match physical intuition. Can simultaneously track neuron activations during the sweep. |
| **`ablate.py`** | Causal ablation study: zeros out neurons via forward hooks and measures output change. Two modes — `single` (ablate a specific set of neurons together) and `rank` (ablate each neuron individually to produce a causal importance ranking). |

### Data Generation

| Script | Purpose |
|--------|---------|
| **`ebm_models/ebm_0d_model_v1.py`** | 0D energy balance model, produces dT/dt for arbitrary timestep |
| **`ebm_models/ebm_0d_model_v2.py`** | 0D energy balance model, run to equilibrium Ts |

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

### 4. Discover Neuron Specializations

```bash
python analyze.py
python analyze.py --threshold 0.6        # stricter specialization cutoff
python analyze.py --layer act_0          # single layer only
```
*Computes neuron-physics correlations and generates heatmap + distribution plots in `figures/`. Prints specialization summary to terminal.*

### 5. Probe Individual Neurons

```bash
python probe.py --layer act_0 --neuron 3
python probe.py --layer act_0 --neuron 3 --samples 10000
```
*Deep dive into a specific neuron: activation histogram, scatter vs all physics variables, and incoming weight profile.*

### 6. Test Physical Understanding

```bash
python sweep.py --feature S0
python sweep.py --feature S0 --track act_0 act_1    # watch neurons during sweep
python sweep.py --feature co2_ppm --baseline median
```
*Sweeps one input across its physical range. Tests whether the network responds to forcing variables the way physics dictates.*

### 7. Prove Causality

```bash
# Rank all neurons by causal importance
python ablate.py --layer act_0 --mode rank

# Test a specific neuron or group
python ablate.py --layer act_0 --neurons 3 --mode single
python ablate.py --layer act_0 --neurons 2 5 --mode single
```
*Cross-reference with `analyze.py` results: neurons that are both highly correlated AND high-impact on ablation are causally essential. Neurons that correlate but don't ablate are passengers.*

---

## 🔬 Recommended Interpretability Workflow

For each new trained model, the recommended analysis sequence is:

1. **`analyze.py`** — identify which neurons correlate with which physics variables
2. **`ablate.py --mode rank`** — establish causal importance ranking for the same layer
3. **Cross-reference** — do the high-correlation neurons match the high-ablation-impact neurons? Mismatches are scientifically interesting.
4. **`probe.py`** — drill into specific neurons of interest (weight profile + scatter plots)
5. **`sweep.py`** — verify that input-output relationships match physical intuition; use `--track` to watch candidate neurons respond in real time

---

## 🔧 Key Design Principles

### Library / CLI Separation

`interpretability.py` is the master analysis library — all analysis logic and plotting utilities live here. The four CLI scripts (`analyze.py`, `probe.py`, `sweep.py`, `ablate.py`) are thin callers that handle argument parsing and invoke library functions. When new analysis capabilities are needed, they are added to `interpretability.py` first, then exposed via a CLI script.

### Schema-Driven Flexibility

The `data_schema` in each experiment YAML defines:
- **features**: Input columns for the neural network
- **targets**: Output columns for the neural network
- **physics_meta**: Columns preserved for interpretability analysis (not used in training)

This allows the same codebase to handle 0D EBMs, 1D Budyko-Sellers models, or any future extensions without code changes.

### Activation-Agnostic Hooks

`ClimateMLP` automatically registers forward hooks on any activation layer (ReLU, Tanh, GELU, etc.). Analysis scripts can inspect activations from any layer without knowing the specific activation function used. Ablation hooks use the same infrastructure for non-destructive interventions.

### Self-Contained Checkpoints

Saved model checkpoints include the full experiment configuration, enabling complete reconstruction and analysis without external dependencies:
- Model weights
- Data scalers
- Experiment config (schema, architecture, training params)
- Activation layer metadata

---

## 🔬 Research Context

This work was developed by **Eric Theodore Wolf** as a case study in **neural network mechanistic interpretability** using elementary climate calculations.

The project explores whether neural networks trained on climate model outputs learn genuine physical understanding (compositional structure, conservation laws) versus sophisticated pattern matching. Methods include correlation spectroscopy, physics counterfactual sweeps, circuit tracing, ablation studies, and causal verification.
