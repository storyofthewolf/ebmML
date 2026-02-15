#!/bin/bash

# ==============================================================================
# CLIMATE AI LAB - AUTOMATED ANALYSIS PIPELINE
# ==============================================================================
# This script executes the full suite of interpretability tools.
# It assumes:
#   1. You are running from the project root.
#   2. 'climate_nn.py' has already been run to train the model.
#   3. 'config.py' points to the correct model file.
# ==============================================================================

# 1. Setup
# Stop execution if any script fails
set -e 

echo "=================================================================="
echo "Starting Automated Analysis Pipeline"
echo "Target Model: $(grep "ACTIVE_MODEL_FILENAME =" config.py | cut -d"'" -f2)"
echo "=================================================================="

# Create output directories if they don't exist
mkdir -p figures
mkdir -p networks

# 2. Run the Core Analysis Scripts
# We use 'python -m scripts.script_name' or direct path execution depending on your setup.
# Here we execute by path, relying on the sys.path.append header we added.

echo ""
echo "[1/5] Mapping Loss Landscape..."
python scripts/loss_landscape.py

echo ""
echo "[2/5] Tracing Ice-Albedo Circuit..."
python scripts/trace_circuit.py

echo ""
echo "[3/5] Tracing Greenhouse/Thermostat Circuit..."
python scripts/inspect_greenhouse_output.py
python scripts/inspect_greenhouse_sensors.py

echo ""
echo "[4/5] Generating Neuron Response Curves..."
python scripts/plot_neuron_3_response.py

echo ""
echo "[5/5] Performing Deep Interpretability Check..."
python scripts/deep_interp.py
python scripts/neuron_ablation.py

# 3. Completion
echo ""
echo "=================================================================="
echo "ANALYSIS COMPLETE"
echo "All figures have been updated in the /figures directory."
echo "=================================================================="
