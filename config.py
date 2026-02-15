# config.py
import os

# =========================================================
# 1. MODEL HYPERPARAMETERS (The Recipe)
# =========================================================
# These settings define the architecture for NEW models you train.
DEFAULT_MODEL_CONFIG = {
    'input_dim': 3,
    'hidden_dims': [8, 8, 8, 8],
    'output_dim': 3,
    'activation': 'ReLU'  # easy toggle: 'Tanh', 'GELU', 'Sigmoid'
}

# =========================================================
# 2. EXPERIMENT SETTINGS (The Focus)
# =========================================================
# Change these filenames to switch the entire lab's focus 
# without editing individual analysis scripts.

ACTIVE_MODEL_FILENAME = 'climate_model.pt'
ACTIVE_DATA_FILENAME = 'ebm_0d_model_v1_climate_data_1M.csv'

# =========================================================
# 3. AUTOMATIC PATH GENERATION (The Address Book)
# =========================================================
# This automatically detects the project root, no matter where
# you run the script from (root vs. scripts/ folder).

# Get the directory where THIS file (config.py) lives
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for all scripts to use
MODEL_PATH = os.path.join(PROJECT_ROOT, 'networks', ACTIVE_MODEL_FILENAME)
DATA_PATH = os.path.join(PROJECT_ROOT, 'training_sets', ACTIVE_DATA_FILENAME)
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# Ensure directories exist (optional safety check)
os.makedirs(os.path.join(PROJECT_ROOT, 'networks'), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"--> CONFIG LOADED: Target Model is '{ACTIVE_MODEL_FILENAME}'")


