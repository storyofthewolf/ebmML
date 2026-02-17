# config.py
"""
Climate NN Configuration
========================
Central configuration for all experiments.

Experiments are defined in YAML files in the experiments/ directory.
Each YAML file contains the complete specification for one experiment.

To switch experiments, change ACTIVE_EXPERIMENT.
To add new experiments, create a new YAML file in experiments/.
"""

import os
import yaml


# =========================================================
# 1. PROJECT STRUCTURE (The Address Book)
# =========================================================
# Automatically detects project root from this file's location.

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory structure
NETWORKS_DIR = os.path.join(PROJECT_ROOT, 'networks')
TRAINING_SETS_DIR = os.path.join(PROJECT_ROOT, 'training_sets')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

# Ensure directories exist
for directory in [NETWORKS_DIR, TRAINING_SETS_DIR, FIGURES_DIR, SCRIPTS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# =========================================================
# 2. ACTIVE EXPERIMENT SELECTION
# =========================================================
# Change this single line to switch the entire project's focus.
# Must match a YAML filename (without .yaml extension) in experiments/

ACTIVE_EXPERIMENT = 'ebm_0d_v1'


# =========================================================
# 3. EXPERIMENT LOADING (The Librarian)
# =========================================================

def _load_experiment_yaml(name):
    """
    Load an experiment specification from its YAML file.
    
    Args:
        name: Experiment name (filename without .yaml extension)
        
    Returns:
        dict: Raw experiment specification from YAML
    """
    yaml_path = os.path.join(EXPERIMENTS_DIR, f"{name}.yaml")
    
    if not os.path.exists(yaml_path):
        available = list_experiment_names()
        raise ValueError(
            f"Experiment '{name}' not found.\n"
            f"Expected file: {yaml_path}\n"
            f"Available experiments: {available}"
        )
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def list_experiment_names():
    """
    Scan experiments/ directory and return list of available experiment names.
    
    Returns:
        list: Experiment names (YAML filenames without extension)
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        return []
    
    names = []
    for filename in os.listdir(EXPERIMENTS_DIR):
        if filename.endswith('.yaml'):
            names.append(filename[:-5])  # Strip .yaml extension
    
    return sorted(names)


def get_experiment(name=None):
    """
    Retrieve a complete experiment specification.
    
    Loads the YAML file and computes derived values (paths, dimensions).
    
    Args:
        name: Experiment name. If None, uses ACTIVE_EXPERIMENT.
        
    Returns:
        dict: Full experiment specification with computed paths and dimensions.
    """
    if name is None:
        name = ACTIVE_EXPERIMENT
    
    # Load from YAML
    exp = _load_experiment_yaml(name)
    
    # Compute full paths from filenames
    exp['paths'] = {
        'data': os.path.join(TRAINING_SETS_DIR, exp['files']['data']),
        'model': os.path.join(NETWORKS_DIR, exp['files']['model']),
        'figures': FIGURES_DIR,
    }
    
    # Compute input/output dimensions from schema
    schema = exp['data_schema']
    exp['model_config']['input_dim'] = len(schema['features'])
    exp['model_config']['output_dim'] = len(schema['targets'])
    
    # Add experiment name for reference
    exp['name'] = name
    
    return exp


def get_model_config(name=None):
    """Convenience function to get just the model configuration."""
    return get_experiment(name)['model_config']


def get_data_schema(name=None):
    """Convenience function to get just the data schema."""
    return get_experiment(name)['data_schema']


def get_paths(name=None):
    """Convenience function to get just the file paths."""
    return get_experiment(name)['paths']


def get_training_config(name=None):
    """Convenience function to get just the training configuration."""
    return get_experiment(name)['training_config']


def list_experiments():
    """List all available experiments with descriptions."""
    names = list_experiment_names()
    
    print("\nAvailable Experiments:")
    print("-" * 60)
    
    for name in names:
        try:
            exp = _load_experiment_yaml(name)
            description = exp.get('description', '(no description)')
            marker = " <-- ACTIVE" if name == ACTIVE_EXPERIMENT else ""
            print(f"  {name}: {description}{marker}")
        except Exception as e:
            print(f"  {name}: (error loading: {e})")
    
    print("-" * 60)


def validate_experiment(name=None):
    """
    Validate that an experiment's configuration is consistent.
    
    Checks:
    - YAML file exists and parses
    - All required keys present
    - Dimensions match schema lengths
    - Files exist (optional warning)
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if name is None:
        name = ACTIVE_EXPERIMENT
    
    errors = []
    warnings = []
    
    # Try to load experiment
    try:
        exp = get_experiment(name)
    except Exception as e:
        print(f"\n❌ Validation FAILED for '{name}':")
        print(f"   ERROR: Could not load experiment: {e}")
        return False
    
    # Check required keys
    required_keys = ['data_schema', 'model_config', 'training_config', 'files']
    for key in required_keys:
        if key not in exp:
            errors.append(f"Missing required key: {key}")
    
    # Check schema has required fields
    schema = exp.get('data_schema', {})
    for field in ['features', 'targets', 'physics_meta']:
        if field not in schema:
            errors.append(f"data_schema missing '{field}'")
    
    # Check model_config has required fields
    model_config = exp.get('model_config', {})
    for field in ['hidden_dims', 'activation']:
        if field not in model_config:
            errors.append(f"model_config missing '{field}'")
    
    # Check training_config has required fields
    training_config = exp.get('training_config', {})
    for field in ['epochs', 'batch_size', 'learning_rate']:
        if field not in training_config:
            errors.append(f"training_config missing '{field}'")
    
    # Check file existence (warnings only)
    paths = exp.get('paths', {})
    if 'data' in paths and not os.path.exists(paths['data']):
        warnings.append(f"Data file not found: {paths['data']}")
    
    # Report
    if errors:
        print(f"\n❌ Validation FAILED for '{name}':")
        for e in errors:
            print(f"   ERROR: {e}")
    else:
        print(f"\n✓ Validation PASSED for '{name}'")
    
    if warnings:
        for w in warnings:
            print(f"   WARNING: {w}")
    
    return len(errors) == 0


def validate_all_experiments():
    """Validate all experiments in the experiments/ directory."""
    names = list_experiment_names()
    
    print(f"\nValidating {len(names)} experiments...")
    
    results = {}
    for name in names:
        results[name] = validate_experiment(name)
    
    passed = sum(results.values())
    failed = len(results) - passed
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    return results


# =========================================================
# 4. BACKWARDS COMPATIBILITY
# =========================================================
# These exports maintain compatibility with existing scripts
# that import DEFAULT_MODEL_CONFIG, MODEL_PATH, DATA_PATH, FIGURES_DIR

_active = get_experiment()

DEFAULT_MODEL_CONFIG = _active['model_config']
MODEL_PATH = _active['paths']['model']
DATA_PATH = _active['paths']['data']
# FIGURES_DIR already defined above

# Also export schema for scripts that need it
DEFAULT_DATA_SCHEMA = _active['data_schema']


# =========================================================
# 5. STARTUP MESSAGE
# =========================================================

print(f"--> CONFIG LOADED: Active experiment is '{ACTIVE_EXPERIMENT}'")
print(f"    Model: {_active['files']['model']}")
print(f"    Data:  {_active['files']['data']}")
