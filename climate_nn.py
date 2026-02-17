"""
Climate NN - Core Module
========================
Schema-driven neural network training for Energy Balance Models.

This module provides:
1. ClimateDataset: Flexible data loading driven by config schema
2. ClimateMLP: Neural network with activation-agnostic hooks for interpretability
3. Training utilities

Interpretability analysis has been moved to interpretability.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

from config import (
    get_experiment, 
    get_model_config, 
    get_data_schema, 
    get_paths,
    get_training_config,
    ACTIVE_EXPERIMENT
)


# =============================================================================
# 1. DATA LOADING (Schema-Driven)
# =============================================================================

class ClimateDataset(Dataset):
    """
    Dataset for energy balance model outputs.
    
    Schema-driven: reads feature/target/physics_meta column names from config,
    enabling flexible adaptation to different EBM complexity levels.
    
    Args:
        dataframe: pandas DataFrame containing the climate data
        data_schema: dict with 'features', 'targets', 'physics_meta' lists
        scaler_X: optional pre-fitted StandardScaler for features
        scaler_Y: optional pre-fitted StandardScaler for targets
    """
    
    def __init__(self, dataframe: pd.DataFrame,
                 data_schema: dict,
                 scaler_X=None, 
                 scaler_Y=None):
        
        self.df = dataframe.reset_index(drop=True)
        self.data_schema = data_schema
        
        # Validate that required columns exist
        self._validate_columns()
        
        # INPUTS: Features from schema
        self.feature_names = data_schema['features']
        X_raw = self.df[self.feature_names].values.astype(np.float32)
        
        # OUTPUTS: Targets from schema
        self.target_names = data_schema['targets']
        Y_raw = self.df[self.target_names].values.astype(np.float32)
        
        # Scaling
        if scaler_X is None:
            self.scaler_X = StandardScaler().fit(X_raw)
        else:
            self.scaler_X = scaler_X
            
        if scaler_Y is None:
            self.scaler_Y = StandardScaler().fit(Y_raw)
        else:
            self.scaler_Y = scaler_Y
            
        self.X = torch.FloatTensor(self.scaler_X.transform(X_raw))
        self.Y = torch.FloatTensor(self.scaler_Y.transform(Y_raw))
        
        # Physics metadata for interpretability analysis
        # Stored as dict of numpy arrays, keyed by column name
        self.physics_meta = {}
        for col in data_schema['physics_meta']:
            if col in self.df.columns:
                self.physics_meta[col] = self.df[col].values
            else:
                print(f"  Warning: physics_meta column '{col}' not found in data")
    
    def _validate_columns(self):
        """Check that all required columns exist in the dataframe."""
        missing = []
        
        for col in self.data_schema['features']:
            if col not in self.df.columns:
                missing.append(f"feature '{col}'")
                
        for col in self.data_schema['targets']:
            if col not in self.df.columns:
                missing.append(f"target '{col}'")
        
        if missing:
            available = list(self.df.columns)
            raise ValueError(
                f"Missing columns in data: {missing}\n"
                f"Available columns: {available}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_physics_meta_names(self):
        """Return list of available physics metadata variable names."""
        return list(self.physics_meta.keys())


def load_and_split_data(csv_path, data_schema, val_size=0.2, seed=42):
    """
    Load CSV and create Train/Val datasets using the provided schema.
    
    Args:
        csv_path: Path to the CSV file
        data_schema: dict with 'features', 'targets', 'physics_meta' lists
        val_size: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print(f"Loading {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}.\n"
            f"Did you run the EBM data generation script?"
        )
        
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} samples with columns: {list(df.columns)}")
    
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
    print(f"  Train: {len(train_df):,} samples, Val: {len(val_df):,} samples")
    
    train_ds = ClimateDataset(train_df, data_schema)
    val_ds = ClimateDataset(
        val_df, 
        data_schema,
        scaler_X=train_ds.scaler_X, 
        scaler_Y=train_ds.scaler_Y
    )
    
    return train_ds, val_ds


# =============================================================================
# 2. NEURAL NETWORK (Activation-Agnostic Hooks)
# =============================================================================

# Registry of activation functions for hook detection
ACTIVATION_MODULES = (
    nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU,
    nn.Tanh, nn.Sigmoid, nn.GELU, nn.SiLU, nn.Mish,
    nn.Softplus, nn.Softsign
)


class ClimateMLP(nn.Module):
    """
    Multi-layer perceptron with hooks for mechanistic interpretability.
    
    Features:
    - Dynamic architecture from config
    - Activation-agnostic hook registration
    - Stores activation metadata for analysis scripts
    
    Args:
        config: dict with 'input_dim', 'output_dim', 'hidden_dims', 'activation'
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get activation function class from torch.nn
        activation_name = config.get('activation', 'ReLU')
        if not hasattr(nn, activation_name):
            raise ValueError(
                f"Unknown activation '{activation_name}'. "
                f"Must be a valid torch.nn activation (e.g., 'ReLU', 'Tanh', 'GELU')"
            )
        activation_fn = getattr(nn, activation_name)
        self.activation_name = activation_name
        
        # Build network layers
        layers = []
        prev_dim = config['input_dim']
        
        for i, h_dim in enumerate(config['hidden_dims']):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, config['output_dim']))
        self.network = nn.Sequential(*layers)
        
        # Hook infrastructure
        self.activations = {}
        self.activation_layer_info = []  # Metadata about activation layers
        self._register_hooks()
        
    def _register_hooks(self):
        """
        Register forward hooks on all activation layers.
        Activation-agnostic: detects any module in ACTIVATION_MODULES.
        """
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        layer_idx = 0
        for i, layer in enumerate(self.network):
            if isinstance(layer, ACTIVATION_MODULES):
                # Name format: 'act_0', 'act_1', etc.
                name = f'act_{layer_idx}'
                layer.register_forward_hook(get_activation(name))
                
                # Store metadata for analysis scripts
                self.activation_layer_info.append({
                    'name': name,
                    'type': layer.__class__.__name__,
                    'network_index': i,
                    'layer_index': layer_idx,
                })
                layer_idx += 1

    def forward(self, x):
        return self.network(x)
    
    def get_hidden_activations(self, x):
        """
        Forward pass that returns all intermediate activations.
        
        Args:
            x: Input tensor
            
        Returns:
            dict: {layer_name: activation_tensor} for all activation layers
        """
        self.activations = {}
        _ = self.forward(x)
        return self.activations
    
    def get_activation_layer_names(self):
        """Return list of activation layer names (for analysis scripts)."""
        return [info['name'] for info in self.activation_layer_info]
    
    def get_activation_info(self):
        """Return full metadata about activation layers."""
        return self.activation_layer_info


# =============================================================================
# 3. MODEL LOADING UTILITY
# =============================================================================

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Reconstruct a trained model from a saved checkpoint.
    
    The checkpoint contains the full experiment config, enabling
    complete reconstruction without external dependencies.
    
    Args:
        checkpoint_path: Path to .pt file
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration (supports both old and new formats)
    config = checkpoint.get('config')
    if config is None:
        raise ValueError(f"No 'config' found in {checkpoint_path}.")
    
    # Handle nested config structure (new format)
    if 'model_config' in config:
        model_config = config['model_config']
    else:
        # Old format: config is the model_config directly
        model_config = config
    
    model = ClimateMLP(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, 
                epochs=20, lr=1e-3, device='cpu',
                print_every=5):
    """
    Train the model with Adam optimizer and MSE loss.
    
    Args:
        model: ClimateMLP instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        print_every: Print progress every N epochs
        
    Returns:
        dict: Training history with 'train_loss' and 'val_loss' lists
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training on {device} for {epochs} epochs...")
    print(f"  Architecture: {model.config['hidden_dims']}")
    print(f"  Activation: {model.activation_name}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                val_loss += loss_fn(model(X), Y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        if (epoch + 1) % print_every == 0:
            print(f"  Epoch {epoch+1:3d}: Train={avg_train:.6f}, Val={avg_val:.6f}")
            
    print(f"  Final: Train={history['train_loss'][-1]:.6f}, Val={history['val_loss'][-1]:.6f}")
    return history


def save_checkpoint(model, train_ds, experiment_config, save_path, history=None):
    """
    Save model checkpoint with full experiment context.
    
    The checkpoint includes everything needed for reconstruction and analysis:
    - Model weights
    - Scalers for data normalization
    - Full experiment configuration
    - Training history (optional)
    - Activation layer metadata
    
    Args:
        model: Trained ClimateMLP instance
        train_ds: Training ClimateDataset (for scalers)
        experiment_config: Full experiment dict from get_experiment()
        save_path: Where to save the .pt file
        history: Optional training history dict
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': train_ds.scaler_X,
        'scaler_Y': train_ds.scaler_Y,
        'config': experiment_config,
        'activation_layers': model.get_activation_info(),
    }
    
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Get full experiment specification
    experiment = get_experiment()  # Uses ACTIVE_EXPERIMENT from config
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    print(f"{'='*60}")
    
    # Extract components
    data_schema = experiment['data_schema']
    model_config = experiment['model_config']
    training_config = experiment['training_config']
    paths = experiment['paths']
    
    # Device selection
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    train_ds, val_ds = load_and_split_data(
        paths['data'],
        data_schema,
        val_size=training_config['val_size'],
        seed=training_config['seed']
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=training_config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=training_config['val_batch_size']
    )
    
    # 2. Build Model
    print(f"\nModel Configuration:")
    print(f"  Input dim:  {model_config['input_dim']} ({data_schema['features']})")
    print(f"  Output dim: {model_config['output_dim']} ({data_schema['targets']})")
    print(f"  Hidden:     {model_config['hidden_dims']}")
    print(f"  Activation: {model_config['activation']}")
    
    model = ClimateMLP(model_config).to(DEVICE)
    
    # 3. Train
    history = train_model(
        model, train_loader, val_loader,
        epochs=training_config['epochs'],
        lr=training_config['learning_rate'],
        device=DEVICE
    )
    
    # 4. Save Checkpoint
    print(f"\nSaving model to {paths['model']}...")
    save_checkpoint(model, train_ds, experiment, paths['model'], history)
    
    # 5. Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {paths['model']}")
    print(f"Physics meta available: {train_ds.get_physics_meta_names()}")
    print(f"Activation layers: {model.get_activation_layer_names()}")
    print(f"\nNext steps:")
    print(f"  - Run interpretability analysis: python interpretability.py")
    print(f"  - Or import: from interpretability import InterpretabilityAnalyzer")
