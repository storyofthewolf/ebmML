"""
Toy Climate NN - Master Script
==============================
1. Loads toy_climate_data_1M.csv
2. Trains the Neural Network
3. Generates Interpretability Plots (Neuron Correlations)
4. Saves model for Loss Landscape analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. DATA LOADING
# =============================================================================

class ToyClimateDataset(Dataset):
    """Dataset for the 1M point synthetic physics data."""
    
    def __init__(self, dataframe: pd.DataFrame, 
                 scaler_X=None, scaler_Y=None,
                 is_validation=False):
        
        self.df = dataframe.reset_index(drop=True)
        
        # INPUTS: Current State + Forcing
        self.feature_names = ['Ts', 'log_pCO2', 'S0']
        X_raw = self.df[self.feature_names].values.astype(np.float32)
        
        # OUTPUTS: Next State + Observables
        self.target_names = ['Ts_next', 'OLR', 'ASR']
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
        
        # Keep raw physics variables for Interpretability Analysis
        self.physics_meta = {
            'Albedo': self.df['Albedo'].values,
            'Emissivity': self.df['Emissivity'].values,
            'N_toa': self.df['N_toa'].values,
            'Ts': self.df['Ts'].values
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_and_split_data(csv_path, val_size=0.2, seed=42):
    """Load CSV and create Train/Val datasets."""
    print(f"Loading {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Did you run the generation script?")
        
    df = pd.read_csv(csv_path)
    
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
    
    train_ds = ToyClimateDataset(train_df)
    val_ds = ToyClimateDataset(val_df, 
                               scaler_X=train_ds.scaler_X, 
                               scaler_Y=train_ds.scaler_Y,
                               is_validation=True)
    
    return train_ds, val_ds

# =============================================================================
# 2. NEURAL NETWORK
# =============================================================================

class ClimateMLP(nn.Module):
    """Simple MLP with hooks for mechanistic interpretability."""
    
    def __init__(self, input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.activations = {}
        self.register_hooks()
        
    def register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation(f'relu_{i//2}'))

    def forward(self, x):
        return self.network(x)
    
    def get_hidden_activations(self, x):
        self.activations = {}
        _ = self.forward(x)
        return self.activations

# =============================================================================
# 3. INTERPRETABILITY ANALYZER (RESTORED)
# =============================================================================

class InterpretabilityAnalyzer:
    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()
        
    def get_all_activations(self, layer_name='relu_0', sample_size=10000):
        """Get activations for a random subset of data."""
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        X_subset = self.dataset.X[indices].to(self.device)
        
        with torch.no_grad():
            acts = self.model.get_hidden_activations(X_subset)[layer_name]
        
        return acts.cpu().numpy(), indices

    def correlate_neurons_with_physics(self):
        """Correlate neurons with HIDDEN physics variables (Albedo/Emissivity)."""
        print("\nRunning Neuron Spectroscopy...")
        
        sample_size = 20000
        acts, indices = self.get_all_activations('relu_0', sample_size)
        
        meta = self.dataset.physics_meta
        Ts = meta['Ts'][indices]
        Albedo = meta['Albedo'][indices]     # GROUND TRUTH
        Emissivity = meta['Emissivity'][indices] # GROUND TRUTH
        N_toa = meta['N_toa'][indices]
        
        n_neurons = acts.shape[1]
        results = []
        
        for i in range(n_neurons):
            a = acts[:, i]
            if np.std(a) < 1e-6: continue 
            
            results.append({
                'neuron': i,
                'corr_Ts': np.corrcoef(a, Ts)[0,1],
                'corr_Albedo': np.corrcoef(a, Albedo)[0,1],
                'corr_Emissivity': np.corrcoef(a, Emissivity)[0,1],
                'corr_N_toa': np.corrcoef(a, N_toa)[0,1]
            })
            
        return pd.DataFrame(results)

def plot_correlations(df_results, savepath):
    """Heatmap of neuron specializations."""
    plt.figure(figsize=(8, 10))
    
    df_sorted = df_results.sort_values('corr_Albedo', ascending=False)
    data = df_sorted[['corr_Ts', 'corr_Albedo', 'corr_Emissivity', 'corr_N_toa']].values
    
    plt.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(4), ['Ts', 'Albedo', 'Emissivity', 'N_toa'])
    plt.yticks(range(len(df_sorted)), df_sorted['neuron'].values)
    plt.title('Neuron Specialization (Sorted by Albedo)')
    
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
    print(f"Saved correlation plot to {savepath}")

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
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
        
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train={avg_train:.6f}, Val={avg_val:.6f}")
            
    return history

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Config
    CSV_PATH = "toy_climate_data_1M.csv" 
    OUT_DIR = "toy_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    train_ds, val_ds = load_and_split_data(CSV_PATH)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4096)
    
    # 2. Build Model
    model = ClimateMLP(input_dim=3, hidden_dims=[8, 8, 8, 8], output_dim=3).to(DEVICE)
    
    # 3. Train
    history = train_model(model, train_loader, val_loader, epochs=20, device=DEVICE)
    
    # 4. Run Interpretability (Restored!)
    analyzer = InterpretabilityAnalyzer(model, train_ds, device=DEVICE)
    df_corr = analyzer.correlate_neurons_with_physics()
    
    print("\nTop 'Albedo' Neurons:")
    print(df_corr.sort_values('corr_Albedo', ascending=False).head(3))
    
    plot_correlations(df_corr, f"{OUT_DIR}/neuron_correlations.png")
    
    # 5. Save Model (for Landscape Script)
    print(f"\nSaving model to {OUT_DIR}/climate_model.pt ...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': train_ds.scaler_X,
        'scaler_Y': train_ds.scaler_Y,
        'config': {
            'input_dim': 3,
            'hidden_dims': [8,8,8,8],
            'output_dim': 3
        }
    }, f'{OUT_DIR}/climate_model.pt')
    
    print("\nDONE! You can now run 'toy_loss_landscape.py'.")
