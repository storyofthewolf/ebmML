import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Import the model structure from your other script
from climate_nn import load_model_from_checkpoint, load_and_split_data

def get_random_direction(model):
    """
    Generate a random direction vector for every weight in the model.
    Normalized by the magnitude of the weights (Filter Normalization).
    """
    direction = []
    for p in model.parameters():
        # Create random Gaussian noise
        d = torch.randn_like(p)
        
        # Normalize
        if d.dim() > 1: # Weights
            d = d / d.norm() * p.norm()
        else: # Bias
            d = d / d.norm() * p.norm()
            
        direction.append(d)
    return direction

def perturb_model(model, direction1, direction2, alpha, beta):
    """
    Shift model weights: W_new = W_old + (alpha * dir1) + (beta * dir2)
    """
    new_model = copy.deepcopy(model)
    params = list(new_model.parameters())
    
    for i, p in enumerate(params):
        p.data.add_(direction1[i] * alpha)
        p.data.add_(direction2[i] * beta)
        
    return new_model

def compute_loss_surface(base_model, loader, range_val=1.0, steps=20):
    """
    Map the loss landscape over a 2D grid.
    """
    print(f"Mapping landscape: grid size {steps}x{steps}...")
    
    # 1. Generate two random axes
    dir1 = get_random_direction(base_model)
    dir2 = get_random_direction(base_model)
    
    # 2. Define the grid
    alphas = np.linspace(-range_val, range_val, steps)
    betas = np.linspace(-range_val, range_val, steps)
    
    loss_surface = np.zeros((steps, steps))
    criterion = torch.nn.MSELoss()
    
    # 3. Walk the grid
    base_model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Create perturbed model
                perturbed_model = perturb_model(base_model, dir1, dir2, alpha, beta)
                
                # Calculate Validation Loss
                total_loss = 0.0
                for X, Y in loader:
                    pred = perturbed_model(X)
                    loss = criterion(pred, Y)
                    total_loss += loss.item()
                
                loss_surface[i, j] = total_loss / len(loader)
                
            print(f"  Row {i+1}/{steps} done...", end='\r')
            
    return alphas, betas, loss_surface

def plot_landscape_3d(alphas, betas, loss_surface, savepath):
    """Create a 3D Surface Plot."""
    X, Y = np.meshgrid(alphas, betas)
    Z = np.log(loss_surface) # Log scale helps visualize valleys better
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                          edgecolor='none', alpha=0.9, antialiased=True)
    
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Log Loss')
    ax.set_title('The Geometry of Understanding: Loss Landscape')
    
    # Mark the center (current model)
    ax.scatter([0], [0], [Z.min()], color='red', s=100, label='Current Model')
    ax.legend()
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"\nSaved landscape to {savepath}")

def plot_landscape_contour(alphas, betas, loss_surface, savepath):
    """Create a 2D Contour Map (Topographic)."""
    X, Y = np.meshgrid(alphas, betas)
    Z = np.log(loss_surface)
    
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Log Loss')
    
    # Add contours lines
    plt.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Mark center
    plt.plot(0, 0, 'r*', markersize=15, label='Your Model')
    
    plt.xlabel('Random Direction 1')
    plt.ylabel('Random Direction 2')
    plt.title('Topographic Map of the Solution Basin')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
    print(f"Saved contour map to {savepath}")

if __name__ == "__main__":
    # Setup
    CSV_PATH = "../training_sets/ebm_0d_model_v1_climate_data_1M.csv"
    checkpoint_path = '../networks/climate_model.pt'
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run climate_nn.py first.")

    _, val_ds = load_and_split_data(CSV_PATH)
    # Use a subset of validation data for faster plotting
    val_loader = DataLoader(val_ds, batch_size=5000, shuffle=False)
    
    # 2. Load Your Model
    print("Loading model...")
     try:
        model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    
    # 3. Compute Landscape
    # range_val=1.0 means we look at weights +/- 100% of their current magnitude
    alphas, betas, Z = compute_loss_surface(model, val_loader, range_val=1.0, steps=25)
    
    # 4. Visualize
    plot_landscape_3d(alphas, betas, Z, '../figures/landscape_3d.png')
    plot_landscape_contour(alphas, betas, Z, '../figures/landscape_contour.png')
