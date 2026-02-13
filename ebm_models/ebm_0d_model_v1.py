import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x, center, steepness):
    """Smooth transition function (0 to 1)."""
    return 1 / (1 + np.exp(-steepness * (x - center)))

class ToyClimateModel:
    def __init__(self):
        # Constants
        self.sigma = 5.67e-8  # Stefan-Boltzmann
        self.C_heat = 20.0    # Heat capacity (arbitrary units for inertia)
        
        # --- TUNABLE PHYSICS PARAMETERS ---
        # 1. Albedo Parameters
        self.alpha_land = 0.28       # Base planetary albedo
        self.alpha_ice_max = 0.40    # Extra albedo from ice
        self.alpha_cloud_max = 0.15  # Extra albedo from low clouds
        
        # Transitions
        self.T_ice_melt = 280.0      # Temp where ice melts (K)
        self.T_cloud_collapse = 310.0 # Temp where cloud deck breaks up (K)
        
        # 2. Greenhouse Parameters
        self.epsilon_base = 0.70     # Base emissivity (transparent atmosphere)
        self.co2_sensitivity = 0.05  # How much CO2 drops emissivity
        self.vapor_feedback = 0.002  # Water vapor feedback strength (per K)
        self.cloud_gh_max = 0.05     # Greenhouse effect of high clouds
        
    def calculate_albedo(self, T):
        """
        Total Albedo = Land + Ice(T) + Cloud(T)
        - Ice: Present below 280K, melts smoothly.
        - Clouds: Present below 310K, collapse abruptly (positive feedback).
        """
        # Ice term: 1.0 when cold, 0.0 when hot
        ice_fraction = 1.0 - sigmoid(T, self.T_ice_melt, steepness=0.5)
        alpha_ice = self.alpha_ice_max * ice_fraction
        
        # Cloud term: 1.0 when cool, 0.0 when very hot (instability)
        cloud_fraction = 1.0 - sigmoid(T, self.T_cloud_collapse, steepness=1.0)
        alpha_cloud = self.alpha_cloud_max * cloud_fraction
        
        return self.alpha_land + alpha_ice + alpha_cloud

    def calculate_emissivity(self, T, log_co2):
        """
        Emissivity = Base - CO2 - WaterVapor(T) - Cloud(T)
        (Lower emissivity = Stronger Greenhouse)
        """
        # CO2 forcing (logarithmic is handled by input)
        gh_co2 = self.co2_sensitivity * log_co2
        
        # Water Vapor: Stronger greenhouse (lower epsilon) as T rises
        # Clamped to avoid unphysical negative emissivity
        gh_vapor = self.vapor_feedback * (T - 280)
        gh_vapor = np.clip(gh_vapor, 0, 0.3) 
        
        # Cloud Greenhouse: High clouds trap heat, assumed constant for now
        gh_cloud = self.cloud_gh_max
        
        # Total Emissivity
        epsilon = self.epsilon_base - gh_co2 - gh_vapor - gh_cloud
        return np.clip(epsilon, 0.05, 1.0)

    def step(self, T, log_pCO2, S0, dt=1.0):
        """Calculate next state based on energy balance."""
        
        # 1. Calculate Radiative Terms
        alpha = self.calculate_albedo(T)
        epsilon = self.calculate_emissivity(T, log_pCO2)
        
        ASR = (S0 / 4) * (1 - alpha)
        OLR = epsilon * self.sigma * (T**4)
        
        # 2. Energy Imbalance
        N = ASR - OLR
        
        # 3. Time Evolution
        dT_dt = N / self.C_heat
        T_next = T + dT_dt * dt
        
        return T_next, N, alpha, epsilon, ASR, OLR

# =============================================================================
# GENERATE 1 MILLION DATA POINTS
# =============================================================================

def generate_training_data(n_samples=1_000_000):
    print(f"Generating {n_samples:,} samples of synthetic climate physics...")
    model = ToyClimateModel()
    
    # 1. Randomly sample the "Phase Space"
    # We want the NN to learn physics everywhere, not just on one trajectory.
    
    # Temperature: 200K (Snowball) to 350K (Runaway Greenhouse)
    T_random = np.random.uniform(220, 360, n_samples)
    
    # CO2: 100ppm to 10,000ppm (Log Uniform)
    # log10(100)=2, log10(10000)=4
    log_co2_random = np.random.uniform(2, 4, n_samples)
    
    # Solar: 0.9 to 1.1 S0 (1361 W/m2)
    S0_random = np.random.uniform(1200, 1500, n_samples)
    
    # 2. Run the Physics Engine (Vectorized)
    T_next, N, alpha, eps, ASR, OLR = model.step(T_random, log_co2_random, S0_random)
    
    # 3. Pack into DataFrame
    df = pd.DataFrame({
        'Ts': T_random,
        'log_pCO2': log_co2_random,
        'S0': S0_random,
        'Ts_next': T_next,   # TARGET VARIABLE
        'N_toa': N,          # DIAGNOSTIC
        'Albedo': alpha,     # HIDDEN VARIABLE (Can the NN find this?)
        'Emissivity': eps,   # HIDDEN VARIABLE (Can the NN find this?)
        'OLR': OLR,
        'ASR': ASR
    })
    
    return df

# =============================================================================
# VISUALIZE THE "GROUND TRUTH" PHYSICS
# =============================================================================

def plot_physics_manifold(df):
    """Visualize the non-linearities the NN must learn."""
    
    # Sort for clean plotting
    df_plot = df.sort_values('Ts')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: The Albedo "Cliffs"
    # We expect two drops: Ice melt (280K) and Cloud collapse (310K)
    ax1.scatter(df_plot['Ts'][::100], df_plot['Albedo'][::100], 
                c='blue', s=1, alpha=0.1)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Planetary Albedo')
    ax1.set_title('Ground Truth: Albedo Feedbacks')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The Greenhouse "Runaway"
    # Emissivity should drop as T rises (Water Vapor Feedback)
    sc = ax2.scatter(df_plot['Ts'][::100], df_plot['Emissivity'][::100], 
                     c=df_plot['log_pCO2'][::100], cmap='viridis', s=1, alpha=0.1)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Effective Emissivity')
    ax2.set_title('Ground Truth: Greenhouse Feedbacks')
    plt.colorbar(sc, ax=ax2, label='log(pCO2)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Generate Data
    df = generate_training_data(1_000_000)
    
    # Save for your PyTorch script
    # This replaces your .dat files!
    save_path = "toy_climate_data_1M.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved dataset to {save_path}")
    
    # Show what we built
    plot_physics_manifold(df)
