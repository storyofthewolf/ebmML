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

    def run_to_equilibrium(self, T_init, log_pCO2, S0, 
                           dt=1.0, max_steps=50000, tol=1e-4,
                           patience=50):
        """
        Integrate the EBM forward from T_init until equilibrium.
        
        Convergence: |dT/dt| < tol for `patience` consecutive steps.
        
        Parameters
        ----------
        T_init : float or array
            Initial temperature(s) (K)
        log_pCO2 : float or array
            log10(CO2 in ppm)
        S0 : float or array
            Solar constant (W/m2)
        dt : float
            Timestep
        max_steps : int
            Maximum integration steps before declaring non-convergence
        tol : float
            Convergence threshold for |dT/dt| (K/step)
        patience : int
            Number of consecutive steps below tol to declare convergence
        
        Returns
        -------
        dict with equilibrium state and diagnostics
        """
        scalar_input = np.isscalar(T_init)
        T = np.atleast_1d(np.float64(T_init)).copy()
        log_pCO2 = np.atleast_1d(np.float64(log_pCO2))
        S0 = np.atleast_1d(np.float64(S0))
        
        n = len(T)
        converged = np.zeros(n, dtype=bool)
        steps_to_converge = np.full(n, max_steps, dtype=int)
        consecutive_stable = np.zeros(n, dtype=int)
        
        for step_i in range(max_steps):
            T_new, N, alpha, epsilon, ASR, OLR = self.step(T, log_pCO2, S0, dt)
            
            dT = np.abs(T_new - T)
            
            # Track consecutive stability for unconverged samples
            newly_stable = (~converged) & (dT < tol)
            newly_unstable = (~converged) & (dT >= tol)
            consecutive_stable[newly_stable] += 1
            consecutive_stable[newly_unstable] = 0
            
            # Check patience criterion
            just_converged = (~converged) & (consecutive_stable >= patience)
            converged[just_converged] = True
            steps_to_converge[just_converged] = step_i
            
            T = T_new
            
            # Early exit if all converged
            if converged.all():
                break
        
        # Final diagnostics at equilibrium
        _, N_eq, alpha_eq, eps_eq, ASR_eq, OLR_eq = self.step(T, log_pCO2, S0, dt)
        
        result = {
            'T_eq': T[0] if scalar_input else T,
            'N_toa_eq': N_eq[0] if scalar_input else N_eq,
            'Albedo_eq': alpha_eq[0] if scalar_input else alpha_eq,
            'Emissivity_eq': eps_eq[0] if scalar_input else eps_eq,
            'ASR_eq': ASR_eq[0] if scalar_input else ASR_eq,
            'OLR_eq': OLR_eq[0] if scalar_input else OLR_eq,
            'converged': converged[0] if scalar_input else converged,
            'steps': steps_to_converge[0] if scalar_input else steps_to_converge,
        }
        return result


# =============================================================================
# BIFURCATION DIAGRAM: Verify Bistability Exists
# =============================================================================

def plot_bifurcation_diagram(S0_range=None, log_co2_values=None):
    """
    Sweep S0 from warm and cold initial conditions to map the hysteresis loop.
    This is the classic Budyko bifurcation diagram.
    """
    model = ToyClimateModel()
    
    if S0_range is None:
        S0_range = np.linspace(1000, 1600, 300)
    if log_co2_values is None:
        log_co2_values = [2.5, 3.0, 3.5]  # ~316, 1000, 3162 ppm
    
    fig, axes = plt.subplots(1, len(log_co2_values), figsize=(5*len(log_co2_values), 5),
                             sharey=True)
    if len(log_co2_values) == 1:
        axes = [axes]
    
    for ax, lco2 in zip(axes, log_co2_values):
        T_warm = np.zeros_like(S0_range)
        T_cold = np.zeros_like(S0_range)
        conv_warm = np.zeros_like(S0_range, dtype=bool)
        conv_cold = np.zeros_like(S0_range, dtype=bool)
        
        for i, s0 in enumerate(S0_range):
            # Warm start
            res_w = model.run_to_equilibrium(350.0, lco2, s0)
            T_warm[i] = res_w['T_eq']
            conv_warm[i] = res_w['converged']
            
            # Cold start
            res_c = model.run_to_equilibrium(220.0, lco2, s0)
            T_cold[i] = res_c['T_eq']
            conv_cold[i] = res_c['converged']
        
        # Plot
        ax.plot(S0_range[conv_warm], T_warm[conv_warm], 'r-', lw=2, label='Warm start (350K)')
        ax.plot(S0_range[conv_cold], T_cold[conv_cold], 'b-', lw=2, label='Cold start (220K)')
        
        # Shade hysteresis region
        bistable = conv_warm & conv_cold & (np.abs(T_warm - T_cold) > 5.0)
        if bistable.any():
            s_bi = S0_range[bistable]
            ax.axvspan(s_bi.min(), s_bi.max(), alpha=0.1, color='purple', 
                      label='Bistable region')
        
        ax.set_xlabel('Solar Constant S₀ (W/m²)')
        ax.set_ylabel('Equilibrium Temperature (K)')
        ax.set_title(f'log(pCO₂) = {lco2:.1f}\n({10**lco2:.0f} ppm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(200, 380)
    
    fig.suptitle('Bifurcation Diagram: Does Hysteresis Exist?', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# GENERATE EQUILIBRIUM TRAINING DATA
# =============================================================================

def generate_equilibrium_data(n_samples=500_000, seed=42):
    """
    Generate equilibrium climate states from random (S0, CO2, T_initial).
    
    The randomized T_initial naturally samples both basins of attraction
    in the bistable regime, so we capture the full hysteresis structure.
    """
    print(f"Generating {n_samples:,} equilibrium climate states...")
    np.random.seed(seed)
    model = ToyClimateModel()
    
    # 1. Randomly sample the forcing/initial condition space
    S0_random = np.random.uniform(1100, 1600, n_samples)
    log_co2_random = np.random.uniform(2, 4, n_samples)
    T_init_random = np.random.uniform(220, 360, n_samples)
    
    # 2. Run each sample to equilibrium
    # We process in batches for memory efficiency
    batch_size = 10000
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Pre-allocate output arrays
    T_eq = np.zeros(n_samples)
    N_eq = np.zeros(n_samples)
    Albedo_eq = np.zeros(n_samples)
    Emissivity_eq = np.zeros(n_samples)
    ASR_eq = np.zeros(n_samples)
    OLR_eq = np.zeros(n_samples)
    converged = np.zeros(n_samples, dtype=bool)
    steps = np.zeros(n_samples, dtype=int)
    
    for batch_i in range(n_batches):
        i0 = batch_i * batch_size
        i1 = min(i0 + batch_size, n_samples)
        
        if batch_i % 10 == 0:
            pct = 100 * i0 / n_samples
            print(f"  Batch {batch_i+1}/{n_batches} ({pct:.0f}%)")
        
        res = model.run_to_equilibrium(
            T_init_random[i0:i1],
            log_co2_random[i0:i1],
            S0_random[i0:i1]
        )
        
        T_eq[i0:i1] = res['T_eq']
        N_eq[i0:i1] = res['N_toa_eq']
        Albedo_eq[i0:i1] = res['Albedo_eq']
        Emissivity_eq[i0:i1] = res['Emissivity_eq']
        ASR_eq[i0:i1] = res['ASR_eq']
        OLR_eq[i0:i1] = res['OLR_eq']
        converged[i0:i1] = res['converged']
        steps[i0:i1] = res['steps']
    
    # 3. Pack into DataFrame
    df = pd.DataFrame({
        # Inputs (what the NN sees)
        'S0': S0_random,
        'log_pCO2': log_co2_random,
        'T_initial': T_init_random,
        # Primary target
        'T_eq': T_eq,
        # Diagnostics (for interpretability, not necessarily NN targets)
        'N_toa_eq': N_eq,
        'Albedo_eq': Albedo_eq,
        'Emissivity_eq': Emissivity_eq,
        'ASR_eq': ASR_eq,
        'OLR_eq': OLR_eq,
        # Convergence metadata
        'converged': converged,
        'steps_to_converge': steps,
    })
    
    n_conv = converged.sum()
    n_fail = (~converged).sum()
    print(f"\nConvergence: {n_conv:,} converged, {n_fail:,} did not converge")
    print(f"  ({100*n_fail/n_samples:.1f}% non-convergent)")
    
    # Flag bistable samples (where T_eq depends strongly on T_initial)
    # We'll detect this in post-processing
    
    return df


# =============================================================================
# VISUALIZE EQUILIBRIUM DATA
# =============================================================================

def plot_equilibrium_overview(df):
    """Visualize the equilibrium dataset and look for bistability signatures."""
    
    df_conv = df[df['converged']].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # --- Panel 1: T_eq vs S0, colored by T_initial ---
    ax = axes[0, 0]
    sc = ax.scatter(df_conv['S0'][::10], df_conv['T_eq'][::10],
                    c=df_conv['T_initial'][::10], cmap='coolwarm', 
                    s=1, alpha=0.3, vmin=220, vmax=360)
    plt.colorbar(sc, ax=ax, label='T_initial (K)')
    ax.set_xlabel('Solar Constant S₀ (W/m²)')
    ax.set_ylabel('Equilibrium Temperature (K)')
    ax.set_title('Equilibrium States: Colored by Initial Condition')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: T_eq vs T_initial (basin structure) ---
    ax = axes[0, 1]
    # Pick a narrow S0 band to see basin structure clearly
    s0_center = 1361.0
    s0_band = 30.0
    mask = (df_conv['S0'] > s0_center - s0_band) & (df_conv['S0'] < s0_center + s0_band)
    df_band = df_conv[mask]
    sc = ax.scatter(df_band['T_initial'][::5], df_band['T_eq'][::5],
                    c=df_band['log_pCO2'][::5], cmap='viridis',
                    s=2, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='log(pCO₂)')
    ax.set_xlabel('Initial Temperature (K)')
    ax.set_ylabel('Equilibrium Temperature (K)')
    ax.set_title(f'Basin Structure (S₀ ≈ {s0_center:.0f} ± {s0_band:.0f} W/m²)')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Equilibrium Albedo vs T_eq ---
    ax = axes[1, 0]
    ax.scatter(df_conv['T_eq'][::10], df_conv['Albedo_eq'][::10],
               c='steelblue', s=1, alpha=0.2)
    ax.set_xlabel('Equilibrium Temperature (K)')
    ax.set_ylabel('Equilibrium Albedo')
    ax.set_title('Albedo at Equilibrium')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: Steps to converge (critical slowing down?) ---
    ax = axes[1, 1]
    conv_mask = df_conv['steps_to_converge'] < 50000
    sc = ax.scatter(df_conv['T_eq'][conv_mask][::10], 
                    df_conv['steps_to_converge'][conv_mask][::10],
                    c=df_conv['S0'][conv_mask][::10], cmap='plasma',
                    s=1, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='S₀ (W/m²)')
    ax.set_xlabel('Equilibrium Temperature (K)')
    ax.set_ylabel('Steps to Converge')
    ax.set_title('Convergence Time (Critical Slowing Down?)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # ---- Step 1: Check for bistability ----
    print("=" * 60)
    print("STEP 1: Bifurcation Diagram - Checking for Hysteresis")
    print("=" * 60)
    fig_bif = plot_bifurcation_diagram()
    fig_bif.savefig('bifurcation_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: bifurcation_diagram.png")
    
    # ---- Step 2: Generate equilibrium dataset ----
    print("\n" + "=" * 60)
    print("STEP 2: Generating Equilibrium Training Data")
    print("=" * 60)
    df = generate_equilibrium_data(n_samples=500_000)
    
    # ---- Step 3: Save ----
    save_path = "toy_climate_equilibrium_500k.csv"
    df.to_csv(save_path, index=False)
    print(f"\nSaved dataset to {save_path}")
    
    # ---- Step 4: Visualize ----
    print("\n" + "=" * 60)
    print("STEP 3: Visualizing Equilibrium Data")
    print("=" * 60)
    fig_eq = plot_equilibrium_overview(df)
    fig_eq.savefig('equilibrium_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: equilibrium_overview.png")
    
    # ---- Summary Statistics ----
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {len(df):,}")
    print(f"Converged:         {df['converged'].sum():,}")
    print(f"Non-converged:     {(~df['converged']).sum():,}")
    print(f"T_eq range:        {df['T_eq'].min():.1f} - {df['T_eq'].max():.1f} K")
    print(f"T_eq mean:         {df['T_eq'].mean():.1f} K")
    print(f"T_eq std:          {df['T_eq'].std():.1f} K")
    
    # Check for bimodality (signature of bistability)
    T_eq_conv = df[df['converged']]['T_eq']
    hist, bin_edges = np.histogram(T_eq_conv, bins=100)
    print(f"\nT_eq distribution peaks:")
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Find local maxima
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=len(T_eq_conv)*0.005, distance=10)
    for p in peaks:
        print(f"  Peak at T_eq ≈ {bin_centers[p]:.1f} K (count: {hist[p]:,})")
    
    if len(peaks) >= 2:
        print("\n*** BISTABILITY DETECTED: Multiple equilibrium peaks ***")
    else:
        print("\n*** WARNING: No clear bistability. Physics tuning may be needed. ***")
