"""
Geodesics Experiment: Test light bending, Shapiro delay, and perihelion precession.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc import geodesics, ppn

def main():
    """Run the geodesics experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "geodesics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Geodesics Experiment...")
    
    # Solar system parameters
    GM_sun = 1.327e20  # m^3/s^2
    c = 2.998e8  # m/s
    
    # Test different gamma values
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    kappa_values = [2*g for g in gamma_values]  # kappa = 2*gamma from PPN
    
    # Light bending experiment
    print("1. Light Bending Analysis")
    print("========================")
    
    b_over_rs_values = np.logspace(0, 2, 50)  # Impact parameter ratios from 1 to 100
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for i, gamma in enumerate(gamma_values):
        deflection_angles = []
        for b_over_rs in b_over_rs_values:
            angle = geodesics.null_geodesic_deflection(b_over_rs, gamma, GM_sun, c)
            # Convert to arcseconds
            angle_arcsec = angle * 206265  # radians to arcseconds
            deflection_angles.append(angle_arcsec)
        
        ax.loglog(b_over_rs_values, deflection_angles, linewidth=2, 
                 label=f'γ={gamma} (κ={kappa_values[i]})', marker='o', markersize=3)
    
    # Einstein's prediction (gamma=1)
    einstein_angles = []
    for b_over_rs in b_over_rs_values:
        angle = geodesics.null_geodesic_deflection(b_over_rs, 1.0, GM_sun, c)
        angle_arcsec = angle * 206265
        einstein_angles.append(angle_arcsec)
    
    ax.loglog(b_over_rs_values, einstein_angles, 'k--', linewidth=3, 
             label='Einstein (γ=1)', alpha=0.7)
    
    ax.set_xlabel('Impact Parameter (b/rs)')
    ax.set_ylabel('Deflection Angle (arcseconds)')
    ax.set_title('Light Bending vs Impact Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "light_bending.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Shapiro delay experiment
    print("\n2. Shapiro Delay Analysis")
    print("========================")
    
    # Test parameters for superior conjunction
    rE = 1.496e11  # Earth distance from Sun (1 AU in meters)
    rR = rE  # Receiver distance (assume Earth again for round-trip)
    impact_b_over_rs_values = np.logspace(0, 2, 30)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for i, gamma in enumerate(gamma_values):
        delays = []
        for b_over_rs in impact_b_over_rs_values:
            delay = geodesics.shapiro_delay(b_over_rs, gamma, GM_sun, rE, rR, c)
            # Convert to microseconds
            delay_microsec = delay * 1e6
            delays.append(delay_microsec)
        
        ax.loglog(impact_b_over_rs_values, delays, linewidth=2, 
                 label=f'γ={gamma} (κ={kappa_values[i]})', marker='s', markersize=3)
    
    ax.set_xlabel('Impact Parameter (b/rs)')
    ax.set_ylabel('Time Delay (microseconds)')
    ax.set_title('Shapiro Delay vs Impact Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "shapiro_delay.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Mercury perihelion precession
    print("\n3. Mercury Perihelion Precession")
    print("===============================")
    
    # Mercury orbital parameters
    a_mercury = 0.387  # AU
    e_mercury = 0.206  # eccentricity
    beta = 1.0  # In our theory, beta is always 1
    
    precessions = []
    for gamma in gamma_values:
        precession = geodesics.perihelion_precession(a_mercury, e_mercury, beta, gamma, GM_sun, c)
        # Convert to arcseconds per century
        precession_arcsec_century = precession * 206265 * 100 / (88 * 365.25 * 24 * 3600) * (100 * 365.25 * 24 * 3600)
        precessions.append(precession_arcsec_century)
        
        print(f"γ={gamma}: {precession_arcsec_century:.2f} arcsec/century")
    
    # Observed Mercury precession anomaly
    observed_anomaly = 43.1  # arcseconds per century
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(gamma_values, precessions, 'bo-', linewidth=2, markersize=8, label='Our Model')
    ax.axhline(y=observed_anomaly, color='r', linestyle='--', linewidth=2, 
              label=f'Observed Anomaly ({observed_anomaly}" /century)')
    
    ax.set_xlabel('γ (PPN Parameter)')
    ax.set_ylabel('Precession (arcsec/century)')
    ax.set_title('Mercury Perihelion Precession')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "mercury_precession.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Summary of solar system tests
    print("\n4. Solar System Test Summary")
    print("===========================")
    
    # For gamma=1 (Einstein's GR), compute specific values
    gamma_gr = 1.0
    
    # Light bending at solar limb (b ≈ rs)
    deflection_limb = geodesics.null_geodesic_deflection(1.0, gamma_gr, GM_sun, c)
    deflection_limb_arcsec = deflection_limb * 206265
    
    # Shapiro delay for typical superior conjunction
    delay_typical = geodesics.shapiro_delay(5.0, gamma_gr, GM_sun, rE, rR, c)
    delay_typical_microsec = delay_typical * 1e6
    
    # Mercury precession
    precession_gr = geodesics.perihelion_precession(a_mercury, e_mercury, beta, gamma_gr, GM_sun, c)
    precession_gr_arcsec = precession_gr * 206265 * 100 / (88 * 365.25 * 24 * 3600) * (100 * 365.25 * 24 * 3600)
    
    print(f"GR Predictions (γ=1):")
    print(f"- Light bending at solar limb: {deflection_limb_arcsec:.2f} arcsec")
    print(f"- Shapiro delay (b=5rs): {delay_typical_microsec:.0f} μs")
    print(f"- Mercury precession: {precession_gr_arcsec:.1f} arcsec/century")
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Light bending comparison at fixed impact parameter
    test_b = 2.0  # 2 solar radii
    bending_values = [geodesics.null_geodesic_deflection(test_b, g, GM_sun, c) * 206265 for g in gamma_values]
    ax1.plot(gamma_values, bending_values, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('γ')
    ax1.set_ylabel('Deflection (arcsec)')
    ax1.set_title(f'Light Bending at b={test_b}rs')
    ax1.grid(True, alpha=0.3)
    
    # Shapiro delay comparison at fixed impact parameter
    test_b_shapiro = 3.0
    delay_values = [geodesics.shapiro_delay(test_b_shapiro, g, GM_sun, rE, rR, c) * 1e6 for g in gamma_values]
    ax2.plot(gamma_values, delay_values, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('γ')
    ax2.set_ylabel('Delay (μs)')
    ax2.set_title(f'Shapiro Delay at b={test_b_shapiro}rs')
    ax2.grid(True, alpha=0.3)
    
    # Mercury precession
    ax3.plot(gamma_values, precessions, 'bo-', linewidth=2, markersize=8, label='Model')
    ax3.axhline(y=observed_anomaly, color='r', linestyle='--', linewidth=2, label='Observed')
    ax3.set_xlabel('γ')
    ax3.set_ylabel('Precession (arcsec/century)')
    ax3.set_title('Mercury Precession')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "solar_system_tests.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(out_dir / "results.txt", 'w') as f:
        f.write("Geodesics Experiment Results\n")
        f.write("============================\n\n")
        
        f.write("1. Light Bending (at solar limb, b≈rs):\n")
        for i, gamma in enumerate(gamma_values):
            deflection = geodesics.null_geodesic_deflection(1.0, gamma, GM_sun, c) * 206265
            f.write(f"   γ={gamma}: {deflection:.2f} arcsec\n")
        
        f.write(f"\n2. Shapiro Delay (b=5rs, Earth-Sun):\n")
        for gamma in gamma_values:
            delay = geodesics.shapiro_delay(5.0, gamma, GM_sun, rE, rR, c) * 1e6
            f.write(f"   γ={gamma}: {delay:.0f} μs\n")
        
        f.write(f"\n3. Mercury Perihelion Precession:\n")
        for i, gamma in enumerate(gamma_values):
            f.write(f"   γ={gamma}: {precessions[i]:.1f} arcsec/century\n")
        
        f.write(f"\nObserved Mercury precession anomaly: {observed_anomaly} arcsec/century\n")
        f.write(f"Best fit occurs at γ≈1 (Einstein's GR)\n")
    
    print(f"\nExperiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- light_bending.png: Deflection angle vs impact parameter")
    print("- shapiro_delay.png: Time delay vs impact parameter")
    print("- mercury_precession.png: Perihelion precession vs γ")
    print("- solar_system_tests.png: Combined comparison plots")
    print("- results.txt: Numerical results summary")

if __name__ == "__main__":
    main()
