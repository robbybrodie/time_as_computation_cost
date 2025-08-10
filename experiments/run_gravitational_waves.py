"""
Gravitational Wave Propagation: Linearized perturbations in TACC metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc import gravitational_waves, constants

def main():
    """Run the gravitational wave propagation experiment."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    out_dir = Path(__file__).resolve().parent / "out" / "gravitational_waves"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Gravitational Wave Propagation Experiment...")
    
    # Solar mass for reference
    M_sun = 1.989e30  # kg
    
    # GW150914-like parameters
    M1 = 36 * M_sun  # kg
    M2 = 29 * M_sun  # kg
    distance = 410e6 * 9.461e15  # meters (410 Mpc)
    
    # Generate waveforms for different κ values
    print("Generating TACC gravitational waveforms...")
    kappa_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    duration = 1.0  # seconds
    
    comparison_data = gravitational_waves.compare_waveforms_tacc_gr(
        M1, M2, distance, kappa_values, duration
    )
    
    # Plot waveform comparisons
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(kappa_values)))
    t = comparison_data['time']
    
    # Plus polarization
    for i, kappa in enumerate(kappa_values):
        h_plus = comparison_data['waveforms'][kappa]['h_plus']
        ax1.plot(t * 1000, h_plus, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('h_plus')
    ax1.set_title('Plus Polarization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-100, 100])
    
    # Cross polarization
    for i, kappa in enumerate(kappa_values):
        h_cross = comparison_data['waveforms'][kappa]['h_cross']
        ax2.plot(t * 1000, h_cross, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('h_cross')
    ax2.set_title('Cross Polarization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-100, 100])
    
    # Detector response
    for i, kappa in enumerate(kappa_values):
        h_detector = comparison_data['waveforms'][kappa]['h_detector']
        ax3.plot(t * 1000, h_detector, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('h_detector')
    ax3.set_title('LIGO Detector Response')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-100, 100])
    
    # Strain amplitude vs frequency
    frequencies = np.logspace(1, 3, 100)  # 10 Hz to 1000 Hz
    
    for i, kappa in enumerate(kappa_values):
        amplitudes = [gravitational_waves.strain_amplitude_tacc(M1, M2, distance, f, kappa) 
                     for f in frequencies]
        ax4.loglog(frequencies, amplitudes, color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Strain Amplitude')
    ax4.set_title('Strain Amplitude vs Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([10, 1000])
    
    plt.tight_layout()
    plt.savefig(out_dir / "waveform_comparison.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Dispersion relation analysis
    print("Analyzing dispersion relations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gravitational wave speed vs κ
    kappa_fine = np.linspace(0.1, 5.0, 200)
    gw_speeds = []
    
    for kappa in kappa_fine:
        v_gw = gravitational_waves.dispersion_relation_tacc(100.0, kappa, N_background=1.0)
        gw_speeds.append(v_gw / constants.c)  # As fraction of c
    
    ax1.plot(kappa_fine, gw_speeds, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='c (GR)')
    ax1.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='κ=2 (GR)')
    ax1.set_xlabel('κ (Constitutive Parameter)')
    ax1.set_ylabel('v_gw / c')
    ax1.set_title('Gravitational Wave Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.5])
    
    # Time delay for multi-messenger astronomy
    distance_mm = 130e6 * 9.461e15  # meters (130 Mpc, GW170817-like)
    time_delays = []
    
    for kappa in kappa_fine:
        delta_t = gravitational_waves.time_delay_tacc(100.0, distance_mm, kappa, N_path=1.0)
        time_delays.append(delta_t)
    
    ax2.plot(kappa_fine, time_delays, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='No delay (GR)')
    ax2.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='κ=2 (GR)')
    ax2.set_xlabel('κ (Constitutive Parameter)')
    ax2.set_ylabel('Time Delay (s)')
    ax2.set_title(f'Multi-messenger Time Delay (d = 130 Mpc)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "dispersion_analysis.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Phase evolution analysis
    print("Computing phase evolution...")
    t_phase = np.linspace(-0.2, 0.05, 1000)  # Time around merger
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Phase evolution for different κ
    for i, kappa in enumerate([1.0, 1.5, 2.0, 2.5, 3.0]):
        phase, frequency = gravitational_waves.phase_evolution_tacc(M1, M2, t_phase, kappa)
        
        # Skip infinite frequencies near merger
        valid_idx = np.isfinite(frequency) & (frequency < 1000)
        ax1.plot(t_phase[valid_idx] * 1000, phase[valid_idx], 
                color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Phase (radians)')
    ax1.set_title('Gravitational Wave Phase Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-200, 50])
    
    # Frequency evolution
    for i, kappa in enumerate([1.0, 1.5, 2.0, 2.5, 3.0]):
        phase, frequency = gravitational_waves.phase_evolution_tacc(M1, M2, t_phase, kappa)
        
        valid_idx = np.isfinite(frequency) & (frequency < 1000)
        ax2.plot(t_phase[valid_idx] * 1000, frequency[valid_idx], 
                color=colors[i], linewidth=2, label=f'κ={kappa}')
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Instantaneous Frequency Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-200, 50])
    ax2.set_ylim([10, 500])
    
    # Frequency domain representation
    for i, kappa in enumerate([1.0, 2.0, 3.0]):
        phase, frequency = gravitational_waves.phase_evolution_tacc(M1, M2, t_phase, kappa)
        h_detector = comparison_data['waveforms'][kappa]['h_detector']
        
        # Simple FFT for frequency domain
        dt = t_phase[1] - t_phase[0]
        fft_freq = np.fft.fftfreq(len(h_detector), dt)
        fft_h = np.abs(np.fft.fft(h_detector))
        
        # Plot only positive frequencies
        pos_freq = fft_freq > 0
        ax3.loglog(fft_freq[pos_freq], fft_h[pos_freq], 
                  color=colors[i*2], linewidth=2, label=f'κ={kappa}')
    
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('|h̃(f)|')
    ax3.set_title('Frequency Domain Waveform')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([10, 500])
    
    # SNR comparison
    snr_values = []
    for kappa in kappa_values:
        # Simple SNR estimate based on amplitude
        h_detector = comparison_data['waveforms'][kappa]['h_detector']
        signal_power = np.mean(h_detector**2)
        # Assume constant noise level
        noise_power = 1e-46  # Approximate LIGO noise level
        snr = np.sqrt(signal_power / noise_power)
        snr_values.append(snr)
    
    ax4.plot(kappa_values, snr_values, 'bo-', linewidth=2, markersize=8)
    ax4.axvline(x=2.0, color='r', linestyle=':', alpha=0.7, label='GR (κ=2)')
    ax4.set_xlabel('κ (Constitutive Parameter)')
    ax4.set_ylabel('Signal-to-Noise Ratio')
    ax4.set_title('SNR vs κ Parameter')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "phase_analysis.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Parameter estimation simulation
    print("Simulating parameter estimation...")
    
    # Fit TACC parameters to synthetic data
    fit_result = gravitational_waves.fit_gw150914_data(kappa_guess=2.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Best-fit waveform comparison
    t_fit = fit_result['time']
    h_best_fit = fit_result['h_detector']
    
    # Compare with GR
    h_plus_gr, h_cross_gr = gravitational_waves.waveform_tacc(
        fit_result['M1'], fit_result['M2'], fit_result['distance'], t_fit, 2.0
    )
    h_gr = gravitational_waves.ligo_response_tacc(h_plus_gr, h_cross_gr)
    
    ax1.plot(t_fit * 1000, h_best_fit, 'b-', linewidth=2, 
             label=f'TACC Best Fit (κ={fit_result["kappa"]:.2f})')
    ax1.plot(t_fit * 1000, h_gr, 'r--', linewidth=2, label='GR (κ=2.0)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Strain')
    ax1.set_title('Parameter Estimation: Best Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-100, 50])
    
    # Residuals
    residuals = h_best_fit - h_gr
    ax2.plot(t_fit * 1000, residuals, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals (TACC - GR)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-100, 50])
    
    plt.tight_layout()
    plt.savefig(out_dir / "parameter_estimation.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Multi-messenger constraints
    print("Analyzing multi-messenger constraints...")
    
    # Simulate GW170817-like event
    gw_time = 0.0  # Reference time
    em_time = 1.74  # seconds after GW (approximate for GW170817)
    distance_170817 = 130e6 * 9.461e15  # meters
    
    mm_constraints = {}
    for kappa in kappa_fine:
        constraint = gravitational_waves.multi_messenger_constraints(
            gw_time, em_time, distance_170817, kappa
        )
        mm_constraints[kappa] = constraint
    
    # Extract constraint data
    residuals = [mm_constraints[k]['residual'] for k in kappa_fine]
    speed_differences = [mm_constraints[k]['fractional_speed_difference'] for k in kappa_fine]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time delay residuals
    ax1.plot(kappa_fine, residuals, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Perfect match')
    ax1.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='GR (κ=2)')
    # Add observational constraint (±1s uncertainty)
    ax1.fill_between(kappa_fine, -1, 1, alpha=0.2, color='orange', label='±1s constraint')
    ax1.set_xlabel('κ (Constitutive Parameter)')
    ax1.set_ylabel('Time Delay Residual (s)')
    ax1.set_title('Multi-messenger Time Delay Constraint')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, 5])
    
    # Speed difference
    ax2.plot(kappa_fine, speed_differences, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7, label='v_gw = c')
    ax2.axvline(x=2.0, color='g', linestyle=':', alpha=0.7, label='GR (κ=2)')
    # GW170817 constraint: |v_gw/c - 1| < 3×10^-15
    constraint_level = 3e-15
    ax2.fill_between(kappa_fine, -constraint_level, constraint_level, 
                     alpha=0.2, color='orange', label='GW170817 constraint')
    ax2.set_xlabel('κ (Constitutive Parameter)')
    ax2.set_ylabel('(v_gw - c) / c')
    ax2.set_title('Gravitational Wave Speed Constraint')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1e-14, 1e-14])
    
    plt.tight_layout()
    plt.savefig(out_dir / "multi_messenger_constraints.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Generate synthetic data for testing
    print("Generating synthetic gravitational wave data...")
    t_synth, h_synth, noise_level = gravitational_waves.generate_synthetic_gw_data(
        M1, M2, distance, kappa_true=2.1, noise_snr=15.0, duration=0.5
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Synthetic data
    ax1.plot(t_synth * 1000, h_synth, 'b-', linewidth=1, alpha=0.7, label='Noisy Data')
    
    # Clean signal for comparison
    h_plus_clean, h_cross_clean = gravitational_waves.waveform_tacc(M1, M2, distance, t_synth, 2.1)
    h_clean = gravitational_waves.ligo_response_tacc(h_plus_clean, h_cross_clean)
    ax1.plot(t_synth * 1000, h_clean, 'r-', linewidth=2, label='Clean Signal')
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Strain')
    ax1.set_title('Synthetic Gravitational Wave Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-100, 100])
    
    # Whitened data (simple high-pass filter)
    from scipy import signal
    b, a = signal.butter(4, 20, fs=len(t_synth)/(t_synth[-1]-t_synth[0]), btype='high')
    h_whitened = signal.filtfilt(b, a, h_synth)
    
    ax2.plot(t_synth * 1000, h_whitened, 'g-', linewidth=1.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Whitened Strain')
    ax2.set_title('Whitened (Filtered) Data')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-100, 100])
    
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_data.png", dpi=160, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(out_dir / "results.txt", 'w') as f:
        f.write("TACC Gravitational Wave Propagation Results\n")
        f.write("==========================================\n\n")
        
        f.write("Parameter Estimation Results:\n")
        f.write(f"  Best-fit κ = {fit_result['kappa']:.6f}\n")
        f.write(f"  χ² = {fit_result['chi2']:.3f}\n")
        f.write(f"  Fit successful: {fit_result['success']}\n")
        f.write(f"  Binary masses: {fit_result['M1']/M_sun:.1f} + {fit_result['M2']/M_sun:.1f} M☉\n")
        f.write(f"  Distance: {fit_result['distance']/9.461e15/1e6:.0f} Mpc\n\n")
        
        f.write("Multi-messenger Constraints (GW170817-like):\n")
        f.write(f"  GW-EM time difference: {em_time:.2f} s\n")
        f.write(f"  Distance: {distance_170817/9.461e15/1e6:.0f} Mpc\n")
        f.write(f"  Speed constraint: |v_gw/c - 1| < 3×10⁻¹⁵\n\n")
        
        f.write("TACC Dispersion Relations:\n")
        f.write("  v_gw = c√B(N_background)\n")
        f.write("  Time delay: Δt = d(1/v_gw - 1/c)\n")
        f.write("  For κ=2: v_gw = c exactly (GR recovery)\n")
        f.write("  For κ≠2: v_gw ≠ c (observable deviation)\n\n")
        
        f.write("Physical Interpretation:\n")
        f.write("- κ controls gravitational wave generation and propagation\n")
        f.write("- Orbital dynamics modified by computational capacity\n")
        f.write("- Wave speed depends on background computational capacity\n")
        f.write("- Multi-messenger astronomy provides tight constraints\n")
        f.write("- Binary inspirals probe strong-field regime of TACC\n\n")
        
        f.write("Key Insights:\n")
        f.write("- GW generation scales with B(N) at orbital separation\n")
        f.write("- Phase evolution modified by computational constraints\n")
        f.write("- Dispersion relation: ω² = k²c²B(N)\n")
        f.write("- LIGO/Virgo detection provides κ constraints\n")
        f.write("- Future space-based detectors (LISA) will improve constraints\n")
    
    print(f"\nGravitational waves experiment completed! Results saved to: {out_dir}")
    print("Generated files:")
    print("- waveform_comparison.png: Polarizations and detector response")
    print("- dispersion_analysis.png: Wave speed and time delays")
    print("- phase_analysis.png: Phase evolution and frequency domain")
    print("- parameter_estimation.png: Best-fit waveforms and residuals")
    print("- multi_messenger_constraints.png: Multi-messenger astronomy constraints")
    print("- synthetic_data.png: Synthetic data generation and filtering")
    print("- results.txt: Detailed analysis and physical interpretation")

if __name__ == "__main__":
    main()
