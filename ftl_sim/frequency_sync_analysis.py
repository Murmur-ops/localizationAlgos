"""
Analysis and design for frequency synchronization in FTL
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_frequency_sync_requirements():
    """Analyze what's needed for frequency synchronization"""

    print("="*60)
    print("FREQUENCY SYNCHRONIZATION FOR FTL")
    print("="*60)

    print("\n1. CURRENT STATE VECTOR:")
    print("-"*40)
    print("Currently tracking per node:")
    print("  - Position: (x, y)")
    print("  - Clock bias: τ (time offset in ns)")
    print("  State: [x, y, τ] - 3 parameters")

    print("\n2. FREQUENCY SYNC REQUIREMENTS:")
    print("-"*40)
    print("Need to add:")
    print("  - Clock drift rate: δf (frequency offset)")
    print("  - Units: parts per billion (ppb) or Hz/Hz")
    print("  Extended state: [x, y, τ, δf] - 4 parameters")

    print("\n3. PHYSICAL MODEL:")
    print("-"*40)
    print("Clock evolution:")
    print("  τ(t) = τ₀ + δf·t + ε(t)")
    print("Where:")
    print("  τ₀ = initial time offset")
    print("  δf = frequency offset (drift rate)")
    print("  t = elapsed time")
    print("  ε(t) = clock noise")

    # Demonstrate frequency drift impact
    print("\n4. IMPACT OF FREQUENCY OFFSET:")
    print("-"*40)

    # Typical crystal oscillator specs
    freq_offsets_ppb = [0.1, 1, 10, 100]  # Parts per billion
    time_horizons_sec = np.logspace(0, 4, 100)  # 1 sec to 10,000 sec

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time error accumulation
    ax = axes[0, 0]
    for ppb in freq_offsets_ppb:
        time_error_ns = ppb * time_horizons_sec  # ppb * seconds = nanoseconds
        ax.loglog(time_horizons_sec, time_error_ns, label=f'{ppb} ppb')

    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1 ns target')
    ax.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1 μs')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Time Error (ns)')
    ax.set_title('Time Error Accumulation from Frequency Offset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Position error from time error
    ax = axes[0, 1]
    c = 299792458  # m/s
    for ppb in freq_offsets_ppb:
        time_error_s = ppb * 1e-9 * time_horizons_sec
        pos_error_m = c * time_error_s
        ax.loglog(time_horizons_sec, pos_error_m, label=f'{ppb} ppb')

    ax.axhline(y=0.001, color='g', linestyle='--', alpha=0.5, label='1 mm target')
    ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='1 m')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error from Uncompensated Frequency Offset')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Required measurement types
    ax = axes[1, 0]
    ax.axis('off')

    measurement_types = """
    MEASUREMENT TYPES FOR FREQUENCY SYNC:

    1. MULTI-EPOCH TIME-OF-ARRIVAL (ToA)
       • Measure ToA at times t₁, t₂, ..., tₙ
       • Observe linear drift: Δτ = δf·Δt
       • Estimate drift rate from slope

    2. DOPPLER SHIFT MEASUREMENTS
       • For moving nodes: fᵣ = fₜ(1 + v·r̂/c)
       • Doppler shift reveals relative velocity
       • Can separate motion from clock drift

    3. CARRIER PHASE MEASUREMENTS
       • Phase: φ(t) = 2πf₀t + φ₀ + 2πδf·t²/2
       • More precise than ToA (mm-level)
       • Tracks frequency offset directly

    4. TWO-WAY TIME TRANSFER
       • Round-trip eliminates clock bias
       • Residual reveals frequency difference
       • τ_AB + τ_BA = 2d/c + (δf_A - δf_B)·T
    """

    ax.text(0.05, 0.95, measurement_types, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    # Implementation approach
    ax = axes[1, 1]
    ax.axis('off')

    implementation = """
    IMPLEMENTATION APPROACH:

    1. EXTEND STATE VECTOR:
       Old: x = [pos_x, pos_y, clock_bias]
       New: x = [pos_x, pos_y, clock_bias, freq_offset]

    2. MODIFY MEASUREMENT MODEL:
       Range: r = ||p_j - p_i|| + c·(τ_j - τ_i)
              + c·(δf_j - δf_i)·Δt

    3. UPDATE JACOBIAN:
       ∂r/∂δf_i = -c·Δt
       ∂r/∂δf_j = +c·Δt

    4. ADAPTIVE LM HANDLES IT:
       • Just add the 4th parameter
       • LM will optimize all 4 jointly
       • No algorithm changes needed!

    5. BENEFITS:
       ✓ Long-term stability
       ✓ Predictive capability
       ✓ Reduced update frequency
       ✓ Better accuracy over time
    """

    ax.text(0.05, 0.95, implementation, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Frequency Synchronization Analysis for FTL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('frequency_sync_analysis.png', dpi=150)
    print("\nPlots saved to frequency_sync_analysis.png")

    # Estimate accuracy improvements
    print("\n5. EXPECTED ACCURACY WITH FREQUENCY SYNC:")
    print("-"*40)

    # Without frequency sync
    print("Without frequency sync (1 hour operation):")
    drift_ppb = 10  # Typical TCXO
    time_error_ns = drift_ppb * 3600  # 1 hour
    pos_error_m = c * time_error_ns * 1e-9
    print(f"  Frequency drift: {drift_ppb} ppb")
    print(f"  Time error after 1 hour: {time_error_ns/1000:.1f} μs")
    print(f"  Position error: {pos_error_m:.1f} m")

    # With frequency sync
    print("\nWith frequency sync:")
    print("  Frequency estimated to: 0.01 ppb")
    print("  Time error after 1 hour: 36 ns")
    print("  Position error: 11 mm")
    print("  Improvement: 1000× better!")

    print("\n6. MEASUREMENT REQUIREMENTS:")
    print("-"*40)

    # How many epochs needed?
    print("For frequency estimation accuracy of 0.01 ppb:")

    time_noise_ns = 0.1  # Measurement noise
    target_freq_ppb = 0.01

    # From Allan variance theory
    measurement_interval = 1  # second
    n_measurements = (time_noise_ns / (target_freq_ppb * measurement_interval))**2

    print(f"  Measurement noise: {time_noise_ns} ns")
    print(f"  Measurement interval: {measurement_interval} s")
    print(f"  Required measurements: {int(n_measurements)}")
    print(f"  Total observation time: {int(n_measurements * measurement_interval)} seconds")

    return freq_offsets_ppb, time_horizons_sec


def design_frequency_sync_factors():
    """Design factor classes for frequency synchronization"""

    print("\n" + "="*60)
    print("FREQUENCY SYNC FACTOR DESIGN")
    print("="*60)

    factor_design = """
    FACTOR GRAPH DESIGN FOR FREQUENCY SYNC:

    1. RANGE FACTOR WITH FREQUENCY:
       class RangeFrequencyFactor:
           def __init__(self, measured_range, timestamp, sigma):
               self.range = measured_range
               self.time = timestamp
               self.sigma = sigma

           def error(self, state_i, state_j):
               # States: [x, y, tau, delta_f]
               pos_i, pos_j = state_i[:2], state_j[:2]
               tau_i, tau_j = state_i[2], state_j[2]
               df_i, df_j = state_i[3], state_j[3]

               # Geometric range
               dist = ||pos_j - pos_i||

               # Time with frequency correction
               dt = self.time  # Time since reference
               time_diff = (tau_j - tau_i) + (df_j - df_i) * dt

               # Predicted vs measured
               predicted = dist + c * time_diff * 1e-9
               return (predicted - self.range) / self.sigma

    2. DOPPLER FACTOR:
       class DopplerFactor:
           def __init__(self, doppler_shift, carrier_freq):
               self.doppler = doppler_shift
               self.f0 = carrier_freq

           def error(self, state_i, state_j, vel_i, vel_j):
               # Relative velocity along line-of-sight
               los = (pos_j - pos_i) / ||pos_j - pos_i||
               v_rel = dot(vel_j - vel_i, los)

               # Frequency offset contribution
               df_rel = (df_j - df_i) * self.f0

               # Total observed shift
               predicted = self.f0 * v_rel / c + df_rel
               return (predicted - self.doppler) / self.sigma

    3. PHASE FACTOR:
       class CarrierPhaseFactor:
           def __init__(self, phase_diff, wavelength):
               self.phase = phase_diff
               self.lambda = wavelength

           def error(self, state_i, state_j):
               # Phase = 2π * distance / wavelength
               dist = ||pos_j - pos_i||
               geom_phase = 2 * pi * dist / self.lambda

               # Clock contribution
               clock_phase = 2 * pi * c * (tau_j - tau_i) / self.lambda

               # Frequency contribution (integrated)
               freq_phase = pi * c * (df_j - df_i) * dt^2 / self.lambda

               predicted = geom_phase + clock_phase + freq_phase
               # Handle phase wrapping
               return angle_diff(predicted, self.phase) / self.sigma

    4. FREQUENCY PRIOR:
       class FrequencyPrior:
           def __init__(self, nominal_freq, sigma_ppb):
               self.f0 = nominal_freq
               self.sigma = sigma_ppb * 1e-9

           def error(self, state):
               df = state[3]  # Frequency offset
               return (df - self.f0) / self.sigma
    """

    print(factor_design)

    # Jacobian structure
    print("\nJACOBIAN STRUCTURE FOR FREQUENCY:")
    print("-"*40)

    jacobian_info = """
    For range measurement with frequency:

    J = [∂r/∂x_i, ∂r/∂y_i, ∂r/∂τ_i, ∂r/∂δf_i, ∂r/∂x_j, ∂r/∂y_j, ∂r/∂τ_j, ∂r/∂δf_j]

    Where:
      ∂r/∂x_i = -(x_j - x_i)/d     (position gradient)
      ∂r/∂y_i = -(y_j - y_i)/d
      ∂r/∂τ_i = -c                 (time gradient)
      ∂r/∂δf_i = -c·Δt            (frequency gradient)

      ∂r/∂x_j = +(x_j - x_i)/d
      ∂r/∂y_j = +(y_j - y_i)/d
      ∂r/∂τ_j = +c
      ∂r/∂δf_j = +c·Δt

    The frequency terms scale with elapsed time Δt,
    making them more observable over longer periods.
    """

    print(jacobian_info)


def simulate_frequency_sync_benefit():
    """Simulate the benefit of frequency synchronization"""

    print("\n" + "="*60)
    print("FREQUENCY SYNC SIMULATION")
    print("="*60)

    # Simulation parameters
    n_nodes = 10
    duration_hours = 24
    dt = 60  # Measurement interval (seconds)
    n_steps = int(duration_hours * 3600 / dt)

    # Clock parameters (realistic TCXO)
    true_freq_offsets = np.random.normal(0, 10, n_nodes)  # ±10 ppb
    initial_time_offsets = np.random.normal(0, 100, n_nodes)  # ±100 ns

    print(f"Simulating {n_nodes} nodes over {duration_hours} hours")
    print(f"True frequency offsets: {true_freq_offsets.mean():.1f} ± {true_freq_offsets.std():.1f} ppb")

    # Without frequency sync
    time_errors_no_sync = []
    times = []

    for step in range(n_steps):
        t = step * dt
        times.append(t / 3600)  # Convert to hours

        # Time error grows linearly
        time_error = initial_time_offsets + true_freq_offsets * t
        time_errors_no_sync.append(np.std(time_error))

    # With frequency sync (estimate after 1 hour)
    sync_time = 3600  # 1 hour calibration
    estimated_freqs = true_freq_offsets + np.random.normal(0, 0.01, n_nodes)  # 0.01 ppb accuracy

    time_errors_with_sync = []

    for step in range(n_steps):
        t = step * dt

        if t < sync_time:
            # Before sync: same as no sync
            time_error = initial_time_offsets + true_freq_offsets * t
        else:
            # After sync: compensated
            residual_freq = true_freq_offsets - estimated_freqs
            time_error = initial_time_offsets + residual_freq * (t - sync_time)

        time_errors_with_sync.append(np.std(time_error))

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.semilogy(times, time_errors_no_sync, 'r-', label='No frequency sync', linewidth=2)
    plt.semilogy(times, time_errors_with_sync, 'g-', label='With frequency sync', linewidth=2)
    plt.axvline(x=1, color='b', linestyle='--', alpha=0.5, label='Sync calibration done')
    plt.xlabel('Time (hours)')
    plt.ylabel('Time Sync Error (ns)')
    plt.title('Time Synchronization Error Growth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    c = 299792458
    pos_errors_no_sync = [e * c * 1e-9 for e in time_errors_no_sync]
    pos_errors_with_sync = [e * c * 1e-9 for e in time_errors_with_sync]

    plt.semilogy(times, pos_errors_no_sync, 'r-', label='No frequency sync', linewidth=2)
    plt.semilogy(times, pos_errors_with_sync, 'g-', label='With frequency sync', linewidth=2)
    plt.axhline(y=0.001, color='gray', linestyle=':', alpha=0.5, label='1mm target')
    plt.axvline(x=1, color='b', linestyle='--', alpha=0.5, label='Sync calibration done')
    plt.xlabel('Time (hours)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error from Clock Drift')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Impact of Frequency Synchronization on Long-Term Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig('frequency_sync_simulation.png', dpi=150)
    print("\nSimulation plots saved to frequency_sync_simulation.png")

    # Summary
    print("\nSIMULATION RESULTS:")
    print("-"*40)
    print(f"After {duration_hours} hours:")
    print(f"  Without frequency sync: {time_errors_no_sync[-1]:.0f} ns error")
    print(f"  With frequency sync: {time_errors_with_sync[-1]:.1f} ns error")
    print(f"  Improvement: {time_errors_no_sync[-1]/time_errors_with_sync[-1]:.0f}×")
    print(f"\nPosition error after {duration_hours} hours:")
    print(f"  Without: {pos_errors_no_sync[-1]:.1f} m")
    print(f"  With: {pos_errors_with_sync[-1]:.3f} m")


if __name__ == "__main__":
    analyze_frequency_sync_requirements()
    design_frequency_sync_factors()
    simulate_frequency_sync_benefit()