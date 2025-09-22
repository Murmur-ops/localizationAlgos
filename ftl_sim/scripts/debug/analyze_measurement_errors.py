#!/usr/bin/env python3
"""
Analyze actual measurement errors in simulation
"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig, compute_rms_bandwidth
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.clocks import ClockModel, ClockEnsemble
from ftl.rx_frontend import matched_filter, detect_toa
from ftl.measurement_covariance import compute_measurement_covariance

np.random.seed(42)

sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)
ch_config = ChannelConfig(environment='indoor_office')
channel_model = SalehValenzuelaChannel(ch_config)

# Set up clocks
n_nodes = 2
clock_model = ClockModel(oscillator_type="TCXO")
clock_ensemble = ClockEnsemble(n_nodes, clock_model)

template = gen_hrp_burst(sig_config, n_repeats=3)
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)

print("Analyzing measurement errors vs predicted variances:")
print("="*60)

# Test parameters
distance = 20.0
n_trials = 20

actual_errors = []
predicted_stds = []

for trial in range(n_trials):
    # Different random channel each time
    is_los = np.random.rand() > 0.3  
    channel = channel_model.generate_channel_realization(distance, is_los=is_los)
    
    # Get clock states
    clock_0 = clock_ensemble.states[0]
    clock_1 = clock_ensemble.states[1]
    clock_bias = clock_1.bias - clock_0.bias
    cfo_hz = clock_1.cfo - clock_0.cfo
    sco_ppm = clock_1.sco_ppm - clock_0.sco_ppm
    
    # True ToA
    true_toa = distance / 3e8 + clock_bias
    
    # Simulate measurement
    snr_db = 15.0
    result = propagate_signal(
        template, channel, sig_config.sample_rate,
        snr_db=snr_db, cfo_hz=cfo_hz, sco_ppm=sco_ppm,
        clock_bias_s=clock_bias
    )
    
    # Detect ToA
    correlation = matched_filter(result['signal'], template)
    toa_result = detect_toa(correlation, sig_config.sample_rate)
    
    # Compute predicted variance
    meas_cov = compute_measurement_covariance(
        correlation, template, sig_config.sample_rate,
        use_feature_scaling=True
    )
    
    # Record error and prediction
    error = abs(toa_result['toa'] - true_toa)
    actual_errors.append(error)
    predicted_stds.append(np.sqrt(meas_cov.toa_variance))
    
    if trial < 5:
        print(f"Trial {trial+1}:")
        print(f"  LOS: {is_los}")
        print(f"  True ToA: {true_toa*1e9:.3f} ns")
        print(f"  Measured ToA: {toa_result['toa']*1e9:.3f} ns")
        print(f"  Error: {error*1e12:.1f} ps ({error*3e8*100:.2f} cm)")
        print(f"  Predicted σ: {np.sqrt(meas_cov.toa_variance)*1e12:.1f} ps")
        print(f"  Variance used: {meas_cov.toa_variance:.2e} s²")

print("\n" + "="*60)
print("Summary:")
actual_std = np.std(actual_errors)
mean_predicted_std = np.mean(predicted_stds)

print(f"  Actual σ_ToA: {actual_std*1e12:.1f} ps ({actual_std*3e8*100:.2f} cm)")
print(f"  Mean predicted σ_ToA: {mean_predicted_std*1e12:.1f} ps")
print(f"  Ratio: {actual_std/mean_predicted_std:.2f}x")

# Check if variances are hitting the floor
min_var_floor = 1e-12
n_at_floor = sum(1 for std in predicted_stds if std**2 <= min_var_floor*1.01)
print(f"\n  Measurements at variance floor: {n_at_floor}/{n_trials}")
