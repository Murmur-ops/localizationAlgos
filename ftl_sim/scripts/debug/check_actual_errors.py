#!/usr/bin/env python3
"""
Check actual measurement errors vs CRLB predictions
"""

import numpy as np
from ftl.signal import gen_hrp_burst, SignalConfig
from ftl.channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from ftl.rx_frontend import matched_filter, detect_toa, toa_crlb
from ftl.signal import compute_rms_bandwidth

np.random.seed(42)

sig_config = SignalConfig(bandwidth=499.2e6, sample_rate=2e9)
ch_config = ChannelConfig(environment='indoor_office')
channel_model = SalehValenzuelaChannel(ch_config)

template = gen_hrp_burst(sig_config, n_repeats=3)
beta_rms = compute_rms_bandwidth(template, sig_config.sample_rate)

print("Comparing CRLB predictions to actual errors:")
print("="*60)

distances = [10, 20, 30, 40]
snr_dbs = [10, 15, 20]

for dist in distances:
    for snr_db in snr_dbs:
        # Generate channel
        channel = channel_model.generate_channel_realization(dist, is_los=True)
        
        # True ToA
        true_toa = dist / 3e8
        
        errors = []
        crlb_stds = []
        
        # Run multiple trials
        for _ in range(10):
            # Propagate signal
            result = propagate_signal(
                template, channel, sig_config.sample_rate,
                snr_db=snr_db, cfo_hz=0, sco_ppm=0, clock_bias_s=0
            )
            
            # Detect ToA
            correlation = matched_filter(result['signal'], template)
            toa_result = detect_toa(correlation, sig_config.sample_rate)
            
            # Actual error
            error = abs(toa_result['toa'] - true_toa)
            errors.append(error)
            
            # CRLB prediction
            snr_linear = 10**(snr_db/10)
            crlb_var = toa_crlb(snr_linear, beta_rms)
            crlb_stds.append(np.sqrt(crlb_var))
        
        actual_std = np.std(errors)
        crlb_std = np.mean(crlb_stds)
        
        print(f"d={dist}m, SNR={snr_db}dB:")
        print(f"  CRLB σ_ToA: {crlb_std*1e12:.1f} ps ({crlb_std*3e8*100:.2f} cm)")
        print(f"  Actual σ_ToA: {actual_std*1e12:.1f} ps ({actual_std*3e8*100:.2f} cm)")
        print(f"  Ratio: {actual_std/crlb_std:.1f}x")

print("\n" + "="*60)
print("FINDING: Actual errors are much larger than CRLB!")
print("This explains why using CRLB variances directly causes issues.")
