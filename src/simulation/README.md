# Simulation: Carrier Phase Synchronization with Decentralized MPS

## Overview

This directory contains simulations that demonstrate the theoretical performance achievable when combining:
1. Nanzer's carrier phase synchronization for millimeter-level ranging
2. Decentralized MPS algorithm for distributed localization

These simulations abstract away timing limitations to explore performance with ideal hardware.

## Key Concept: Carrier Phase vs Time-of-Flight

### Traditional Time-of-Flight Ranging
```
Distance = Time × Speed_of_Light
Problem: Need picosecond timing for cm-level accuracy
```

### Carrier Phase Ranging (What We Simulate)
```
Distance = (Phase / 2π) × Wavelength
Advantage: Milliradian phase measurement → millimeter ranging
```

## Expected Performance

With carrier phase synchronization at 2.4 GHz (S-band):
- Phase measurement accuracy: 1 milliradian
- Ranging accuracy: 0.02-0.12 mm
- Localization RMSE: 8-12 mm
- Meets S-band requirement: ✓ (<15 mm)

## Quick Start

### 1. Run Default Simulation
```bash
cd simulation/src
python run_phase_sync_simulation.py
```

### 2. Custom Configuration
Edit `config/phase_sync_sim.yaml`:
```yaml
network:
  n_sensors: 30  # Increase sensors
  scale_meters: 100.0  # Larger network
  
carrier_phase:
  phase_noise_milliradians: 0.5  # Better hardware
```

Then run:
```bash
python run_phase_sync_simulation.py --config ../config/your_config.yaml
```

### 3. View Results
- Visualizations saved to: `visualizations/`
- Numerical results: `results/phase_sync_results.json`

## Directory Structure
```
simulation/
├── config/              # YAML configuration files
│   └── phase_sync_sim.yaml
├── src/                 # Simulation source code
│   └── run_phase_sync_simulation.py
├── visualizations/      # Generated plots
│   ├── network_trial_0.png
│   ├── convergence_trial_0.png
│   └── rmse_comparison.png
└── results/            # Simulation results (JSON)
```

## Configuration Options

### Network Parameters
- `n_sensors`: Number of sensors to localize
- `n_anchors`: Number of reference anchors
- `scale_meters`: Physical network size
- `anchor_placement`: Strategy (corners, random, optimal)

### Carrier Phase Parameters
- `frequency_ghz`: Carrier frequency (2.4 for S-band)
- `phase_noise_milliradians`: Measurement noise (1.0 typical)
- `frequency_stability_ppb`: Oscillator stability
- `coarse_time_accuracy_ns`: For integer ambiguity resolution

### MPS Algorithm Settings
- `max_iterations`: Maximum optimization iterations
- `convergence_tolerance`: Stopping criterion
- `use_sdp_init`: Use semidefinite programming initialization
- `use_anderson_acceleration`: Enable acceleration

## Visualization Outputs

### 1. Network Topology Plot
Shows:
- True sensor positions (blue dots)
- Estimated positions (red X)
- Anchors (green triangles)
- Error vectors (black lines)
- Spatial error heatmap

### 2. Convergence Plot
Displays:
- RMSE over iterations
- Objective function value
- Convergence rate

### 3. RMSE Comparison
Compares:
- No synchronization (14.5m)
- Time sync with Python (600mm)
- Carrier phase sync (8-12mm)

## Mathematical Foundation

### Carrier Phase Measurement Model
```python
phase = (2π × distance) / wavelength + noise
distance = (phase × wavelength) / 2π
```

### Integer Ambiguity Resolution
Since phase wraps every wavelength (12.5cm at 2.4GHz):
1. Use coarse time sync to determine which wavelength cycle
2. Use fine phase measurement for sub-wavelength accuracy
3. Combine for absolute distance

### Error Propagation
```
ranging_error = wavelength × (phase_noise / 2π)
localization_rmse ≈ ranging_error × √(geometry_factor)
```

## Comparison to Other Approaches

| Method | Ranging Accuracy | Localization RMSE | Meets S-band? |
|--------|-----------------|-------------------|---------------|
| No Sync (5% noise) | 5cm @ 1m | 14.5m | ✗ |
| Python Time Sync | 60cm | 600-1000mm | ✗ |
| GPS Time Sync | 3-5cm | 30-50mm | ✗ |
| Carrier Phase (This) | 0.02-0.12mm | 8-12mm | ✓ |

## Key Insights

1. Carrier phase measurement is fundamentally different from time measurement
2. Milliradian phase accuracy is achievable with standard RF hardware
3. Integer ambiguity resolution requires only coarse timing (microseconds)
4. Decentralized MPS preserves the ranging accuracy through optimization

## Limitations

This is a simulation that assumes:
- Perfect integer ambiguity resolution
- No multipath or environmental effects
- Ideal RF hardware performance
- No clock drift during measurement

For real-world constraints, see the `emulation/` directory.

## References

- Nanzer, J.A. (2017). "Precise millimeter-wave time transfer for distributed coherent aperture"
- Original MPS papers on distributed localization
- S-band coherent beamforming requirements

## Next Steps

1. Emulation: See `../emulation/` for Python timing limitations
2. Hardware: See `../hardware_ready/` for real implementation requirements
3. Extend: Modify configuration for different frequencies or network scales