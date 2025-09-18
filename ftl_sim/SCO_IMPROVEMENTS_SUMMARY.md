# Sample Clock Offset (SCO) Improvements

## Overview
These changes add Sample Clock Offset (SCO) modeling to the FTL system, enabling more realistic RF simulations with ADC sample rate errors.

---

## Key Changes

### 1. Clock State Extension (3D → 4D)
**File**: `ftl/clocks.py`
- Added `sco_ppm` to ClockState
- State vector now: `[bias, drift, cfo, sco_ppm]`
- SCO coherent with oscillator drift (same crystal drives RF and ADC)
- Process noise model includes SCO evolution

### 2. SCO Signal Processing
**File**: `ftl/channel.py`
- Added `apply_sample_clock_offset()` function
- Uses polyphase resampling for accurate SCO simulation
- Models receiver sampling at different rate than transmitter
- Handles both fast and slow sampling cases

### 3. Enhanced RX Frontend
**File**: `ftl/rx_frontend.py`
Major improvements:
- Better ToA detection with sub-sample refinement
- Enhanced NLOS detection with more correlation features
- CRLB-based covariance calculation
- Feature-based variance inflation for multipath
- SCO estimation from correlation peak width
- Added 8 correlation shape features:
  - RMS width
  - Peak-to-sidelobe ratio
  - Multipath ratio
  - Excess delay
  - Lead width (rise time)
  - Rise slope
  - Early-late energy ratio
  - Kurtosis

### 4. Signal Generation Updates
**File**: `ftl/signal.py`
- Updated SignalConfig dataclass
- Added RMS bandwidth calculation
- Better structured waveform generation
- Improved pilot tone handling

### 5. Solver Enhancements
**File**: `ftl/solver.py`
- Minor improvements to handle 4D clock states
- Better error handling

---

## Impact on System

### Timing Accuracy
- SCO causes accumulating timing errors
- 10 ppm SCO over 1 μs → 10 ps time error → 3 mm range error
- Critical for high-precision localization

### NLOS Robustness
Enhanced correlation features enable better:
- Multipath detection
- NLOS classification
- Adaptive variance inflation
- Improved outlier rejection

### Realistic Simulation
- Models real hardware imperfections
- ADC sample rate mismatch between nodes
- Coherent CFO/SCO from same oscillator
- More accurate error propagation

---

## Testing Status

### Working ✓
- Clock state with SCO (4D)
- SCO initialization and propagation
- SCO application to signals
- Basic clock tests (18 passing)

### Needs Update
- Signal tests need updating for new SignalConfig
- Some test arguments need adjustment

---

## Example Usage

```python
# Clock with SCO
state = ClockState(
    bias=1e-6,      # 1 μs
    drift=1e-9,     # 1 ppb
    cfo=100.0,      # 100 Hz
    sco_ppm=5.0     # 5 ppm ADC error
)

# Apply SCO to signal
signal_with_sco = apply_sample_clock_offset(
    signal,
    sco_ppm=10.0,
    sample_rate=1e9
)

# Enhanced ToA detection
result = detect_toa(
    correlation,
    sample_rate,
    mode='leading_edge',  # Better for NLOS
    enable_subsample=True
)

# NLOS classification
features = extract_correlation_features(correlation, peak_idx)
classification = classify_propagation(correlation)
if classification['type'] == 'NLOS':
    variance *= 2.0  # Inflate for NLOS
```

---

## Next Steps

1. Fix signal tests to match new SignalConfig
2. Integrate SCO into full FTL pipeline
3. Test impact on consensus performance
4. Add SCO compensation algorithms
5. Document SCO effects on localization accuracy

These improvements make the FTL system more realistic by modeling actual hardware imperfections that affect RF ranging systems.