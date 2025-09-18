# Realistic RF Simulation Implementation Plan

## Executive Summary
Transform the FTL simulation from 75% fake to 90% realistic by integrating proven RF models from RadarSim and implementing missing physics components.

## Phase 1: Integrate RadarSim Propagation Models (Week 1)

### 1.1 Import Realistic Channel Models
**FROM:** `/RadarSim/src/radar_simulation/propagation.py`
- [ ] PropagationChannel class with multipath
- [ ] Swerling fluctuation models for RCS
- [ ] Atmospheric attenuation models
- [ ] Clutter and interference models

**Implementation Tasks:**
```python
# Replace fake propagation in ftl_complete_demo.py line 111-126
from RadarSim.propagation import PropagationChannel, MultipathParameters

channel = PropagationChannel(
    propagation_mode=PropagationMode.MULTIPATH,
    enable_fluctuations=True
)
```

### 1.2 Add Doppler Effects
- [ ] Implement proper Doppler shift calculations
- [ ] Add time-varying channel response
- [ ] Include acceleration effects

## Phase 2: Hardware Impairments (Week 1-2)

### 2.1 I/Q Imbalance and Phase Noise
**FROM:** `/RadarSim/src/iq_generator.py`
- [ ] Import IQ signal generation with impairments
- [ ] Add ADC quantization effects
- [ ] Implement phase noise models

**Key Parameters:**
- I/Q amplitude imbalance: ±0.5 dB typical
- I/Q phase imbalance: ±5° typical
- Phase noise: -80 dBc/Hz @ 1kHz offset
- ADC bits: 12-14 bits typical

### 2.2 Oscillator Models
- [ ] Crystal drift: ±10 ppb/day
- [ ] Temperature coefficient: ±1 ppm/°C
- [ ] Allan variance modeling
- [ ] Aging effects: ±1 ppm/year

## Phase 3: Signal Processing Realism (Week 2)

### 3.1 Acquisition and Tracking
**FROM:** `/RadarSim/src/radar_simulation/processing.py`
- [ ] Code acquisition with serial/parallel search
- [ ] Early-prompt-late correlators
- [ ] Carrier tracking loops (Costas, PLL)
- [ ] Code tracking loops (DLL)

**Implementation:**
```python
class GoldCodeAcquisition:
    def __init__(self, search_range_hz=10000, search_bins=100):
        self.doppler_bins = np.linspace(-search_range_hz, search_range_hz, search_bins)
        self.code_phases = np.arange(0, code_length)

    def acquire(self, signal):
        # 2D search over frequency and code phase
        for freq in self.doppler_bins:
            for phase in self.code_phases:
                correlation = self.correlate_at_offset(signal, freq, phase)
```

### 3.2 Realistic Correlation
- [ ] FFT-based correlation for efficiency
- [ ] Matched filter implementation
- [ ] Pulse compression with realistic sidelobes
- [ ] Range-Doppler processing

## Phase 4: Time Synchronization Protocol (Week 2-3)

### 4.1 Multi-Round Sync Protocol
Based on our `ftl_convergence_demo.py`:
- [ ] Implement Kalman-filtered time sync
- [ ] Add drift estimation and compensation
- [ ] Include packet delays and retransmissions
- [ ] Model asymmetric path delays

**Protocol Structure:**
```python
class RealisticTimeSync:
    def __init__(self):
        self.sync_rounds = 10
        self.kalman_filter = KalmanTimeFilter()

    def sync_exchange(self, node_i, node_j):
        # Two-way time transfer
        t1 = node_i.send_timestamp()
        t2 = node_j.receive_timestamp()
        t3 = node_j.send_timestamp()
        t4 = node_i.receive_timestamp()

        # Kalman update with measurements
        self.kalman_filter.update(t1, t2, t3, t4)
```

### 4.2 Network Effects
- [ ] Packet loss modeling (1-5% typical)
- [ ] Variable network delays
- [ ] Protocol overhead
- [ ] MAC layer collisions

## Phase 5: Multipath and Interference (Week 3)

### 5.1 Multipath Propagation
**FROM:** `/RadarSim/src/radar_simulation/propagation.py`
- [ ] Ray tracing for multipath components
- [ ] Rician/Rayleigh fading models
- [ ] Delay spread and coherence bandwidth
- [ ] Angle of arrival variations

### 5.2 Interference Modeling
- [ ] Co-channel interference
- [ ] Adjacent channel interference
- [ ] Intentional jamming
- [ ] Unintentional RFI

## Phase 6: Environmental Effects (Week 3-4)

### 6.1 Atmospheric Effects
- [ ] Temperature gradients
- [ ] Humidity absorption
- [ ] Ionospheric delays (if applicable)
- [ ] Rain attenuation

### 6.2 Ground Effects
- [ ] Ground reflection multipath
- [ ] Terrain shadowing
- [ ] Diffraction effects
- [ ] Surface roughness

## Phase 7: Validation and Testing (Week 4)

### 7.1 Theoretical Validation
- [ ] Compare with Cramér-Rao bounds
- [ ] Validate against published results
- [ ] Check SNR-dependent performance
- [ ] Verify multipath resilience

### 7.2 Performance Metrics
- [ ] Acquisition probability vs SNR
- [ ] Time sync convergence rates
- [ ] Localization accuracy vs geometry
- [ ] Multipath error statistics

## Implementation Priority

### CRITICAL (Must Have):
1. **Realistic propagation** - Use RadarSim's PropagationChannel
2. **Hardware impairments** - I/Q imbalance, phase noise
3. **Iterative time sync** - Multi-round Kalman filtering
4. **Multipath effects** - At least 2-ray model

### IMPORTANT (Should Have):
5. **Acquisition/tracking loops** - Serial search, DLL/PLL
6. **Atmospheric effects** - Basic attenuation
7. **Interference modeling** - AWGN + narrowband
8. **Network delays** - Packet loss, retransmissions

### NICE TO HAVE:
9. **Advanced multipath** - Ray tracing
10. **Terrain effects** - Diffraction, shadowing
11. **Weather effects** - Rain, fog
12. **Antenna patterns** - Directional gains

## File Modifications Required

### Core Files to Update:
1. `ftl_complete_demo.py` - Replace fake propagation (lines 111-126)
2. `spread_spectrum.py` - Add real correlation (line 116)
3. `ftl_demo.py` - Fix PLL/Kalman (lines 166-201)
4. `demo_simple.py` - Add channel models

### New Files to Create:
1. `rf_channel.py` - Integrated channel model from RadarSim
2. `hardware_impairments.py` - I/Q, phase noise, ADC
3. `acquisition_tracking.py` - Code/carrier loops
4. `time_sync_protocol.py` - Realistic sync

### Files to Import from RadarSim:
1. `propagation.py` - Channel models
2. `iq_generator.py` - Signal generation
3. `processing.py` - DSP functions
4. `environment.py` - Environmental models

## Success Metrics

### Before (Current State):
- 25% Real / 75% Fake
- No multipath modeling
- Single-pass time sync
- No hardware effects
- Simplified correlation

### After (Target State):
- 90% Real / 10% Simplified
- Full multipath propagation
- Iterative Kalman sync
- I/Q and phase noise
- FFT-based correlation
- Acquisition/tracking loops

## Estimated Timeline
- **Week 1**: Propagation models + hardware impairments
- **Week 2**: Signal processing + time sync
- **Week 3**: Multipath + interference
- **Week 4**: Validation + documentation

## Risk Mitigation
1. **Complexity**: Start with 2-ray model, add complexity incrementally
2. **Performance**: Use FFT correlation, optimize critical paths
3. **Validation**: Compare each component against theory before integration
4. **Dependencies**: Copy needed RadarSim modules to avoid coupling

## Next Steps
1. Copy PropagationChannel from RadarSim
2. Create rf_channel.py wrapper
3. Replace fake propagation in ftl_complete_demo.py
4. Add unit tests for each component
5. Validate against Cramér-Rao bounds