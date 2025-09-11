# FTL System Flow: From Boot to Localization

## System Overview
The system performs distributed localization through three critical phases:
1. **Frequency Synchronization** - Align carrier and sampling clocks
2. **Time Synchronization** - Establish common time reference
3. **Localization** - Determine positions using synchronized measurements

---

## Phase 0: System Initialization (Boot)

### Hardware Initialization
```
[POWER ON]
    ↓
[Initialize RF Frontend]
    - Set carrier frequency (2.4 GHz)
    - Configure bandwidth (100 MHz)
    - Set TX power (20 dBm)
    - Initialize ADC/DAC (200 Msps)
    ↓
[Initialize Timing Hardware]
    - Start local oscillator
    - Initialize MAC/PHY timestamp unit
    - Set timestamp resolution (10ns)
    ↓
[Load Waveform Parameters]
    - Gold codes (1023 chips)
    - Pilot frequencies [-5, -3, -1, 0, 1, 3, 5] MHz
    - Frame structure timings
```

### Software Initialization
```python
# From test_full_system.py
class SimulatedNode:
    def __init__(self, config: NodeConfig):
        # RF components
        self.generator = SpreadSpectrumGenerator(config)
        self.correlator = RangingCorrelator(config)
        
        # Sync components  
        self.pll = PilotPLL(PLLConfig())
        self.time_sync = PTPTimeSync(TimeSyncConfig(), node_id)
        
        # State machine
        self.state = NodeState.DISCOVERING
```

---

## Phase 1: Network Discovery

### State: DISCOVERING
```
[Node Boots]
    ↓
[Listen for Beacons]
    - Scan for BEACON messages
    - Detect active nodes
    ↓
[Transmit Own Beacon]
    - Include: node_id, is_anchor, position (if anchor)
    - Use TDMA slot based on hash(node_id)
    ↓
[Build Neighbor Table]
    - Record discovered nodes
    - Note which are anchors
    - Store initial positions (anchors only)
```

**Message Flow:**
```
Node 1 (Anchor): →[BEACON: id=1, pos=(0,0), anchor=true]→ All nodes
Node 2 (Anchor): →[BEACON: id=2, pos=(10,0), anchor=true]→ All nodes  
Node 5 (Unknown): →[BEACON: id=5, pos=null, anchor=false]→ All nodes
```

---

## Phase 2: Frequency Synchronization (The 'F' in FTL)

### State: SYNCING (Frequency)
```
[Receive Pilot Signal]
    ↓
[Coarse CFO Estimation]
    - Use autocorrelation on repeated symbols
    - Get rough frequency offset estimate
    ↓
[PLL Tracking]
    - Phase detector on pilot tones
    - Loop filter (PI controller)
    - Update frequency estimate
    ↓
[Monitor Lock Status]
    - Check phase variance < 0.1 rad
    - Check frequency variance < 100 Hz
    - Declare lock when stable
```

**Implementation (frequency_sync.py):**
```python
class PilotPLL:
    def process_pilot(self, received_signal):
        # 1. Coarse acquisition
        if self.coarse_cfo_estimate is None:
            self.coarse_cfo_estimate = self.estimate_coarse_cfo(signal)
            
        # 2. Fine tracking loop
        for sample in received_signal:
            # Compensate with current estimate
            compensated = sample * np.exp(-1j * (self.phase + 2*π*self.frequency*t))
            
            # Phase error detection
            phase_error = np.angle(compensated)
            
            # Loop filter update
            self.frequency += alpha * phase_error + beta * phase_error_integral
            
        # 3. Check lock
        self.locked = (phase_variance < 0.1 and freq_variance < 100)
```

**Consensus (Optional):**
```python
# Distributed frequency agreement
class DistributedFrequencyConsensus:
    def consensus_update(self):
        # Average with neighbors
        new_cfo = weight_self * local_cfo + Σ(weight_neighbor * neighbor_cfo)
```

---

## Phase 3: Time Synchronization (The 'T' in FTL)

### State: SYNCING (Time)
```
[PTP-Style Exchange]
    ↓
[Node A: Record t1] → [SYNC_REQ] → [Node B: Record t2]
                                    ↓
[Node A: Record t4] ← [SYNC_RESP(t2,t3)] ← [Node B: Record t3]
    ↓
[Calculate Offset and Delay]
    - offset = ((t2-t1) - (t4-t3))/2
    - round_trip = (t4-t1) - (t3-t2)
    - one_way_delay = round_trip/2
    ↓
[Kalman Filter Update]
    - Track offset and drift
    - Estimate clock skew
```

**Implementation (frequency_sync.py):**
```python
class PTPTimeSync:
    def process_sync_exchange(self, t1, t2, t3, t4):
        # Calculate time offset
        offset_ns = ((t2 - t1) - (t4 - t3)) / 2.0
        rtt_ns = (t4 - t1) - (t3 - t2)
        
        # Update Kalman filter
        kf.update(offset_ns, rtt_ns)
        
        return {
            'offset_ns': offset_ns,
            'filtered_offset_ns': kf.offset_estimate,
            'filtered_skew_ppm': kf.skew_estimate * 1e6
        }
```

---

## Phase 4: Ranging Measurements

### State: RANGING
```
[Generate Ranging Signal]
    - Spread spectrum waveform (Gold code)
    - Add pilot tones
    ↓
[Transmit in TDMA Slot]
    - Use assigned time slot
    - Record TX timestamp (t_tx)
    ↓
[Receive and Correlate]
    - Correlate with known PN sequence
    - Find peak (Time of Arrival)
    - Sub-sample interpolation
    - Record RX timestamp (t_rx)
    ↓
[Calculate Distance]
    - Apply time sync corrections
    - distance = c * (t_rx - t_tx - offset) / 2
    ↓
[Quality Assessment]
    - SNR from correlation peak
    - Multipath detection
    - NLOS identification
```

**Implementation (spread_spectrum.py + test_full_system.py):**
```python
class RangingCorrelator:
    def correlate(self, received_signal):
        # Cross-correlation
        correlation = signal.correlate(received_signal, reference_pn)
        peak_idx = np.argmax(correlation)
        
        # Sub-sample interpolation (parabolic)
        y1, y2, y3 = correlation[peak_idx-1:peak_idx+2]
        x_offset = -(y3-y1)/(2*(y1-2*y2+y3))
        fine_peak = peak_idx + x_offset
        
        # Time of arrival
        toa_seconds = fine_peak / sample_rate
        
        # Quality metrics
        snr_db = 10*log10(peak_value/noise_floor)
        
        return {'toa_seconds': toa_seconds, 'snr_db': snr_db}
```

---

## Phase 5: Localization (The 'L' in FTL)

### State: LOCALIZING
```
[Collect Measurements]
    - Gather all ranging results
    - Filter outliers
    - Weight by quality scores
    ↓
[Initial Position Estimate]
    - Random initialization near center
    - Or use previous position
    ↓
[Iterative Optimization]
    For each iteration:
        1. Compute residuals (measured - estimated distances)
        2. Apply Huber loss (robust to outliers)
        3. Calculate Jacobian
        4. Levenberg-Marquardt update
        5. Check convergence
    ↓
[Distributed Consensus (Optional)]
    - Share position estimates with neighbors
    - Apply ADMM consensus update
```

**Implementation (robust_solver.py):**
```python
class RobustLocalizer:
    def solve(self, initial_positions, measurements, anchor_positions):
        positions = initial_positions
        
        for iteration in range(max_iterations):
            # 1. Compute residuals
            residuals = []
            for edge in measurements:
                est_dist = norm(pos_i - pos_j)
                weight = sqrt(edge.quality / edge.variance)
                residual = weight * (est_dist - edge.distance)
                residuals.append(residual)
            
            # 2. Huber weighting for robustness
            weights = [huber_weight(r) for r in residuals]
            
            # 3. Compute cost
            cost = sum([huber_loss(r) for r in residuals])
            
            # 4. Levenberg-Marquardt update
            J = compute_jacobian(positions, measurements)
            H = J.T @ W @ J  # Weighted Hessian
            g = J.T @ W @ residuals  # Gradient
            
            # Add damping
            H_damped = H + lambda_lm * I
            delta = -solve(H_damped, g)
            
            # 5. Update positions
            positions += delta
            
            # 6. Check convergence
            if abs(cost - prev_cost) < threshold:
                break
                
        return positions
```

---

## Complete System Timeline

```
Time    Event                           State
----    -----                           -----
0ms     Power on                        BOOTING
10ms    Hardware initialized            DISCOVERING
100ms   First beacon sent              DISCOVERING
500ms   Neighbor table built           DISCOVERING → SYNCING

1000ms  Start frequency sync           SYNCING (F)
1100ms  Coarse CFO acquired           SYNCING (F)
1500ms  PLL locked                     SYNCING (F) → SYNCING (T)

2000ms  Start time sync                SYNCING (T)
2010ms  First PTP exchange (t1→t4)    SYNCING (T)
2020ms  Offset calculated              SYNCING (T)
2100ms  Multiple exchanges complete    SYNCING (T) → RANGING

3000ms  First ranging signal sent      RANGING
3001ms  TOA correlation complete       RANGING
3010ms  Distance calculated            RANGING
4000ms  All pairs measured             RANGING → LOCALIZING

4100ms  Optimization starts            LOCALIZING
4110ms  Iteration 1: RMSE = 10m       LOCALIZING
4120ms  Iteration 2: RMSE = 5m        LOCALIZING
4150ms  Iteration 5: RMSE = 0.5m      LOCALIZING
4200ms  Converged                      LOCALIZED

4500ms  Next ranging cycle begins      RANGING
```

---

## Key Design Decisions

### Why This Order (F→T→L)?

1. **Frequency First**: 
   - Can't demodulate signals without carrier lock
   - Required for coherent reception
   - Enables phase-based measurements

2. **Time Second**:
   - Needs working communication (requires frequency sync)
   - Critical for ranging accuracy
   - Establishes common reference frame

3. **Localization Last**:
   - Requires accurate range measurements
   - Needs synchronized time base
   - Builds on F and T foundations

### Hardware Dependencies

- **Frequency Sync**: Depends on oscillator stability (10 ppm drift)
- **Time Sync**: Limited by timestamp resolution (10ns → 3m)
- **Localization**: Limited by bandwidth (100 MHz → 1.5m resolution)

### Robustness Features

- **Coarse + Fine** frequency acquisition
- **Kalman filtering** for time sync
- **Huber loss** for NLOS resistance
- **Quality weighting** based on SNR
- **TDMA** to avoid collisions

---

## Summary

The FTL algorithm progresses through well-defined phases:

1. **Boot & Discovery** (100-500ms): Find neighbors, identify anchors
2. **Frequency Sync** (500-1000ms): Lock PLLs, align carriers
3. **Time Sync** (500-1000ms): PTP exchanges, Kalman filtering  
4. **Ranging** (1000-2000ms): Spread spectrum correlation, TOA
5. **Localization** (100-500ms): Robust optimization, position estimate

Total time from boot to first position: **~4-5 seconds**
Update rate after initial sync: **10-100 Hz**

The system achieves sub-meter accuracy with proper RF conditions and degrades gracefully with NLOS/multipath.