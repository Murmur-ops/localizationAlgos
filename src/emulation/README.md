# Emulation: Real Python Timing Constraints

## Overview

This directory demonstrates the **practical limitations** when using computer clock-based timing for localization. Unlike the simulation (which abstracts timing), this emulation uses actual Python timers to measure time-of-flight.

## Key Finding: Why Time-Based Localization Fails

Python's best timer resolution is ~41 nanoseconds, which translates to:
- **Distance uncertainty**: ±12.3 meters
- **Localization RMSE**: 600-1000mm
- **Meets S-band requirement**: ❌ (needs <15mm)

## Quick Start

### 1. Test Python Timing Limits
```bash
cd src/emulation
python test_python_timing_limits.py
```

Expected output:
```
PYTHON TIMER RESOLUTION TEST
time.perf_counter_ns():
  Minimum resolution: 41.0 ns
  → Distance resolution: 1229 cm
```

### 2. Run Time Synchronization Algorithms
```bash
# Two-Way Time Transfer (TWTT)
python twtt.py

# Frequency synchronization
python frequency_sync.py

# Consensus clock
python consensus_clock.py
```

## Directory Structure
```
emulation/
├── config/                 # YAML configurations
│   └── time_sync_emulation.yaml
├── test_python_timing_limits.py  # Test Python timer resolution
├── twtt.py                # Two-way time transfer
├── frequency_sync.py      # Frequency synchronization
├── consensus_clock.py     # Consensus clock algorithm
├── honest_summary.py      # Honest timing assessment
└── results/               # Emulation results (gitignored)
```

## Configuration Options

Edit `config/time_sync_emulation.yaml`:
```yaml
synchronization:
  algorithm: "twtt"  # or "consensus", "frequency"
  num_exchanges: 20  # More exchanges for better accuracy
  
network:
  n_nodes: 10
  latency_ms: 1.0  # Network latency simulation
```

## Timing Methods Compared

| Method | Timer Used | Resolution | Distance Error | Localization RMSE |
|--------|-----------|------------|----------------|-------------------|
| time.time() | System clock | ~1μs | ±300m | >1000mm |
| time.perf_counter() | High-res counter | ~41ns | ±12m | 600-1000mm |
| time.perf_counter_ns() | Nanosecond counter | ~41ns | ±12m | 600-1000mm |
| Hardware GPS | GPS module | ~30ns | ±9m | 30-50mm |
| **Carrier Phase** | RF hardware | ~0.1ns | ±3cm | 8-12mm |

## Why Computer Clocks Can't Achieve S-band Requirements

### 1. Timer Resolution Limit
- CPU clock cycles: ~0.3ns (3.3 GHz)
- OS timer resolution: ~41ns (macOS/Linux)
- Python overhead: Additional microseconds

### 2. Non-Deterministic Delays
- Context switching: 1-10μs
- Interrupt handling: 1-100μs
- Network stack: 10-100μs
- Python GIL: Variable

### 3. Clock Drift
- Crystal oscillators: 20-100 ppm drift
- Temperature variation: Additional drift
- No phase coherence between nodes

## Time Synchronization Algorithms

### Two-Way Time Transfer (TWTT)
```python
# Basic TWTT exchange
t1 = sender.send_time()
t2 = receiver.receive_time()
t3 = receiver.send_time()
t4 = sender.receive_time()

offset = ((t2 - t1) - (t4 - t3)) / 2
rtt = (t4 - t1) - (t3 - t2)
```

### Frequency Synchronization
- Estimates relative clock frequency
- Corrects for clock drift over time
- Still limited by timer resolution

### Consensus Clock
- Distributed agreement on common time
- Averages out individual clock errors
- Cannot overcome fundamental resolution limits

## Key Insights

1. **Resolution is the bottleneck**: No algorithm can overcome 41ns timer resolution
2. **Distance = Time × Speed of Light**: 41ns = 12.3 meters
3. **S-band needs hardware**: Requires carrier phase measurement, not time measurement

## Comparison to Simulation

| Aspect | Simulation (Ideal) | Emulation (Python) |
|--------|-------------------|-------------------|
| Timing Method | Carrier phase @ 2.4GHz | Computer clock |
| Resolution | 0.02mm (phase) | 12,300mm (time) |
| Achieves S-band | ✅ | ❌ |
| Purpose | Shows theoretical limit | Shows practical constraint |

## Running Experiments

### 1. Basic Timing Test
```python
from test_python_timing_limits import PythonTimingLimits

tester = PythonTimingLimits()
tester.test_timer_resolution()
tester.analyze_implications()
```

### 2. Network Simulation
```python
from time_sync.twtt import TWTTNetwork

network = TWTTNetwork(n_nodes=10)
network.synchronize()
print(f"Sync accuracy: {network.get_sync_error()}mm")
```

## Limitations

This emulation demonstrates that:
1. **Python/software timing is insufficient** for S-band requirements
2. **Hardware timestamps are necessary** for sub-meter ranging
3. **Carrier phase measurement** (not time) is the solution

## Next Steps

1. **See simulation/** for theoretical performance with carrier phase
2. **See hardware/** for interfacing with real RF hardware
3. **Consider FPGA/ASIC** for deterministic timing

## References

- Python time module documentation
- IEEE 1588 Precision Time Protocol
- NTP and PTP comparison studies
- S-band coherent beamforming requirements