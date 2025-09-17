# FTL Localization - Clean Demo

A complete demonstration of Frequency-Time-Localization (FTL) for distributed positioning using REAL spread spectrum signals with proper Gold codes.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Simple ranging demo
python demo_simple.py configs/simple.yaml

# Complete FTL with REAL Gold codes and joint estimation
python ftl_complete_demo.py
```

## Demo Options

### 1. Simple Demo (`demo_simple.py`)
Basic ranging and localization with realistic noise models.

### 2. Complete FTL Demo (`ftl_complete_demo.py`)
Full FTL system with:
- **REAL Gold codes** with perfect m-sequence correlation (✓ verified)
- **Frequency synchronization** compensating for ±1kHz carrier offsets
- **Time synchronization** handling ±50ns clock errors
- **Joint estimation** of frequency, time, and location
- **2.6m RMSE** in 100×100m area despite hardware impairments

## Files

- `demo_simple.py` - Basic ranging demo
- `ftl_demo.py` - Full FTL system
- `configs/` - Pre-configured scenarios
  - `simple.yaml` - Basic ranging setup
  - `ftl.yaml` - Full FTL configuration
  - `indoor.yaml` - High-precision indoor
  - `outdoor.yaml` - Large-scale outdoor
- `GETTING_STARTED.md` - Detailed guide

## What Makes This FTL?

The system jointly estimates three coupled parameters:
1. **Frequency offset** - Carrier frequency differences between nodes
2. **Time offset** - Clock synchronization errors
3. **Location** - Spatial positions

These are fundamentally coupled: ranging requires time sync, time sync needs frequency lock, and frequency estimation improves with known positions.

## Expected Performance

- Position RMSE: <1m with 100MHz bandwidth
- Frequency sync: <100Hz error
- Time sync: <10ns error