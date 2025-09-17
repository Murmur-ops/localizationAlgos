# FTL Localization - Clean Demo

A complete demonstration of Frequency-Time-Localization (FTL) for distributed positioning using spread spectrum signals.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Simple ranging demo
python demo_simple.py configs/simple.yaml

# Full FTL with joint frequency-time-location estimation
python ftl_demo.py configs/ftl.yaml
```

## Two Demo Modes

### 1. Simple Demo (`demo_simple.py`)
Basic ranging and localization with realistic noise models.

### 2. FTL Demo (`ftl_demo.py`)
Complete FTL system with:
- Gold code spread spectrum signals
- Frequency synchronization (PLL)
- Time synchronization (Kalman filter)
- Joint estimation of frequency, time, and location

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