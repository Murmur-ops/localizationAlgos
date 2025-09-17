# Getting Started with FTL Localization

This is a minimal demo of Frequency-Time-Localization (FTL) showing distributed localization with realistic RF physics.

## Quick Start

```bash
# Run the simple demo
python demo_simple.py configs/simple.yaml

# Try different scenarios
python demo_simple.py configs/indoor.yaml   # High accuracy indoor
python demo_simple.py configs/outdoor.yaml  # Challenging outdoor
```

## What This Demo Shows

The demo implements a complete localization system with:
- **Realistic RF ranging** with bandwidth and SNR effects
- **Distributed optimization** using Levenberg-Marquardt
- **Automatic measurement generation** based on range and SNR
- **Visual results** showing true vs estimated positions

## Configuration Files

### `configs/simple.yaml`
Basic 10-node setup in 20×20m area. Expected RMSE: ~0.5m

### `configs/indoor.yaml`
Indoor scenario with UWB (500 MHz). Expected RMSE: <0.2m

### `configs/outdoor.yaml`
Large outdoor area with WiFi (20 MHz). Expected RMSE: 2-5m

## Key Parameters

The four fundamental parameters that affect accuracy:

1. **SNR** (`snr_db`): Signal quality, affects precision
2. **Bandwidth** (`bandwidth_hz`): Sets resolution floor (c/2B)
3. **Anchors**: More anchors = better geometry
4. **Area size**: Larger areas need more anchors

## Understanding Results

The demo outputs:
- **RMSE**: Root mean square error across all nodes
- **Visualization**: Network topology and localization errors
- **Per-node errors**: Individual node performance

Good performance indicators:
- RMSE < 1m for indoor
- RMSE < 5m for outdoor
- Convergence in <50 iterations

## Customization

Edit the YAML files to experiment with:
- Different area sizes
- Number of nodes and anchors
- Channel conditions (bandwidth, SNR)
- Ranging limits

## Algorithm

The demo uses simplified Levenberg-Marquardt optimization:
1. Generate ranging measurements with realistic noise
2. Initialize unknowns at area center
3. Iteratively minimize measurement residuals
4. Weight measurements by SNR quality

## Next Steps

For the full system with more features, see the parent directory:
- Spread spectrum signals
- Clock synchronization
- NLOS detection
- Distributed ADMM solver