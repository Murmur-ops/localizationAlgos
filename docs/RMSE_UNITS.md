# RMSE Units Documentation

## Quick Answer
**RMSE units are in meters** when using default configurations.

## Detailed Explanation

### Default Behavior
1. **Network positions** are generated in a normalized space [0, 1]
2. **Default scale** is 10.0 meters (10m x 10m area for 2D)
3. **RMSE** is calculated in the same units as the positions

### Unit System

| Component | Default Value | Unit | Notes |
|-----------|--------------|------|-------|
| Network scale | 10.0 | meters | Size of deployment area |
| Position coordinates | [0, 1] normalized | scaled to meters | Multiply by scale for real units |
| Distance measurements | - | meters | Computed from positions |
| RMSE output | - | meters | Same units as positions |
| Convergence tolerance | 0.0001 | meters | 0.1mm default precision |

### Examples

#### Small Network (10 sensors)
- Network area: 10m x 10m
- RMSE of 0.2 = 20 centimeters average error
- RMSE of 0.01 = 1 centimeter average error

#### Large Network (100 sensors)  
- Network area: 10m x 10m (dense) or can be scaled
- RMSE of 1.0 = 1 meter average error
- RMSE of 0.5 = 50 centimeters average error

#### S-band Precision
- Network area: typically 10m x 10m
- RMSE of 0.00014 = 0.14 millimeters (meets S-band requirement)
- Target: < 0.015 = less than 15 millimeters

### How to Change Units

#### Method 1: Scale the Network
```yaml
network:
  scale: 100.0  # 100m x 100m area
```
Now RMSE will still be in meters but relative to a larger area.

#### Method 2: Interpret Differently
If you want positions in kilometers:
- Set scale: 1.0 
- Interpret RMSE as kilometers instead of meters

### Special Cases

#### Carrier Phase Synchronization
When using carrier phase (simulation mode):
- Positions: still in meters
- Phase measurements: radians
- Ranging accuracy: millimeters
- Final RMSE: millimeters (converted from phase)

#### Time-of-Flight
When using time-based ranging:
- Positions: meters
- Time measurements: nanoseconds
- Distance = time × speed_of_light
- RMSE: meters

### RMSE Calculation

The RMSE is calculated as:
```python
errors = []
for each sensor:
    error = ||estimated_position - true_position||  # Euclidean distance
    errors.append(error²)
RMSE = sqrt(mean(errors))
```

Units are preserved throughout:
- If positions are in meters → RMSE is in meters
- If positions are in millimeters → RMSE is in millimeters

### Typical RMSE Values

| Scenario | Expected RMSE | Quality |
|----------|--------------|---------|
| Ideal conditions (2% noise) | 0.1-0.3 m | Excellent |
| Moderate noise (5%) | 0.3-0.8 m | Good |
| High noise (10%) | 0.8-2.0 m | Acceptable |
| S-band carrier phase | 0.0001-0.001 m | Millimeter precision |
| Python timing emulation | 600-1000 m | Poor (timing limits) |

### Converting Units

To convert RMSE to different units:
- **Meters to centimeters**: multiply by 100
- **Meters to millimeters**: multiply by 1000
- **Meters to feet**: multiply by 3.28084
- **Meters to inches**: multiply by 39.3701

### In Configuration Files

When you see in YAML configs:
```yaml
algorithm:
  tolerance: 0.0001  # This is 0.1mm when scale=10m
```

The tolerance is in the same units as positions (meters by default).