# Units Convention for FTL Simulation Framework

## Executive Summary

This document establishes the standard units used throughout the FTL simulation framework to ensure consistency and prevent unit conversion errors.

---

## 1. Core Physical Units

### 1.1 Distance and Position
- **Primary Unit**: **meters (m)**
- **Usage**: All distances, positions, and ranges
- **Examples**:
  - `x_m`, `y_m`: Position coordinates in meters
  - `range_meas_m`: Measured range in meters
  - `geometric_range`: Distance between nodes in meters

### 1.2 Time
- **Primary Unit**: **seconds (s)** for time intervals
- **Secondary Unit**: **nanoseconds (ns)** for clock biases
- **Usage**:
  - Time intervals: seconds
  - Clock bias: nanoseconds (for numerical stability)
  - ToA measurements: seconds
- **Examples**:
  - `delta_t`: Time elapsed in seconds
  - `bias_ns`: Clock bias in nanoseconds
  - `toa_s`: Time of arrival in seconds

### 1.3 Frequency
- **Primary Units**:
  - **Hertz (Hz)** for absolute frequencies
  - **parts per million (ppm)** for carrier frequency offset (CFO)
  - **parts per billion (ppb)** for clock drift
- **Usage**:
  - Carrier frequency: Hz
  - Bandwidth: Hz
  - CFO: ppm (dimensionless, 1e-6)
  - Clock drift: ppb (dimensionless, 1e-9)
- **Examples**:
  - `fc`: Carrier frequency in Hz (e.g., 6.5e9)
  - `bandwidth_hz`: Signal bandwidth in Hz (e.g., 499.2e6)
  - `cfo_ppm`: CFO in ppm
  - `drift_ppb`: Clock drift in ppb

---

## 2. Derived Units

### 2.1 Velocity
- **Unit**: **meters per second (m/s)**
- **Primary Use**: Speed of light constant
- **Value**: `c = 299,792,458 m/s`

### 2.2 Variance and Standard Deviation
- **Position Variance**: **m²** (square meters)
- **Position Std Dev**: **m** (meters)
- **Time Variance**: **s²** (seconds squared)
- **Time Std Dev**: **s** (seconds)
- **Clock Bias Variance**: **ns²** (nanoseconds squared)
- **Drift Variance**: **ppb²** (ppb squared)
- **CFO Variance**: **ppm²** (ppm squared)

### 2.3 Information and Weights
- **Information**: **1/variance** in appropriate units
- **Square Root Information**: **1/std** in appropriate units
- **Weight**: Same as information

---

## 3. State Vector Convention

The standard 5D state vector uses:
```python
state = [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
```

| Index | Parameter | Unit | Description |
|-------|-----------|------|-------------|
| 0 | x | meters | X position |
| 1 | y | meters | Y position |
| 2 | bias | nanoseconds | Clock time offset |
| 3 | drift | ppb | Clock frequency drift |
| 4 | cfo | ppm | Carrier frequency offset |

Optional 6th dimension:
| 5 | sco | ppm | Sample clock offset |

---

## 4. Measurement Units

### 4.1 Range Measurements
- **Unit**: **meters**
- **Conversion**: `range_m = toa_s * c`
- **Variance**: **m²**

### 4.2 Time Measurements
- **ToA**: **seconds**
- **TDOA**: **seconds**
- **Clock Bias in Measurements**: Convert ns to seconds for ranging
  - `clock_contribution_m = bias_ns * c * 1e-9`

### 4.3 Frequency Measurements
- **CFO Difference**: **ppm**
- **Doppler Shift**: **Hz**

---

## 5. Conversion Factors

### 5.1 Time Conversions
```python
# Nanoseconds to seconds
time_s = time_ns * 1e-9

# Seconds to nanoseconds
time_ns = time_s * 1e9

# Clock bias to range
range_m = bias_ns * c * 1e-9
```

### 5.2 Frequency Conversions
```python
# ppm to absolute frequency
freq_hz = freq_ppm * fc * 1e-6

# ppb to fractional frequency
fractional = drift_ppb * 1e-9

# Hz to ppm (given carrier frequency fc)
cfo_ppm = cfo_hz / fc * 1e6
```

### 5.3 Distance/Time Conversions
```python
# Time to distance (speed of light)
distance_m = time_s * c

# Distance to time
time_s = distance_m / c

# With clock bias
total_range_m = geometric_range_m + (bias_ns * c * 1e-9)
```

---

## 6. Numerical Considerations

### 6.1 Why These Units?

**Meters for distance**: Natural scale for indoor/outdoor positioning (1-100m typical)

**Nanoseconds for clock bias**:
- 1 ns = 30 cm ranging error
- Typical biases: 10-1000 ns
- Avoids numerical issues with seconds (would be 1e-9 to 1e-6)

**ppb for drift**:
- Typical crystal drift: 0.1-10 ppb
- Natural scale, avoids very small numbers

**ppm for CFO**:
- Typical CFO: 1-20 ppm
- Standard in RF systems

### 6.2 Numerical Stability Rules

1. **Avoid mixing scales**: Keep similar quantities in similar scales
2. **Use scaled units in optimization**: ns, ppb, ppm instead of seconds
3. **Convert at interfaces**: Only convert units when interfacing with external systems

---

## 7. Code Examples

### 7.1 Creating a State
```python
from ftl.factors_scaled import ScaledState

state = ScaledState(
    x_m=10.5,           # 10.5 meters
    y_m=20.3,           # 20.3 meters
    bias_ns=150.0,      # 150 nanoseconds
    drift_ppb=2.5,      # 2.5 ppb
    cfo_ppm=15.0        # 15 ppm
)
```

### 7.2 Computing Range with Clock
```python
def compute_range_with_clock(pi_m, pj_m, bi_ns, bj_ns):
    """
    Args:
        pi_m, pj_m: Positions in meters
        bi_ns, bj_ns: Clock biases in nanoseconds
    Returns:
        Total range in meters
    """
    geometric = np.linalg.norm(pi_m - pj_m)
    clock_m = (bj_ns - bi_ns) * 299792458.0 * 1e-9
    return geometric + clock_m
```

### 7.3 Converting Measurements
```python
# ToA to range
toa_s = 1.5e-6  # 1.5 microseconds
range_m = toa_s * 299792458.0  # 450 meters

# Range variance to ToA variance
range_var_m2 = 0.01  # 1 cm squared
toa_var_s2 = range_var_m2 / (299792458.0**2)
```

---

## 8. Unit Validation Checklist

When implementing new features:

- [ ] Positions in meters
- [ ] Clock bias in nanoseconds
- [ ] Clock drift in ppb
- [ ] CFO in ppm
- [ ] ToA measurements in seconds
- [ ] Range measurements in meters
- [ ] Bandwidth in Hz
- [ ] SNR as linear (not dB) in calculations
- [ ] Variance in squared units of measurement
- [ ] Comments specify units for all variables

---

## 9. Common Pitfalls

### 9.1 Mixing Time Units
❌ **Wrong**:
```python
clock_contribution = (bj_s - bi_s) * c  # Mixing seconds and nanoseconds
```

✅ **Correct**:
```python
clock_contribution = (bj_ns - bi_ns) * c * 1e-9  # Consistent units
```

### 9.2 Forgetting Conversions
❌ **Wrong**:
```python
variance_m2 = crlb_s2  # Forgot to convert time variance to range
```

✅ **Correct**:
```python
variance_m2 = crlb_s2 * c**2  # Convert time to range variance
```

### 9.3 Wrong Frequency Units
❌ **Wrong**:
```python
freq_offset = cfo_hz / fc  # Result is fractional, not ppm
```

✅ **Correct**:
```python
freq_offset_ppm = cfo_hz / fc * 1e6  # Convert to ppm
```

---

## 10. Reference Constants

```python
# Physical constants
SPEED_OF_LIGHT_M_S = 299792458.0  # meters per second
SPEED_OF_LIGHT_M_NS = 0.299792458  # meters per nanosecond

# Typical values for validation
TYPICAL_INDOOR_RANGE_M = 10.0      # 10 meters
TYPICAL_CLOCK_BIAS_NS = 100.0      # 100 nanoseconds
TYPICAL_DRIFT_PPB = 1.0            # 1 ppb
TYPICAL_CFO_PPM = 10.0             # 10 ppm
TYPICAL_BANDWIDTH_HZ = 500e6       # 500 MHz
TYPICAL_CARRIER_HZ = 6.5e9         # 6.5 GHz
```

---

## Summary

This convention ensures:
1. **Numerical stability** through appropriate scaling
2. **Consistency** across all modules
3. **Clarity** in code and documentation
4. **Correctness** in physical calculations

Always document units in:
- Variable names (suffix with `_m`, `_ns`, `_ppb`, etc.)
- Comments
- Function docstrings
- Test assertions

---

*Document Version: 1.0*
*Framework: FTL Consensus Simulation*
*Last Updated: 2024*