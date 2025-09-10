# Distributed Sensor Network Localization Protocol

## Complete Real-World Protocol Sequence

This document describes the full distributed protocol for sensor network localization, from initial deployment through final position estimation. All operations are fully distributed with no central coordinator.

## Protocol Overview

The protocol consists of five main phases:
1. **Node Discovery** - Nodes find their neighbors
2. **Distance Measurement** - TWTT-based ranging between neighbors  
3. **Consensus Normalization** - Distributed agreement on scale factor
4. **Clock Synchronization** - Consensus-based time alignment
5. **Position Estimation** - MPS algorithm for localization

## Phase 1: Node Discovery (T+0 to T+10 seconds)

### 1.1 Initial Broadcast
Each node broadcasts a discovery beacon containing:
- Node ID
- Timestamp (local clock)
- Discovery sequence number

```
DISCOVERY_BEACON {
    node_id: uint32
    timestamp_ns: int64
    sequence: uint16
    tx_power_dbm: int8
}
```

### 1.2 Neighbor Registration
Nodes maintain a neighbor table based on received beacons:
- Store node ID, RSSI, last seen timestamp
- Consider neighbor "active" if beacon received within 5 seconds
- Build adjacency matrix from active neighbors

### 1.3 Topology Formation
- Each node determines its local connectivity
- Exchanges neighbor lists with immediate neighbors
- Builds 2-hop neighborhood awareness

## Phase 2: Distance Measurement (T+10 to T+30 seconds)

### 2.1 Two-Way Travel Time (TWTT) Protocol
For each neighbor pair (i,j), perform symmetric TWTT:

```
Node i → Node j: RANGE_REQUEST {
    sender_id: i
    target_id: j  
    tx_time_i: T1
    sequence: seq
}

Node j → Node i: RANGE_RESPONSE {
    sender_id: j
    target_id: i
    tx_time_j: T2
    rx_time_j: T2 - T1 + prop_delay
    original_seq: seq
}

Node i computes:
    round_trip = (T3 - T1) - (T2_response - T2_request)
    distance = round_trip * c / 2
```

### 2.2 Multiple Measurements
- Perform 10 TWTT exchanges per neighbor pair
- Use median to filter outliers
- Store variance for uncertainty estimation

### 2.3 Distance Table Construction
Each node i maintains:
```
distances[j] = measured_distance_to_j
variances[j] = measurement_variance_to_j
```

## Phase 3: Consensus Normalization (T+30 to T+40 seconds)

### 3.1 Local Maximum Discovery
Each node:
- Finds its maximum measured distance: `local_max = max(distances)`
- Initializes consensus value: `consensus_max = local_max`

### 3.2 Max-Consensus Protocol
Iterative consensus to find global maximum:

```
For iteration k = 1 to convergence:
    For each node i:
        1. Broadcast current consensus_max[i] to neighbors
        2. Receive consensus_max[j] from all neighbors j
        3. Update: consensus_max[i] = max(
            consensus_max[i],
            α * max(consensus_max[neighbors]) + (1-α) * consensus_max[i]
        )
    
    Check convergence: max_change < threshold
```

Typical parameters:
- α = 0.5 (mixing parameter)
- threshold = 0.001 * initial_max
- Max iterations = 100

### 3.3 Normalization Factor
Once converged:
- All nodes agree on `global_max_distance`
- Normalization factor: `scale = 1.0 / global_max_distance`
- Normalized distances: `d_norm[j] = distances[j] * scale`

## Phase 4: Clock Synchronization (T+40 to T+50 seconds)

### 4.1 Offset Estimation
Using TWTT measurements, estimate clock offsets:

```
For each neighbor j:
    offset[j] = (T2_rx - T1_tx - T3_rx + T2_tx) / 2
```

### 4.2 Consensus Clock Protocol
Average offsets through consensus:

```
For iteration k = 1 to convergence:
    For each node i:
        1. Exchange current clock_offset[i] with neighbors
        2. Compute: avg_neighbor_offset = mean(clock_offset[neighbors])
        3. Update: clock_offset[i] = β * avg_neighbor_offset + (1-β) * clock_offset[i]
```

Parameters:
- β = 0.3 (clock mixing parameter)
- Convergence threshold = 1 nanosecond

### 4.3 Synchronized Time
- Global time estimate: `t_global = t_local + clock_offset`
- Synchronization accuracy: typically < 10 nanoseconds
- Enables coherent carrier phase measurements

## Phase 5: Position Estimation (T+50 to T+120 seconds)

### 5.1 MPS Algorithm Initialization
Each node initializes:
```
X[i] = random_position_in_unit_square
S[i] = lifted_matrix(X[i])  # [1, x'; x, xx']
```

### 5.2 Distributed MPS Iterations
Main iteration loop (runs at each node):

```
For iteration t = 1 to max_iterations:
    # Step 1: Local proximal update
    S[i] = prox_psd(S[i] - α * gradient[i])
    
    # Step 2: Exchange with neighbors
    For each neighbor j in adjacency[i]:
        Send S[i] to j
        Receive S[j] from j
    
    # Step 3: Consensus averaging
    S_consensus[i] = (1-γ) * S[i] + γ * mean(S[neighbors])
    
    # Step 4: Extract position estimate
    X[i] = S_consensus[i][1:3, 0]  # Extract position from lifted matrix
    
    # Step 5: Check convergence
    If change < threshold:
        Break
```

Parameters:
- α = 10.0 (step size)
- γ = 0.999 (consensus weight)
- max_iterations = 1000

### 5.3 Anchor Integration
If anchors present:
- Anchors broadcast their known positions
- Non-anchor nodes incorporate as constraints
- Improves absolute positioning accuracy

## Protocol Timing Summary

| Phase | Duration | Start | End | Operations |
|-------|----------|-------|-----|------------|
| Discovery | 10s | T+0 | T+10 | Find neighbors, build topology |
| Ranging | 20s | T+10 | T+30 | TWTT measurements |
| Normalization | 10s | T+30 | T+40 | Consensus on max distance |
| Clock Sync | 10s | T+40 | T+50 | Time alignment |
| Localization | 70s | T+50 | T+120 | MPS position estimation |

**Total Time**: ~2 minutes from deployment to positions

## Message Complexity

Per node, per protocol phase:
- Discovery: O(Δ) messages, where Δ = average degree
- Ranging: O(Δ × R) messages, R = rounds per neighbor
- Normalization: O(Δ × I_n) messages, I_n = consensus iterations
- Clock Sync: O(Δ × I_c) messages, I_c = clock iterations  
- Localization: O(Δ × I_mps) messages, I_mps = MPS iterations

**Total**: O(Δ × (R + I_n + I_c + I_mps)) ≈ O(Δ × 1000) messages per node

## Failure Handling

### Node Failures
- Detected through missing heartbeats (> 5 seconds)
- Removed from adjacency matrix
- Consensus continues with remaining nodes

### Message Loss
- All critical messages use acknowledgments
- Retransmit after 100ms timeout
- Maximum 3 retries before marking link failed

### Partial Connectivity
- Protocol works with any connected graph
- Disconnected components localize independently
- Can merge when connectivity restored

## Performance Characteristics

### Accuracy
- Distance measurements: ±1cm (with good TWTT hardware)
- Clock synchronization: <10ns 
- Position estimates: 5-10% of network diameter (typical)
- With anchors: 1-3% of network diameter

### Scalability
- Tested up to 100 nodes
- Communication is neighbor-only (no flooding)
- Computation is O(n) per node
- Storage is O(Δ) per node

### Energy Efficiency
- Most energy in ranging phase (radio transmissions)
- Consensus uses low-power broadcasts
- Can duty-cycle after initial localization
- Periodic updates every 60 seconds

## Implementation Notes

### Hardware Requirements
- Radio: Sub-GHz or 2.4GHz with ranging capability
- Clock: 1ppm stability or better
- Processing: 32-bit MCU, 64KB RAM minimum
- Storage: 16KB for protocol state

### Software Architecture
```
Application Layer
    ├── Localization Manager
    ├── Protocol State Machine
    └── Result Publisher

Protocol Layer  
    ├── Discovery Protocol
    ├── TWTT Ranging
    ├── Consensus Engine
    └── MPS Solver

Hardware Abstraction
    ├── Radio Driver
    ├── Timer/Clock
    └── Storage
```

### Configuration Parameters
```python
# Tunable parameters for deployment
DISCOVERY_BEACON_INTERVAL = 1.0  # seconds
TWTT_ROUNDS_PER_NEIGHBOR = 10
CONSENSUS_MIXING_PARAMETER = 0.5
CONSENSUS_THRESHOLD = 0.001
CLOCK_SYNC_MIXING = 0.3
MPS_ALPHA = 10.0
MPS_GAMMA = 0.999
MPS_MAX_ITERATIONS = 1000
POSITION_UPDATE_INTERVAL = 60  # seconds
```

## Practical Deployment Example

### Scenario: Indoor Warehouse Tracking
- 50 sensor nodes on assets
- 6 anchor nodes at known positions
- Goal: Track asset locations to ±50cm

### Deployment Steps
1. **Install anchors** at surveyed positions
2. **Power on sensor nodes** on assets
3. **Wait 2 minutes** for initial localization
4. **Monitor positions** via gateway node
5. **Periodic updates** every 60 seconds

### Expected Performance
- Initial fix: 2 minutes
- Position accuracy: ±30cm with anchors
- Update rate: 1/minute
- Battery life: 6 months (CR2032)

## Verification and Testing

### Protocol Verification
```bash
# Test discovery phase
python -m tests.test_discovery --nodes 20 --timeout 10

# Test ranging accuracy  
python -m tests.test_twtt --pairs 50 --rounds 10

# Test consensus convergence
python -m tests.test_consensus --nodes 30 --iterations 100

# Full protocol test
python -m tests.test_full_protocol --nodes 30 --anchors 6
```

### Field Testing Checklist
- [ ] RF environment survey (interference, multipath)
- [ ] Anchor position survey (< 1cm accuracy)
- [ ] Node placement (ensure connectivity)
- [ ] Ranging calibration (antenna delays)
- [ ] Clock drift characterization
- [ ] Full protocol execution
- [ ] Position accuracy validation
- [ ] Long-term stability test (24 hours)

## References

1. Matrix-Parametrized Proximal Splitting (MPS): arXiv:2503.13403v1
2. TWTT ranging: IEEE 802.15.4a standard
3. Consensus protocols: Olfati-Saber et al., 2007
4. Distributed optimization: Boyd et al., 2011

---

*This protocol enables fully distributed, scalable sensor network localization without central coordination. All operations are peer-to-peer with local communication only.*