# Code Walkthrough: Decentralized SNL Implementation

This document provides a detailed walkthrough of the codebase, explaining the key components and how they work together.

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Algorithm Flow                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Problem Definition (SNLProblem)                             │
│     └─> Network parameters, noise levels, algorithm settings    │
│                                                                 │
│  2. Network Generation                                          │
│     ├─> Random sensor positions                                 │
│     ├─> Anchor placement                                        │
│     └─> Distance measurements with noise                        │
│                                                                 │
│  3. Matrix Parameter Computation                                │
│     ├─> Distributed Sinkhorn-Knopp                            │
│     └─> L, Z, W matrix generation                              │
│                                                                 │
│  4. MPS Algorithm Execution                                     │
│     ├─> Block 1: Y-update (consensus)                         │
│     ├─> Block 2: X-update (localization)                      │
│     └─> Convergence checking                                   │
│                                                                 │
│  5. Results Analysis                                            │
│     └─> RMSE, convergence metrics, visualization               │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Core Components

### 1. Problem Definition (`snl_main.py`)

```python
@dataclass
class SNLProblem:
    """Defines a sensor network localization problem"""
    n_sensors: int = 30          # Total number of sensors to localize
    n_anchors: int = 6           # Number of anchors (known positions)
    d: int = 2                   # Dimension (2D or 3D localization)
    communication_range: float = 0.7  # Max distance for communication
    max_neighbors: int = 7       # Maximum neighbors per sensor
    noise_factor: float = 0.05   # Measurement noise (5% = 0.05)
    gamma: float = 0.999         # Relaxation parameter for MPS
    alpha_mps: float = 10.0      # Proximal operator parameter
    alpha_admm: float = 150.0    # ADMM penalty parameter
    tol: float = 1e-4           # Convergence tolerance
    seed: Optional[int] = None   # Random seed for reproducibility
```

**Key insights:**
- `gamma` close to 1 provides stability but slower convergence
- `alpha_mps` controls the strength of the proximal term
- `communication_range` determines network connectivity

### 2. Sensor Data Structure

```python
@dataclass
class SensorData:
    """Stores all data for a single sensor"""
    sensor_id: int
    position: np.ndarray         # Current position estimate [x, y]
    neighbors: List[int]         # IDs of neighboring sensors
    neighbor_distances: Dict[int, float]  # Measured distances
    anchor_neighbors: List[int]  # Which anchors are in range
    anchor_distances: Dict[int, float]    # Distances to anchors
    
    # Algorithm variables
    Y_k: np.ndarray = field(default_factory=lambda: np.zeros(2))
    X_k: np.ndarray = field(default_factory=lambda: np.zeros(2))
    U_k: np.ndarray = field(default_factory=lambda: np.zeros(2))  # ADMM
```

### 3. Matrix Operations

The algorithm uses three key matrices:

#### L Matrix (Laplacian-like)
```python
def compute_L_matrix(self):
    """
    L matrix properties:
    - Doubly stochastic (rows and columns sum to 1)
    - Encodes network topology
    - Used for consensus operations
    """
    # Distributed Sinkhorn-Knopp algorithm
    for iteration in range(max_iter):
        # Row normalization (local)
        for i in range(n):
            row_sum = sum(L[i, :])
            L[i, :] /= row_sum
        
        # Column normalization (requires communication)
        col_sums = self._gather_column_sums()
        for j in range(n):
            L[:, j] /= col_sums[j]
```

#### Z and W Matrices
```python
def compute_Z_W_from_L(self):
    """
    Z = 2I - L - L^T  (Consensus matrix)
    W = Z             (Convergence matrix, can be different)
    """
    I = np.eye(n)
    Z = 2 * I - L - L.T
    W = Z.copy()  # Can use different W for optimization
```

### 4. MPS Algorithm Implementation

The core MPS algorithm uses a 2-block structure:

```python
def run_mps_distributed(self, max_iter=1000):
    """Main MPS algorithm loop"""
    
    for k in range(max_iter):
        # Block 1: Y-update (Consensus step)
        for sensor_id in self.sensor_ids:
            # Apply L matrix multiplication
            v_Y = self._apply_L_multiplication(sensor_id, 'Y')
            
            # Proximal operator for indicator function
            Y_new = self._prox_indicator_psd(
                sensor_id, 
                self.X_k[sensor_id] - v_Y
            )
            
            # Relaxation
            self.Y_k[sensor_id] = (self.gamma * Y_new + 
                                  (1 - self.gamma) * self.X_k[sensor_id])
        
        # Block 2: X-update (Localization step)
        for sensor_id in self.sensor_ids:
            # Apply W matrix multiplication
            v_X = self._apply_W_multiplication(sensor_id, 'X')
            
            # Proximal operator for g_i (distance constraints)
            self.X_k[sensor_id] = self._prox_gi(
                sensor_id,
                self.Y_k[sensor_id] - v_X
            )
        
        # Check convergence
        if self._check_convergence():
            break
```

### 5. Proximal Operators

The algorithm uses two main proximal operators:

#### Proximal Operator for Indicator Function
```python
def _prox_indicator_psd(self, sensor_id, v):
    """
    Projects onto the positive semidefinite cone
    Ensures consensus constraints are satisfied
    """
    # Simplified version - full implementation uses SDP
    return np.clip(v, -bound, bound)
```

#### Proximal Operator for g_i
```python
def _prox_gi(self, sensor_id, v):
    """
    Enforces distance constraints to neighbors and anchors
    Solves: argmin_x { g_i(x) + (α/2)||x - v||² }
    """
    sensor = self.sensor_data[sensor_id]
    
    # ADMM sub-solver for complex constraints
    x = v.copy()
    for admm_iter in range(50):
        # Update based on distance measurements
        for neighbor_id, measured_dist in sensor.neighbor_distances.items():
            # Project onto distance constraint
            diff = x - neighbor_position
            current_dist = np.linalg.norm(diff)
            if current_dist > 0:
                # Soft constraint enforcement
                x += alpha * (measured_dist/current_dist - 1) * diff
    
    return x
```

### 6. Distributed Communication

The implementation handles distributed communication efficiently:

```python
class DistributedCommunicator:
    """Manages inter-sensor communication"""
    
    def broadcast_to_neighbors(self, sensor_id, data):
        """Send data to all neighbors of a sensor"""
        sensor = self.sensor_data[sensor_id]
        for neighbor_id in sensor.neighbors:
            if self._is_local(neighbor_id):
                # Direct memory access
                self.receive_buffer[neighbor_id].append(data)
            else:
                # MPI send
                dest_rank = self._get_rank(neighbor_id)
                self.comm.send(data, dest=dest_rank, tag=sensor_id)
    
    def gather_from_neighbors(self, sensor_id):
        """Collect data from all neighbors"""
        messages = []
        
        # Local neighbors
        messages.extend(self.receive_buffer[sensor_id])
        
        # Remote neighbors
        for source_rank in self.neighbor_ranks[sensor_id]:
            data = self.comm.recv(source=source_rank, tag=MPI.ANY_TAG)
            messages.append(data)
        
        return messages
```

### 7. Convergence Detection

The implementation uses sophisticated convergence checking:

```python
def _check_convergence(self):
    """Check if algorithm has converged"""
    
    # 1. Position change criterion
    max_change = max(np.linalg.norm(self.X_k[i] - self.X_prev[i]) 
                    for i in self.sensor_ids)
    
    if max_change < self.tol:
        return True
    
    # 2. Objective value plateau detection
    if len(self.objective_history) >= self.early_termination_window:
        recent = self.objective_history[-self.early_termination_window:]
        relative_change = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-10)
        
        if relative_change < self.early_termination_threshold:
            self.early_termination_triggered = True
            return True
    
    # 3. Constraint violation check
    constraint_violation = self._compute_constraint_violation()
    if constraint_violation < self.tol:
        return True
    
    return False
```

### 8. MPI Optimization

The MPI implementation includes several optimizations:

```python
class OptimizedMPISNL:
    """Production-ready MPI implementation"""
    
    def __init__(self):
        # Pre-allocate communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        
        # Use non-blocking communication
        self.send_requests = []
        self.recv_requests = []
    
    def _update_with_overlap(self):
        """Overlap computation with communication"""
        
        # Start non-blocking sends
        for neighbor_proc in self.neighbor_procs:
            req = self.comm.Isend(data, dest=neighbor_proc)
            self.send_requests.append(req)
        
        # Start non-blocking receives
        for neighbor_proc in self.neighbor_procs:
            req = self.comm.Irecv(buffer, source=neighbor_proc)
            self.recv_requests.append(req)
        
        # Do local computation while waiting
        self._compute_local_updates()
        
        # Complete communication
        MPI.Request.Waitall(self.send_requests + self.recv_requests)
        
        # Process received data
        self._process_received_data()
```

## 🔄 Algorithm Flow Example

Here's a complete example tracing through the algorithm:

```python
# Step 1: Initialize
problem = SNLProblem(n_sensors=10, n_anchors=3)
solver = DistributedSNL(problem)

# Step 2: Generate network
# - Creates random positions for sensors
# - Places anchors at known locations
# - Generates noisy distance measurements
solver.generate_network()

# Step 3: Compute matrices
# - Runs distributed Sinkhorn-Knopp
# - Generates L, Z, W matrices
solver.compute_matrix_parameters()

# Step 4: Run MPS iterations
for iteration in range(max_iter):
    # Block 1: Consensus update (Y)
    # Each sensor:
    # 1. Computes L * Y (distributed operation)
    # 2. Applies proximal operator
    # 3. Updates Y with relaxation
    
    # Block 2: Localization update (X)
    # Each sensor:
    # 1. Computes W * X (distributed operation)
    # 2. Solves local optimization with distance constraints
    # 3. Updates position estimate
    
    # Check convergence
    if converged:
        break

# Step 5: Analyze results
error = compute_rmse(estimated_positions, true_positions)
```

## 🎯 Key Design Decisions

1. **2-Block Structure**: Enables parallel updates without complex coordination
2. **Distributed Matrices**: Only store local blocks, reducing memory
3. **Sparse Operations**: Exploit network sparsity for efficiency
4. **Early Termination**: Detect convergence without global coordination
5. **Non-blocking MPI**: Overlap communication with computation
6. **Proximal Operators**: Handle constraints elegantly

## 🚀 Performance Optimizations

1. **Communication Minimization**
   - Only communicate with direct neighbors
   - Batch messages when possible
   - Use collective operations for global reductions

2. **Memory Efficiency**
   - Store only local matrix blocks
   - Use sparse representations
   - Pre-allocate buffers

3. **Computational Efficiency**
   - Cache matrix-vector products
   - Vectorize operations
   - Use BLAS when available

4. **Scalability**
   - O(neighbors) memory per sensor
   - O(neighbors) communication per iteration
   - Linear speedup with MPI processes

This architecture enables the algorithm to scale efficiently to thousands of sensors while maintaining near-optimal localization accuracy.