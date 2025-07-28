# Development Guide

## Current Status

We have created a Git repository with the initial implementation on the `main` branch and a `full-implementation` branch for completing the paper reproduction.

### Repository Structure

- **main branch**: Initial working implementation with:
  - Basic MPS and ADMM algorithms
  - Distributed Sinkhorn-Knopp
  - OARS integration
  - Experiment framework

- **full-implementation branch**: For completing paper reproduction

## Full Implementation Tasks

### High Priority

1. **L Matrix Multiplication for 2-Block Structure**
   - Implement proper communication between first and second blocks
   - Add L matrix construction from Z matrix
   - Implement distributed matrix-vector multiplication

2. **Objective and Iteration Tracking**
   - Track objective value at each iteration
   - Store convergence history
   - Implement proper stopping criteria

3. **Early Termination**
   - Monitor objective value history
   - Implement 100-iteration window check
   - Store intermediate results for analysis

4. **Complete ADMM Implementation**
   - Add proper iteration counting
   - Track convergence metrics
   - Implement communication patterns

### Medium Priority

5. **Warm Start for Mobile Sensors**
   - Implement velocity-based prediction
   - Add previous solution initialization
   - Test with simulated sensor movement

6. **2-Block Communication Patterns**
   - Implement parallel execution of first block
   - Add synchronization between blocks
   - Optimize communication overhead

7. **Unit Tests**
   - Test Sinkhorn-Knopp convergence
   - Verify matrix parameter properties
   - Test proximal operators

### Low Priority

8. **Performance Benchmarks**
   - Compare with centralized methods
   - Measure communication overhead
   - Profile computation time

9. **Debugging Tools**
   - Add distributed logging
   - Implement convergence visualization
   - Add MPI debugging utilities

## Development Workflow

1. **Working on full-implementation branch**:
   ```bash
   git checkout full-implementation
   # Make changes
   git add -p  # Review changes
   git commit -m "Descriptive message"
   ```

2. **Testing changes**:
   ```bash
   # Unit tests
   pytest tests/

   # Small scale test
   mpirun -np 10 python snl_main.py

   # Full experiment
   ./run_snl.sh -n 30 -e 50
   ```

3. **Merging to main**:
   ```bash
   git checkout main
   git merge full-implementation
   git push origin main
   ```

## Key Implementation Areas

### 1. L Matrix Construction (snl_main.py)

```python
def construct_L_matrix(self, Z: np.ndarray) -> np.ndarray:
    """Construct lower triangular L from Z = 2I - L - L^T"""
    # Implementation needed
    pass
```

### 2. Objective Tracking (snl_main.py)

```python
def _compute_objective_with_constraints(self) -> Tuple[float, float]:
    """Compute objective value and constraint violation"""
    # Implementation needed
    pass
```

### 3. Early Termination (snl_main.py)

```python
def check_early_termination(self, objective_history: List[float], 
                          window: int = 100) -> bool:
    """Check if we should terminate early"""
    # Implementation needed
    pass
```

## Testing Strategy

1. **Unit Tests**: Core algorithm components
2. **Integration Tests**: Full algorithm runs
3. **Convergence Tests**: Verify paper results
4. **Scaling Tests**: Performance with different network sizes

## Next Steps

1. Start with implementing proper L matrix operations
2. Add objective tracking to both algorithms
3. Implement early termination logic
4. Run experiments to verify paper results