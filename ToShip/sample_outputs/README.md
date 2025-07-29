# Sample Outputs

When you run the scripts in this package, you'll generate the following types of visualizations:

## 1. **sensor_network_topology.png**
- Shows the sensor network layout with sensors (blue dots) and anchors (red triangles)
- Includes connectivity histogram showing neighbor distribution

## 2. **algorithm_convergence.png**
- Plots objective value and RMSE over iterations
- Compares MPS vs ADMM convergence rates
- Shows early termination points

## 3. **crlb_comparison.png**
- Compares algorithm performance to the Cramér-Rao Lower Bound
- Shows 80-85% efficiency across different noise levels
- Includes MPS efficiency summary box

## 4. **localization_results.png**
- Before/after visualization of sensor positions
- Shows initial estimates vs final MPS results
- Displays RMSE improvement metrics

## 5. **simple_example_results.png** (from simple_example.py)
- Side-by-side comparison of initial vs final positions
- Color-coded error visualization
- RMSE metrics in text boxes

## 6. **simple_example_convergence.png** (from simple_example.py)
- Log-scale plot of RMSE over iterations
- Shows algorithm convergence behavior

To generate these figures:
```bash
# Generate all analysis figures
python generate_figures.py

# Generate simple example figures
python simple_example.py

# Generate CRLB analysis
python crlb_analysis.py
```