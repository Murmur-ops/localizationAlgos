#!/bin/bash
# Run OARS matrix method comparison experiments

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Running OARS Matrix Method Comparison${NC}"
echo "====================================="

# Check environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate || source activate_env.sh
fi

# Default parameters
N_PROCESSES=${1:-10}
N_EXPERIMENTS=${2:-10}

echo "Configuration:"
echo "  MPI Processes: $N_PROCESSES"
echo "  Experiments per method: $N_EXPERIMENTS"
echo ""

# Create results directory
mkdir -p results_oars
mkdir -p results_oars/logs

# Run OARS comparison
echo -e "${GREEN}Comparing matrix generation methods...${NC}"
mpirun -np $N_PROCESSES python run_experiments_oars.py 2>&1 | tee results_oars/logs/oars_comparison_$(date +%Y%m%d_%H%M%S).log

echo ""
echo -e "${GREEN}✓ OARS comparison complete!${NC}"
echo "Results saved in: results_oars/"

# Check if we should generate plots
if [ -f "results_oars/matrix_scaling.png" ]; then
    echo "Scaling plot saved: results_oars/matrix_scaling.png"
fi

# Display summary if available
if [ -f "results_oars/oars_analysis.json" ]; then
    echo ""
    echo "Summary of results:"
    python -c "
import json
with open('results_oars/oars_analysis.json', 'r') as f:
    data = json.load(f)
    for method, stats in data.items():
        print(f\"{method}: Error = {stats['mps_error']['mean']:.6f} ± {stats['mps_error']['std']:.6f}\")
"
fi