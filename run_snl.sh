#!/bin/bash
# Run script for Decentralized Sensor Network Localization experiments
# This script manages MPI execution of the SNL algorithms

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Decentralized Sensor Network Localization Runner${NC}"
echo "================================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}Error: Virtual environment not found. Run ./setup.sh first${NC}"
        exit 1
    fi
fi

# Check MPI installation
if ! command -v mpirun &> /dev/null; then
    echo -e "${RED}Error: MPI not found. Please install OpenMPI or MPICH${NC}"
    exit 1
fi

# Default parameters
N_PROCESSES=30
EXPERIMENT_TYPE="comparison"
N_EXPERIMENTS=50
OUTPUT_DIR="results"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--processes)
            N_PROCESSES="$2"
            shift 2
            ;;
        -e|--experiments)
            N_EXPERIMENTS="$2"
            shift 2
            ;;
        -t|--type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [options]

Options:
    -n, --processes NUM      Number of MPI processes (default: 30)
    -e, --experiments NUM    Number of experiments to run (default: 50)
    -t, --type TYPE         Experiment type: comparison, parameter, early, single (default: comparison)
    -o, --output DIR        Output directory (default: results)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Experiment types:
    comparison    - Compare MPS vs ADMM algorithms
    parameter     - Parameter sensitivity study
    early         - Early termination analysis (300 experiments)
    single        - Single experiment with visualization
    
Examples:
    $0                      # Run default comparison with 30 processes
    $0 -n 10 -e 20         # Run with 10 processes, 20 experiments
    $0 -t parameter        # Run parameter study
    $0 -t early -n 30      # Run early termination analysis
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Set log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/logs/run_${EXPERIMENT_TYPE}_${TIMESTAMP}.log"

echo "Configuration:"
echo "  MPI Processes: $N_PROCESSES"
echo "  Experiment Type: $EXPERIMENT_TYPE"
echo "  Number of Experiments: $N_EXPERIMENTS"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Log File: $LOG_FILE"
echo ""

# Function to run experiments
run_experiment() {
    local script="$1"
    local args="$2"
    
    echo -e "${GREEN}Running: mpirun -np $N_PROCESSES python $script $args${NC}"
    
    if [ "$VERBOSE" = true ]; then
        mpirun -np $N_PROCESSES python "$script" $args 2>&1 | tee "$LOG_FILE"
    else
        mpirun -np $N_PROCESSES python "$script" $args > "$LOG_FILE" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment completed successfully${NC}"
    else
        echo -e "${RED}✗ Experiment failed. Check log file: $LOG_FILE${NC}"
        exit 1
    fi
}

# Main execution based on experiment type
case $EXPERIMENT_TYPE in
    comparison)
        echo "Running algorithm comparison experiments..."
        
        # Create temporary Python script for comparison
        cat > temp_comparison.py << EOF
import sys
sys.path.append('.')
from run_experiments import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    n_experiments=$N_EXPERIMENTS,
    save_dir="$OUTPUT_DIR"
)

runner = ExperimentRunner(config)
results = runner.run_batch_experiments()

if runner.rank == 0:
    analysis = runner.analyze_results(results)
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"MPS Mean Error: {analysis['mps_error']['mean']:.6f} ± {analysis['mps_error']['std']:.6f}")
    print(f"ADMM Mean Error: {analysis['admm_error']['mean']:.6f} ± {analysis['admm_error']['std']:.6f}")
    print(f"Mean Error Ratio (ADMM/MPS): {analysis['error_ratio']['mean']:.2f}")
    print("="*50)
EOF
        
        run_experiment "temp_comparison.py" ""
        rm temp_comparison.py
        ;;
        
    parameter)
        echo "Running parameter sensitivity study..."
        
        # Create temporary Python script for parameter study
        cat > temp_parameter.py << EOF
import sys
sys.path.append('.')
from run_experiments import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    n_experiments=5,  # 5 per parameter value
    n_sensors_list=[20, 30, 40],
    n_anchors_list=[4, 6, 8],
    noise_factors=[0.01, 0.05, 0.1],
    communication_ranges=[0.5, 0.7, 0.9],
    save_dir="$OUTPUT_DIR"
)

runner = ExperimentRunner(config)
results = runner.run_parameter_study()
EOF
        
        run_experiment "temp_parameter.py" ""
        rm temp_parameter.py
        ;;
        
    early)
        echo "Running early termination analysis (300 experiments)..."
        
        # Create temporary Python script for early termination
        cat > temp_early.py << EOF
import sys
sys.path.append('.')
from run_experiments import EarlyTerminationAnalysis

analyzer = EarlyTerminationAnalysis(save_dir="$OUTPUT_DIR")
results = analyzer.run_early_termination_study(n_experiments=300)
EOF
        
        run_experiment "temp_early.py" ""
        rm temp_early.py
        ;;
        
    single)
        echo "Running single experiment with visualization..."
        
        # Run single experiment
        run_experiment "snl_main.py" ""
        
        # Generate visualizations
        echo -e "${GREEN}Generating visualizations...${NC}"
        python visualize_results.py --results-dir "$OUTPUT_DIR" --figures-dir figures
        ;;
        
    *)
        echo -e "${RED}Unknown experiment type: $EXPERIMENT_TYPE${NC}"
        exit 1
        ;;
esac

# Post-processing
echo ""
echo -e "${GREEN}Experiment complete!${NC}"
echo "Results saved in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Generate visualizations if not single experiment
if [ "$EXPERIMENT_TYPE" != "single" ] && [ -f "visualize_results.py" ]; then
    echo ""
    read -p "Generate visualizations? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating visualizations..."
        python visualize_results.py --results-dir "$OUTPUT_DIR" --figures-dir figures
        echo -e "${GREEN}Visualizations saved in: figures/${NC}"
    fi
fi

# Summary statistics
echo ""
echo "Summary:"
echo "--------"
if [ -f "$LOG_FILE" ]; then
    # Extract key metrics from log
    grep -E "(MPS Error:|ADMM Error:|Error ratio:)" "$LOG_FILE" | tail -3 || true
fi

echo ""
echo -e "${GREEN}✓ All tasks completed successfully!${NC}"