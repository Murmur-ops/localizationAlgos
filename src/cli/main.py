#!/usr/bin/env python3
"""
Main CLI entry point for Decentralized Localization System
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_simulate(args):
    """Run carrier phase synchronization simulation"""
    from simulation.src.run_phase_sync_simulation import CarrierPhaseSimulation
    
    print("Running carrier phase synchronization simulation...")
    sim = CarrierPhaseSimulation(args.config)
    sim.run()


def cmd_emulate(args):
    """Run timing emulation with Python constraints"""
    if args.test == "timing":
        from emulation.src.test_python_timing_limits import main as test_timing
        print("Testing Python timing limitations...")
        test_timing()
    elif args.test == "twtt":
        from emulation.src.time_sync.twtt import main as test_twtt
        print("Running Two-Way Time Transfer emulation...")
        test_twtt()
    else:
        print(f"Unknown test: {args.test}")
        sys.exit(1)


def cmd_benchmark(args):
    """Run performance benchmarks"""
    print("Running benchmarks...")
    
    results = {}
    
    # Run simulation benchmark
    if args.component in ["all", "simulation"]:
        print("\n1. Simulation (Carrier Phase)...")
        from simulation.src.run_phase_sync_simulation import CarrierPhaseSimulation
        config_path = Path(__file__).parent.parent / "simulation/config/phase_sync_sim.yaml"
        sim = CarrierPhaseSimulation(str(config_path))
        # Run quietly
        sim.config['output']['verbose'] = False
        sim.config['visualization']['show_plots'] = False
        sim.run()
        
    # Run emulation benchmark
    if args.component in ["all", "emulation"]:
        print("\n2. Emulation (Python Timing)...")
        from emulation.src.test_python_timing_limits import PythonTimingLimits
        tester = PythonTimingLimits()
        tester.test_timer_resolution(num_samples=1000)
    
    print("\nBenchmark complete!")


def cmd_test(args):
    """Run test suite"""
    import pytest
    
    test_path = Path(__file__).parent.parent.parent / "tests"
    
    if args.component:
        test_file = test_path / f"test_{args.component}.py"
        if test_file.exists():
            sys.exit(pytest.main([str(test_file), "-v"]))
        else:
            print(f"Test file not found: {test_file}")
            sys.exit(1)
    else:
        sys.exit(pytest.main([str(test_path), "-v"]))


def cmd_visualize(args):
    """Generate visualizations from results"""
    print(f"Generating visualization for: {args.result_file}")
    
    from core.visualization.network_plots import NetworkVisualizer
    import json
    
    with open(args.result_file, 'r') as f:
        data = json.load(f)
    
    viz = NetworkVisualizer(style=args.style)
    
    if args.plot_type == "network":
        viz.plot_network_scenario(
            data.get('true_positions', {}),
            data.get('estimated_positions', {}),
            data.get('anchor_positions', []),
            data.get('network_scale', 10.0),
            title=args.title or "Network Localization",
            save_path=args.output
        )
    elif args.plot_type == "convergence":
        viz.plot_convergence(
            data.get('iterations', []),
            data.get('rmse_history', []),
            data.get('objective_history', []),
            title=args.title or "Algorithm Convergence",
            save_path=args.output
        )
    else:
        print(f"Unknown plot type: {args.plot_type}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Decentralized Localization System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default config
  %(prog)s simulate
  
  # Run simulation with custom config
  %(prog)s simulate --config my_config.yaml
  
  # Test Python timing limitations
  %(prog)s emulate --test timing
  
  # Run all benchmarks
  %(prog)s benchmark --component all
  
  # Run tests
  %(prog)s test
  
  # Generate visualization
  %(prog)s visualize results.json --plot-type network --output network.png
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Decentralized Localization v1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulate command
    parser_sim = subparsers.add_parser(
        "simulate",
        help="Run carrier phase synchronization simulation"
    )
    parser_sim.add_argument(
        "--config",
        default="src/simulation/config/phase_sync_sim.yaml",
        help="Path to simulation config file"
    )
    parser_sim.set_defaults(func=cmd_simulate)
    
    # Emulate command
    parser_emu = subparsers.add_parser(
        "emulate",
        help="Run timing emulation with Python constraints"
    )
    parser_emu.add_argument(
        "--test",
        choices=["timing", "twtt", "frequency", "consensus"],
        default="timing",
        help="Which emulation test to run"
    )
    parser_emu.set_defaults(func=cmd_emulate)
    
    # Benchmark command
    parser_bench = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks"
    )
    parser_bench.add_argument(
        "--component",
        choices=["all", "simulation", "emulation", "algorithms"],
        default="all",
        help="Which component to benchmark"
    )
    parser_bench.set_defaults(func=cmd_benchmark)
    
    # Test command
    parser_test = subparsers.add_parser(
        "test",
        help="Run test suite"
    )
    parser_test.add_argument(
        "component",
        nargs="?",
        help="Specific component to test (optional)"
    )
    parser_test.set_defaults(func=cmd_test)
    
    # Visualize command
    parser_viz = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from results"
    )
    parser_viz.add_argument(
        "result_file",
        help="Path to JSON result file"
    )
    parser_viz.add_argument(
        "--plot-type",
        choices=["network", "convergence", "comparison"],
        default="network",
        help="Type of plot to generate"
    )
    parser_viz.add_argument(
        "--output",
        default="output.png",
        help="Output file path"
    )
    parser_viz.add_argument(
        "--style",
        choices=["publication", "presentation", "default"],
        default="publication",
        help="Plot style"
    )
    parser_viz.add_argument(
        "--title",
        help="Plot title (optional)"
    )
    parser_viz.set_defaults(func=cmd_visualize)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()