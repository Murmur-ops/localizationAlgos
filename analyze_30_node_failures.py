#!/usr/bin/env python3
"""
Analyze why certain nodes fail in 30-node localization
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.channel.propagation import RangingChannel, ChannelConfig, PropagationType
from src.localization.robust_solver import RobustLocalizer, MeasurementEdge
from demo_30_nodes_large import generate_measurements, smart_initialization, load_config


def analyze_connectivity(measurements: list, anchors: dict, unknowns: dict):
    """Analyze measurement graph connectivity"""

    # Build graph
    G = nx.Graph()

    # Add all nodes
    for aid in anchors:
        G.add_node(aid, node_type='anchor')
    for uid in unknowns:
        G.add_node(uid, node_type='unknown')

    # Add edges from measurements
    anchor_connections = {uid: [] for uid in unknowns}

    for m in measurements:
        G.add_edge(m.node_i, m.node_j, weight=1/m.variance, distance=m.distance)

        # Track anchor connections
        if m.node_i in anchors and m.node_j in unknowns:
            anchor_connections[m.node_j].append(m.node_i)
        elif m.node_j in anchors and m.node_i in unknowns:
            anchor_connections[m.node_i].append(m.node_j)

    # Analyze each unknown node
    node_analysis = {}

    for uid in unknowns:
        # Number of anchor connections
        n_anchors = len(anchor_connections[uid])

        # Shortest path to each anchor
        anchor_distances = {}
        for aid in anchors:
            if nx.has_path(G, uid, aid):
                path = nx.shortest_path(G, uid, aid)
                # Sum of measurement distances along path
                path_dist = 0
                for i in range(len(path)-1):
                    path_dist += G[path[i]][path[i+1]]['distance']
                anchor_distances[aid] = {
                    'hops': len(path) - 1,
                    'distance': path_dist
                }

        # Degree (total connections)
        degree = G.degree(uid)

        # Clustering coefficient (local connectivity)
        clustering = nx.clustering(G, uid)

        node_analysis[uid] = {
            'n_anchor_connections': n_anchors,
            'anchor_distances': anchor_distances,
            'degree': degree,
            'clustering': clustering,
            'position': unknowns[uid]
        }

    return G, node_analysis


def visualize_connectivity(G: nx.Graph, anchors: dict, unknowns: dict, results: dict):
    """Visualize the measurement graph and localization errors"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Prepare node colors based on error
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in anchors:
            node_colors.append('red')
            node_sizes.append(500)
        elif node in results:
            error = results[node]['error']
            if error < 1.0:
                node_colors.append('green')
            elif error < 5.0:
                node_colors.append('yellow')
            else:
                node_colors.append('orange')
            node_sizes.append(200)
        else:
            node_colors.append('gray')
            node_sizes.append(100)

    # Get positions for layout
    pos = {}
    for aid, apos in anchors.items():
        pos[aid] = apos
    for uid, upos in unknowns.items():
        pos[uid] = upos

    # 1. Physical connectivity graph
    ax = axes[0, 0]
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes,
            with_labels=True, ax=ax, edge_color='gray', alpha=0.5)
    ax.set_title('Measurement Graph Connectivity', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # 2. Anchor connectivity
    ax = axes[0, 1]
    anchor_conn_values = []
    positions_list = []

    for uid in unknowns:
        if uid in results:
            n_anchors = sum(1 for m in G.edges(uid)
                          if any(n in anchors for n in [m[0], m[1]]))
            anchor_conn_values.append(n_anchors)
            positions_list.append(unknowns[uid])

    positions_array = np.array(positions_list)
    sc = ax.scatter(positions_array[:, 0], positions_array[:, 1],
                   c=anchor_conn_values, cmap='coolwarm', s=100)

    # Add anchors
    anchor_pos = np.array(list(anchors.values()))
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=500, c='red', marker='^', edgecolors='black', linewidth=2)

    plt.colorbar(sc, ax=ax, label='Number of Anchor Connections')
    ax.set_title('Direct Anchor Connectivity', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)

    # 3. Error vs anchor connections
    ax = axes[1, 0]
    errors = []
    n_anchor_conns = []

    for uid, analysis in node_analysis.items():
        if uid in results:
            errors.append(results[uid]['error'])
            n_anchor_conns.append(analysis['n_anchor_connections'])

    ax.scatter(n_anchor_conns, errors, alpha=0.7, s=50)
    ax.set_xlabel('Number of Direct Anchor Connections', fontsize=12)
    ax.set_ylabel('Localization Error (m)', fontsize=12)
    ax.set_title('Error vs Anchor Connectivity', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. Error heatmap
    ax = axes[1, 1]

    # Create grid for heatmap
    x_grid = np.linspace(0, 50, 20)
    y_grid = np.linspace(0, 50, 20)
    error_grid = np.zeros((len(y_grid), len(x_grid)))

    for uid, result in results.items():
        true_pos = result['true']
        error = result['error']

        # Find nearest grid point
        i = np.argmin(np.abs(y_grid - true_pos[1]))
        j = np.argmin(np.abs(x_grid - true_pos[0]))
        error_grid[i, j] = error

    im = ax.imshow(error_grid, extent=[0, 50, 0, 50], origin='lower',
                   cmap='RdYlGn_r', aspect='equal', alpha=0.7)

    # Overlay anchor positions
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1],
              s=500, c='blue', marker='^', edgecolors='black', linewidth=2)

    plt.colorbar(im, ax=ax, label='Localization Error (m)')
    ax.set_title('Spatial Error Distribution', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('30-Node Localization Failure Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def run_with_initial_positions(config, measurements, anchors, unknowns,
                               initial_positions, label="Custom"):
    """Run localization with specific initial positions"""

    # Initialize solver
    solver = RobustLocalizer(
        dimension=2,
        huber_delta=config['solver']['huber_delta']
    )
    solver.max_iterations = 50
    solver.convergence_threshold = config['system']['convergence_threshold']

    # Filter measurements
    filtered = [m for m in measurements if m.node_i in unknowns or m.node_j in unknowns]

    # Remap IDs
    unknown_ids = sorted(unknowns.keys())
    id_mapping = {aid: aid for aid in anchors}
    for i, uid in enumerate(unknown_ids):
        id_mapping[uid] = len(anchors) + i

    remapped_measurements = []
    for m in filtered:
        remapped = MeasurementEdge(
            node_i=id_mapping[m.node_i],
            node_j=id_mapping[m.node_j],
            distance=m.distance,
            quality=m.quality,
            variance=m.variance
        )
        remapped_measurements.append(remapped)

    remapped_anchors = {id_mapping[aid]: anchors[aid] for aid in anchors}

    # Solve
    optimized_positions, info = solver.solve(
        initial_positions,
        remapped_measurements,
        remapped_anchors
    )

    # Extract results
    results = {}
    errors = []
    for i, uid in enumerate(unknown_ids):
        est_pos = optimized_positions[i*2:(i+1)*2]
        true_pos = unknowns[uid]
        error = np.linalg.norm(est_pos - true_pos)
        errors.append(error)
        results[uid] = {
            'true': true_pos,
            'estimated': est_pos,
            'error': error
        }

    rmse = np.sqrt(np.mean(np.array(errors)**2))
    print(f"\n{label} Initialization:")
    print(f"  RMSE: {rmse:.2f}m")
    print(f"  Median: {np.median(errors):.2f}m")
    print(f"  < 1m: {sum(e < 1 for e in errors)}/{len(errors)}")
    print(f"  > 10m: {sum(e > 10 for e in errors)}/{len(errors)}")

    return results, rmse


if __name__ == "__main__":
    print("="*70)
    print("ANALYZING 30-NODE LOCALIZATION FAILURES")
    print("="*70)

    # Load configuration and generate measurements
    config = load_config()
    measurements, anchors, unknowns, _ = generate_measurements(config)

    # Analyze connectivity
    G, node_analysis = analyze_connectivity(measurements, anchors, unknowns)

    print("\nConnectivity Analysis:")
    print("-" * 40)

    # Group nodes by anchor connectivity
    by_anchor_conn = {}
    for uid, analysis in node_analysis.items():
        n = analysis['n_anchor_connections']
        if n not in by_anchor_conn:
            by_anchor_conn[n] = []
        by_anchor_conn[n].append(uid)

    for n_anchors in sorted(by_anchor_conn.keys()):
        nodes = by_anchor_conn[n_anchors]
        print(f"  {n_anchors} anchor connections: {len(nodes)} nodes - {nodes[:5]}...")

    # Test different initialization strategies
    print("\n" + "="*70)
    print("TESTING INITIALIZATION STRATEGIES")
    print("="*70)

    # 1. Smart trilateration
    init_smart = smart_initialization(unknowns, measurements, anchors)
    results_smart, rmse_smart = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_smart, "Smart Trilateration"
    )

    # 2. Random initialization
    np.random.seed(42)
    init_random = np.random.randn(len(unknowns) * 2) * 10 + 25
    results_random, rmse_random = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_random, "Random"
    )

    # 3. Center initialization
    init_center = np.ones(len(unknowns) * 2) * 25
    results_center, rmse_center = run_with_initial_positions(
        config, measurements, anchors, unknowns, init_center, "All at Center"
    )

    # Visualize the best results
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    fig = visualize_connectivity(G, anchors, unknowns, results_smart)
    plt.savefig('30_node_failure_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved analysis to: 30_node_failure_analysis.png")

    # Print detailed failure analysis
    print("\n" + "="*70)
    print("FAILED NODE DETAILS (>10m error)")
    print("="*70)

    failed_nodes = [(uid, r['error']) for uid, r in results_smart.items() if r['error'] > 10]
    failed_nodes.sort(key=lambda x: x[1], reverse=True)

    for uid, error in failed_nodes[:5]:
        analysis = node_analysis[uid]
        print(f"\nNode {uid} - Error: {error:.1f}m")
        print(f"  Position: {analysis['position']}")
        print(f"  Anchor connections: {analysis['n_anchor_connections']}")
        print(f"  Total degree: {analysis['degree']}")
        print(f"  Anchor distances (hops):")
        for aid, info in analysis['anchor_distances'].items():
            print(f"    Anchor {aid}: {info['hops']} hops, {info['distance']:.1f}m path")

    plt.show()