#!/usr/bin/env python3
"""
Diagnose network connectivity issues
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def analyze_network_connectivity():
    """Analyze the network graph connectivity"""

    # Same setup as experiment
    anchor_positions = {
        0: np.array([0.0, 0.0]),
        1: np.array([50.0, 0.0]),
        2: np.array([50.0, 50.0]),
        3: np.array([0.0, 50.0])
    }

    # Generate same 30 unknown nodes
    np.random.seed(42)
    unknown_positions = {}
    for i in range(4, 34):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(5, 45)
        unknown_positions[i] = np.array([x, y])

    max_range = 20.0  # Communication range

    # Build connectivity graph
    G = nx.Graph()

    # Add all nodes
    for id in range(34):
        G.add_node(id)

    # Add edges based on communication range
    # Anchor to unknown connections
    n_anchor_connections = 0
    for anchor_id, anchor_pos in anchor_positions.items():
        for unknown_id, unknown_pos in unknown_positions.items():
            dist = np.linalg.norm(unknown_pos - anchor_pos)
            if dist <= max_range:
                G.add_edge(anchor_id, unknown_id)
                n_anchor_connections += 1

    # Unknown to unknown connections
    n_peer_connections = 0
    for i in range(4, 34):
        for j in range(i+1, 34):
            dist = np.linalg.norm(unknown_positions[i] - unknown_positions[j])
            if dist <= max_range:
                G.add_edge(i, j)
                n_peer_connections += 1

    print("=" * 70)
    print("NETWORK CONNECTIVITY ANALYSIS")
    print("=" * 70)

    print(f"\nBASIC STATISTICS:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Anchor-unknown edges: {n_anchor_connections}")
    print(f"  Unknown-unknown edges: {n_peer_connections}")

    # Check connectivity
    print(f"\nCONNECTIVITY:")
    print(f"  Is connected: {nx.is_connected(G)}")
    print(f"  Number of components: {nx.number_connected_components(G)}")

    # Analyze components
    components = list(nx.connected_components(G))
    for i, comp in enumerate(components):
        anchors_in_comp = [n for n in comp if n < 4]
        print(f"  Component {i}: {len(comp)} nodes, {len(anchors_in_comp)} anchors")

    # Check nodes with direct anchor connections
    nodes_with_anchor = []
    for unknown_id in range(4, 34):
        anchor_neighbors = [n for n in G.neighbors(unknown_id) if n < 4]
        if anchor_neighbors:
            nodes_with_anchor.append(unknown_id)

    print(f"\nANCHOR CONNECTIVITY:")
    print(f"  Nodes with direct anchor connection: {len(nodes_with_anchor)}/30")
    print(f"  Nodes without anchor connection: {30 - len(nodes_with_anchor)}/30")

    # Analyze degree distribution
    degrees = dict(G.degree())
    unknown_degrees = [degrees[i] for i in range(4, 34)]

    print(f"\nDEGREE STATISTICS (connections per unknown node):")
    print(f"  Mean: {np.mean(unknown_degrees):.1f}")
    print(f"  Min: {np.min(unknown_degrees)}")
    print(f"  Max: {np.max(unknown_degrees)}")

    # Check for isolated or weakly connected nodes
    weak_nodes = [i for i in range(4, 34) if degrees[i] < 3]
    print(f"  Nodes with < 3 connections: {len(weak_nodes)}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Network graph
    ax = axes[0]
    pos = {}
    for id, p in anchor_positions.items():
        pos[id] = p
    for id, p in unknown_positions.items():
        pos[id] = p

    # Draw network
    nx.draw_networkx_nodes(G, pos, nodelist=list(range(4)),
                          node_color='red', node_size=300, node_shape='^', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(range(4, 34)),
                          node_color='lightblue', node_size=100, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(4)},
                           font_size=8, ax=ax)

    ax.set_xlim(-5, 55)
    ax.set_ylim(-5, 55)
    ax.set_title(f'Network Graph (Range: {max_range}m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Degree histogram
    ax = axes[1]
    ax.hist(unknown_degrees, bins=range(0, max(unknown_degrees)+2),
           edgecolor='black', alpha=0.7)
    ax.set_xlabel('Node Degree (number of connections)')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Connectivity Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('network_connectivity_analysis.png', dpi=150)
    print("\nPlot saved to: network_connectivity_analysis.png")

    # DIAGNOSIS
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")

    if n_anchor_connections < 20:
        print("✗ CRITICAL: Too few anchor connections!")
        print(f"  Only {n_anchor_connections} anchor-unknown connections")
        print("  Most nodes rely on multi-hop paths to anchors")
        print("  This causes error accumulation")

    if not nx.is_connected(G):
        print("✗ CRITICAL: Network is disconnected!")
        print("  Some nodes cannot reach anchors at all")

    if len(nodes_with_anchor) < 15:
        print("✗ PROBLEM: Most nodes lack direct anchor connections")
        print(f"  Only {len(nodes_with_anchor)}/30 have direct anchor links")

    print("\nRECOMMENDATIONS:")
    print("1. Increase communication range to 30m")
    print("2. Add more anchors (8-12 for 30 nodes)")
    print("3. Use hierarchical approach with well-connected nodes as pseudo-anchors")
    print("4. Ensure each unknown has ≥2 anchor connections")

    return G, anchor_positions, unknown_positions


if __name__ == "__main__":
    analyze_network_connectivity()