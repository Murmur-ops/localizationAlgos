"""
Unit tests for geometry module
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftl.geometry import (
    NodeGeometry, PlacementType,
    place_grid_nodes, place_poisson_nodes, place_anchors,
    get_connectivity_matrix, check_graph_rigidity
)


class TestNodeGeometry(unittest.TestCase):
    """Test NodeGeometry class"""

    def test_node_creation(self):
        """Test basic node creation"""
        node = NodeGeometry(node_id=1, x=10.0, y=20.0, is_anchor=False)
        self.assertEqual(node.node_id, 1)
        self.assertEqual(node.x, 10.0)
        self.assertEqual(node.y, 20.0)
        self.assertFalse(node.is_anchor)

    def test_position_property(self):
        """Test position property returns correct numpy array"""
        node = NodeGeometry(node_id=0, x=5.0, y=7.0)
        pos = node.position
        np.testing.assert_array_equal(pos, np.array([5.0, 7.0]))

    def test_distance_calculation(self):
        """Test distance calculation between nodes"""
        node1 = NodeGeometry(node_id=0, x=0, y=0)
        node2 = NodeGeometry(node_id=1, x=3, y=4)
        dist = node1.distance_to(node2)
        self.assertAlmostEqual(dist, 5.0)  # 3-4-5 triangle


class TestGridPlacement(unittest.TestCase):
    """Test grid placement functions"""

    def test_grid_placement_count(self):
        """Test that grid placement creates N×N nodes"""
        for n in [3, 5, 10]:
            nodes = place_grid_nodes(n=n, area_size=100.0)
            self.assertEqual(len(nodes), n * n)

    def test_grid_placement_bounds(self):
        """Test that all nodes are within area bounds"""
        area_size = 100.0
        nodes = place_grid_nodes(n=10, area_size=area_size)
        for node in nodes:
            self.assertGreaterEqual(node.x, 0)
            self.assertLessEqual(node.x, area_size)
            self.assertGreaterEqual(node.y, 0)
            self.assertLessEqual(node.y, area_size)

    def test_grid_placement_spacing(self):
        """Test regular spacing without jitter"""
        n = 3
        area_size = 90.0
        nodes = place_grid_nodes(n=n, area_size=area_size, jitter_std=0)

        # Expected spacing is area_size / (n + 1)
        expected_spacing = area_size / (n + 1)

        # Check first row
        for i in range(n):
            expected_x = expected_spacing * (i + 1)
            self.assertAlmostEqual(nodes[i * n].x, expected_x, places=5)

    def test_grid_placement_jitter(self):
        """Test that jitter adds randomness"""
        nodes1 = place_grid_nodes(n=5, area_size=100, jitter_std=1.0, seed=42)
        nodes2 = place_grid_nodes(n=5, area_size=100, jitter_std=1.0, seed=43)

        # Positions should be different with different seeds
        positions_same = all(
            n1.x == n2.x and n1.y == n2.y
            for n1, n2 in zip(nodes1, nodes2)
        )
        self.assertFalse(positions_same)

    def test_grid_placement_reproducible(self):
        """Test that same seed gives same results"""
        nodes1 = place_grid_nodes(n=5, area_size=100, jitter_std=1.0, seed=42)
        nodes2 = place_grid_nodes(n=5, area_size=100, jitter_std=1.0, seed=42)

        # Positions should be identical with same seed
        for n1, n2 in zip(nodes1, nodes2):
            self.assertEqual(n1.x, n2.x)
            self.assertEqual(n1.y, n2.y)


class TestPoissonPlacement(unittest.TestCase):
    """Test Poisson disk sampling placement"""

    def test_poisson_placement_count(self):
        """Test that Poisson placement creates requested number of nodes"""
        n_total = 20
        nodes = place_poisson_nodes(n_total=n_total, area_size=100, min_distance=5)
        self.assertEqual(len(nodes), n_total)

    def test_poisson_minimum_distance(self):
        """Test minimum distance constraint"""
        min_dist = 10.0
        nodes = place_poisson_nodes(
            n_total=10,
            area_size=100,
            min_distance=min_dist,
            seed=42
        )

        # Check all pairwise distances
        violated = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = nodes[i].distance_to(nodes[j])
                if dist < min_dist * 0.9:  # Allow 10% tolerance
                    violated = True
                    break

        # Some violations are acceptable if placement is difficult
        # But shouldn't be too many
        self.assertFalse(violated or len(nodes) < 5)


class TestAnchorPlacement(unittest.TestCase):
    """Test anchor placement strategies"""

    def test_corner_placement(self):
        """Test corner anchor placement"""
        nodes = place_grid_nodes(n=5, area_size=100)
        nodes_with_anchors = place_anchors(
            nodes, m=4, area_size=100,
            placement=PlacementType.CORNERS
        )

        # Should have original nodes + 4 anchors
        self.assertEqual(len(nodes_with_anchors), len(nodes) + 4)

        # Check anchor positions are at corners
        anchors = [n for n in nodes_with_anchors if n.is_anchor]
        self.assertEqual(len(anchors), 4)

        # Check corners (with some tolerance)
        corner_positions = [(0, 0), (100, 0), (100, 100), (0, 100)]
        for anchor in anchors:
            is_corner = any(
                abs(anchor.x - cx) < 1 and abs(anchor.y - cy) < 1
                for cx, cy in corner_positions
            )
            self.assertTrue(is_corner)

    def test_perimeter_placement(self):
        """Test perimeter anchor placement"""
        nodes = place_grid_nodes(n=5, area_size=100)
        m = 8
        nodes_with_anchors = place_anchors(
            nodes, m=m, area_size=100,
            placement=PlacementType.PERIMETER
        )

        # Check correct number of anchors
        anchors = [n for n in nodes_with_anchors if n.is_anchor]
        self.assertEqual(len(anchors), m)

        # Check all anchors are on perimeter
        for anchor in anchors:
            on_edge = (
                abs(anchor.x) < 1 or abs(anchor.x - 100) < 1 or
                abs(anchor.y) < 1 or abs(anchor.y - 100) < 1
            )
            self.assertTrue(on_edge)

    def test_corner_placement_insufficient_anchors(self):
        """Test that corner placement requires at least 4 anchors"""
        nodes = place_grid_nodes(n=5, area_size=100)
        with self.assertRaises(ValueError):
            place_anchors(nodes, m=3, area_size=100,
                         placement=PlacementType.CORNERS)


class TestConnectivity(unittest.TestCase):
    """Test connectivity matrix generation"""

    def test_connectivity_matrix_shape(self):
        """Test connectivity matrix has correct shape"""
        nodes = place_grid_nodes(n=3, area_size=30)
        conn = get_connectivity_matrix(nodes, comm_radius=15)
        self.assertEqual(conn.shape, (9, 9))

    def test_connectivity_symmetry(self):
        """Test connectivity matrix is symmetric"""
        nodes = place_grid_nodes(n=4, area_size=40)
        conn = get_connectivity_matrix(nodes, comm_radius=20)
        np.testing.assert_array_equal(conn, conn.T)

    def test_connectivity_no_self_loops(self):
        """Test no self-connections"""
        nodes = place_grid_nodes(n=3, area_size=30)
        conn = get_connectivity_matrix(nodes, comm_radius=50)
        self.assertEqual(np.diag(conn).sum(), 0)

    def test_connectivity_radius(self):
        """Test connectivity respects communication radius"""
        # Create two nodes at known distance
        nodes = [
            NodeGeometry(0, x=0, y=0),
            NodeGeometry(1, x=10, y=0),
            NodeGeometry(2, x=20, y=0)
        ]

        # Test with radius that only connects neighbors
        conn = get_connectivity_matrix(nodes, comm_radius=15)
        self.assertTrue(conn[0, 1])   # 0-1 connected (dist=10)
        self.assertTrue(conn[1, 2])   # 1-2 connected (dist=10)
        self.assertFalse(conn[0, 2])  # 0-2 not connected (dist=20)


class TestGraphRigidity(unittest.TestCase):
    """Test graph rigidity checking"""

    def test_rigidity_sufficient_anchors(self):
        """Test rigidity with sufficient anchor connections"""
        # Create simple setup with known connectivity
        nodes = [
            NodeGeometry(0, x=0, y=0, is_anchor=True),
            NodeGeometry(1, x=10, y=0, is_anchor=True),
            NodeGeometry(2, x=5, y=5, is_anchor=True),
            NodeGeometry(3, x=5, y=2, is_anchor=False)  # Unknown node
        ]

        # Full connectivity
        conn = get_connectivity_matrix(nodes, comm_radius=100)
        rigidity = check_graph_rigidity(nodes, conn, min_anchor_edges=3)

        # Node 3 should be rigid (connected to 3 anchors)
        self.assertTrue(rigidity[3])

    def test_rigidity_insufficient_anchors(self):
        """Test rigidity with insufficient anchor connections"""
        # Create setup where unknown node only sees 2 anchors
        nodes = [
            NodeGeometry(0, x=0, y=0, is_anchor=True),
            NodeGeometry(1, x=10, y=0, is_anchor=True),
            NodeGeometry(2, x=100, y=100, is_anchor=True),  # Far anchor
            NodeGeometry(3, x=5, y=2, is_anchor=False)  # Unknown node
        ]

        # Limited connectivity
        conn = get_connectivity_matrix(nodes, comm_radius=20)
        rigidity = check_graph_rigidity(nodes, conn, min_anchor_edges=3)

        # Node 3 should not be rigid (only 2 nearby anchors)
        self.assertFalse(rigidity[3])


if __name__ == "__main__":
    unittest.main()