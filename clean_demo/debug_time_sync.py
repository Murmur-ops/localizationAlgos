"""
Debug time sync to understand convergence issue
"""

import numpy as np

# Simulate one round of time sync
def test_time_sync_formula():
    """Test if the formula is working correctly"""

    # Setup: Node has +50ns offset, anchor has 0ns offset
    true_offset_node = 50.0  # ns
    true_offset_anchor = 0.0  # ns
    propagation_delay = 500.0  # ns
    processing_delay = 1000.0  # ns

    # Current time (arbitrary)
    current_time = 1e9  # 1 second in ns

    print("TEST SCENARIO:")
    print(f"  Node true offset: {true_offset_node}ns")
    print(f"  Anchor true offset: {true_offset_anchor}ns")
    print(f"  Propagation delay: {propagation_delay}ns")
    print(f"  Processing delay: {processing_delay}ns")
    print()

    # Simulate two-way exchange (node to anchor)
    # t1: Node sends (in node's time)
    t1 = current_time + true_offset_node
    print(f"t1 (node sends): {t1:.0f}ns")

    # t2: Anchor receives (in anchor's time)
    t2 = current_time + propagation_delay + true_offset_anchor
    print(f"t2 (anchor receives): {t2:.0f}ns")

    # t3: Anchor replies (in anchor's time)
    t3 = t2 + processing_delay
    print(f"t3 (anchor replies): {t3:.0f}ns")

    # t4: Node receives (in node's time)
    t4 = current_time + 2*propagation_delay + processing_delay + true_offset_node
    print(f"t4 (node receives): {t4:.0f}ns")
    print()

    # Try different formulas
    print("FORMULA TESTS:")

    # Formula 1: ((t2-t1) - (t4-t3))/2
    offset1 = ((t2 - t1) - (t4 - t3)) / 2
    print(f"1. ((t2-t1) - (t4-t3))/2 = {offset1:.1f}ns")
    print(f"   Error: {abs(offset1 - true_offset_node):.1f}ns")

    # Formula 2: ((t2-t1) + (t4-t3))/2
    offset2 = ((t2 - t1) + (t4 - t3)) / 2
    print(f"2. ((t2-t1) + (t4-t3))/2 = {offset2:.1f}ns")
    print(f"   Error: {abs(offset2 - true_offset_node):.1f}ns")

    # Formula 3: ((t2-t1) - (t3-t4))/2
    offset3 = ((t2 - t1) - (t3 - t4)) / 2
    print(f"3. ((t2-t1) - (t3-t4))/2 = {offset3:.1f}ns")
    print(f"   Error: {abs(offset3 - true_offset_node):.1f}ns")

    # Formula 4: ((t2-t1) + (t3-t4))/2
    offset4 = ((t2 - t1) + (t3 - t4)) / 2
    print(f"4. ((t2-t1) + (t3-t4))/2 = {offset4:.1f}ns")
    print(f"   Error: {abs(offset4 - true_offset_node):.1f}ns")

    print()
    print("ANALYSIS:")
    print(f"  t2-t1 = {t2-t1:.1f} (should be prop_delay - offset)")
    print(f"  t4-t3 = {t4-t3:.1f} (should be prop_delay + offset)")
    print(f"  t3-t4 = {t3-t4:.1f} (negative of above)")
    print()

    # The correct formula should give us the node's offset
    print("CORRECT FORMULA:")
    print("  For node measuring against anchor (anchor offset = 0):")
    print("  Node's offset = -((t2-t1) - (t4-t3))/2")
    offset_correct = -((t2 - t1) - (t4 - t3)) / 2
    print(f"  Result: {offset_correct:.1f}ns")
    print(f"  Error: {abs(offset_correct - true_offset_node):.1f}ns")

if __name__ == "__main__":
    test_time_sync_formula()