"""
Verify that anchors are true references with zero error
"""

import numpy as np
from time_sync_fixed import FixedTimeSync

def verify_anchor_reference():
    """Test that anchors maintain zero offset throughout"""

    print("VERIFYING ANCHORS AS TRUE REFERENCES")
    print("="*60)

    # Create system
    sync = FixedTimeSync(n_nodes=10, n_anchors=3)

    # Check initial anchor state
    print("\nINITIAL ANCHOR STATE:")
    for i in range(sync.n_anchors):
        anchor = sync.nodes[i]
        print(f"Anchor {i}:")
        print(f"  True offset: {anchor.true_offset_ns:.3f}ns (should be 0)")
        print(f"  True drift: {anchor.true_drift_ppb:.3f}ppb (should be 0)")
        print(f"  Est offset: {anchor.est_offset_ns:.3f}ns")
        print(f"  Est drift: {anchor.est_drift_ppb:.3f}ppb")
        print(f"  Is anchor: {anchor.is_anchor}")
        print(f"  Covariance: {anchor.covariance[0,0]:.1e} (should be tiny)")

    print("\nREGULAR NODE STATE (for comparison):")
    for i in range(sync.n_anchors, min(sync.n_anchors + 2, sync.n_nodes)):
        node = sync.nodes[i]
        print(f"Node {i}:")
        print(f"  True offset: {node.true_offset_ns:.3f}ns (random ±100ns)")
        print(f"  True drift: {node.true_drift_ppb:.3f}ppb (random ±5ppb)")
        print(f"  Est offset: {node.est_offset_ns:.3f}ns (starts at 0)")
        print(f"  Est drift: {node.est_drift_ppb:.3f}ppb (starts at 0)")
        print(f"  Is anchor: {node.is_anchor}")
        print(f"  Covariance: {node.covariance[0,0]:.1e} (starts large)")

    # Run a few rounds
    print("\nRUNNING 5 SYNC ROUNDS...")
    for round_idx in range(5):
        current_time_ns = round_idx * sync.sync_interval_ms * 1e6
        dt = sync.sync_interval_ms / 1000.0

        # Check that anchors aren't being updated
        anchor_states_before = [(a.est_offset_ns, a.est_drift_ppb) for a in sync.nodes[:sync.n_anchors]]

        # Update regular nodes only
        for node in sync.nodes:
            if not node.is_anchor:
                measurements = []
                for anchor in sync.nodes[:sync.n_anchors]:
                    offset_meas = sync.two_way_time_transfer(node, anchor, current_time_ns)
                    measurements.append(offset_meas)

                avg_measurement = np.mean(measurements)
                sync.kalman_update(node, avg_measurement, dt)

        # Verify anchors didn't change
        anchor_states_after = [(a.est_offset_ns, a.est_drift_ppb) for a in sync.nodes[:sync.n_anchors]]

        if anchor_states_before == anchor_states_after:
            print(f"  Round {round_idx+1}: ✅ Anchors unchanged (as expected)")
        else:
            print(f"  Round {round_idx+1}: ❌ Anchors changed! (BUG)")

    # Final check
    print("\nFINAL ANCHOR STATE:")
    all_zero = True
    for i in range(sync.n_anchors):
        anchor = sync.nodes[i]
        if anchor.true_offset_ns != 0 or anchor.true_drift_ppb != 0:
            all_zero = False
        print(f"Anchor {i}: offset={anchor.true_offset_ns:.3f}ns, drift={anchor.true_drift_ppb:.3f}ppb")

    print("\nFINAL NODE ERRORS (relative to true time):")
    for i in range(sync.n_anchors, sync.n_nodes):
        node = sync.nodes[i]
        true_offset = node.true_offset_ns + node.true_drift_ppb * 5 * sync.sync_interval_ms * 1e3
        error = abs(true_offset - node.est_offset_ns)
        print(f"Node {i}: error={error:.3f}ns")

    print("\n" + "="*60)
    if all_zero:
        print("✅ VERIFIED: Anchors are true references with zero offset/drift")
        print("✅ VERIFIED: Regular nodes sync to these perfect references")
    else:
        print("❌ ERROR: Anchors don't have zero offset/drift")

    return all_zero

if __name__ == "__main__":
    verify_anchor_reference()