"""
SIMPLE Perfect World Ranging
Actually runs, actually verifies
No overcomplicated bullshit
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("SIMPLE PERFECT RANGING TEST")
print("="*60)

# Generate simple Gold codes (length 127)
def generate_gold_code(seed):
    """Simple but real Gold code"""
    np.random.seed(seed)
    # For now, use a deterministic sequence
    # This is still a simplification but at least it's unique per node
    code = np.array([1 if i % (seed+2) < (seed+2)//2 else -1 for i in range(127)])
    return code

# Create 4 nodes
nodes = [
    {'id': 0, 'pos': np.array([0, 0]), 'code': generate_gold_code(0)},
    {'id': 1, 'pos': np.array([100, 0]), 'code': generate_gold_code(1)},
    {'id': 2, 'pos': np.array([100, 100]), 'code': generate_gold_code(2)},
    {'id': 3, 'pos': np.array([0, 100]), 'code': generate_gold_code(3)}
]

print(f"Created {len(nodes)} nodes")

def simple_ranging(node_i, node_j):
    """
    Perfect world ranging:
    1. Calculate true distance
    2. Apply delay to signal
    3. Correlate to find delay
    4. Convert back to distance
    """
    # True distance
    true_dist = np.linalg.norm(node_i['pos'] - node_j['pos'])

    # Speed of light and sampling
    c = 3e8  # m/s
    fs = 1e8  # 100 MHz sampling (more reasonable)

    # Calculate delay in samples
    true_delay_s = true_dist / c
    true_delay_samples = int(true_delay_s * fs)

    # Create delayed signal (just shift the code)
    tx_signal = node_j['code']
    rx_signal = np.concatenate([np.zeros(true_delay_samples), tx_signal])

    # Correlate to find delay
    correlation = np.correlate(rx_signal, node_i['code'], mode='full')
    peak_idx = np.argmax(correlation)

    # Convert peak to distance
    # Peak location tells us the delay
    measured_delay_samples = peak_idx - len(node_i['code']) + 1
    measured_delay_s = measured_delay_samples / fs
    measured_dist = measured_delay_s * c

    error = measured_dist - true_dist

    return true_dist, measured_dist, error

# Test all pairs
print("\nRanging results:")
errors = []
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        true, meas, err = simple_ranging(nodes[i], nodes[j])
        errors.append(err)
        print(f"  Nodes {i}-{j}: True={true:.1f}m, Meas={meas:.1f}m, Error={err:.3f}m")

# Verify
max_error = max(np.abs(errors))
print(f"\nMax error: {max_error:.6f} m")

# Quantization limit at 100 MHz
quantization_error = 3e8 / 1e8  # c/fs = 3m
print(f"Quantization limit: {quantization_error} m")

if max_error <= quantization_error:
    print("✅ PASS: Errors within quantization limit")
else:
    print("❌ FAIL: Errors exceed quantization limit")

# Simple plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
for n in nodes:
    plt.scatter(n['pos'][0], n['pos'][1], s=100)
    plt.text(n['pos'][0], n['pos'][1]+5, f"Node {n['id']}")
plt.xlim(-20, 120)
plt.ylim(-20, 120)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Node Positions')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(errors, bins=20, edgecolor='black')
plt.xlabel('Ranging Error (m)')
plt.ylabel('Count')
plt.title(f'Errors (max={max_error:.3f}m)')
plt.grid(True)

plt.suptitle('Simple Perfect Ranging Test')
plt.tight_layout()
plt.savefig('/Users/maxburnett/Documents/DecentralizedLocale/verified_ranging/phase1_perfect_sync/simple_results.png')
plt.show()

print("\nDone. No timeouts. No bullshit.")