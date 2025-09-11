# Decentralized Array: On-Air Message Set & Procedure

This document defines a deployment-grade message set and over-the-air procedure for a **decentralized array** with **unique node IDs** and **some initialized anchors**. It supports: **(1) frequency lock**, **(2) time sync (offset+skew)**, **(3) RTT ranging**, and **(4) decentralized localization updates**—without a fusion center.

**Assumptions**
- 64-bit node IDs (`node_id`), unique.
- Hardware TX/RX timestamping at the MAC/PHY boundary.
- One narrow **control channel** (for beacons/sync/control) and a wideband **ranging channel**.
- Little-endian, fixed-width fields; times are **TAI** nanoseconds.

---

## Superframe (control-plane schedule)
A leader/anchor emits a repeating **superframe** (e.g., 100 ms):
- **PILOT slots** (always on): continuous pilot/comb for CFO/SRO lock.
- **SYNC slots**: two-way time messages in reserved micro-slots (hash of `node_id` → slot).
- **RANGING slots**: RTT bursts with neighbors (TDMA).
- **JOIN/CONTENTion**: CSMA for late joiners to request slots.

Parameters to publish in the beacon: `slot_period_ms`, `pilot_center_freq`, `pilot_spacing_hz`, `sync_slot_count`, `rng_slot_count`.

---

## Message set (wire format)

All control messages begin with:
```
Header {
  uint8   version     // e.g., 1
  uint8   msg_type    // enumerated below
  uint16  hdr_len     // bytes including header
  uint32  seq         // per-sender sequence
  uint64  src_id      // sender node_id
}
AuthTrailer {
  uint64  auth_nonce
  uint128 auth_tag    // 16B AEAD tag (e.g., AES-GCM)
}
```
> All payload fields are **hardware-timestamped** where noted.

### 1) Beacon / Pilot announcement (anchors & leader)
```
BEACON {
  uint8   role        // 0=NODE,1=ANCHOR,2=LEADER
  uint8   quality     // oscillator tier, 0..255
  uint16  superframe_ms
  uint64  t_anchor_tx // TAI_ns at TX (anchor/leaders only) [HW stamp]
  float32 fc_hz       // control channel center
  float32 rng_fc_hz   // ranging channel center
  float32 pilot_spacing_hz
  float32 sample_rate_hz
  uint8   has_pos     // 0/1
  float32 pos_x_m     // if has_pos==1 (anchor position)
  float32 pos_y_m
  float32 pos_z_m
  float32 pos_cov[6]  // xx,yy,zz,xy,xz,yz (optional, else zeros)
}
```
**Purpose:** discovery, leader/anchor advertisement, pilot parameters.

### 2) Frequency-lock (pilot is continuous; no packet needed)
Each node runs a PLL on the pilot/comb and exposes:
- `cfo_hz` (residual CFO estimate),
- `sro_ppm` (sample-rate offset),
- `pll_var` (quality metric).

Optionally, nodes exchange those as **DFAC** consensus messages:
```
FREQ_CONSENSUS {
  float32 cfo_hz_est
  float32 sro_ppm_est
  float32 cfo_var
  float32 sro_var
}
```
Nodes update their NCOs by weighted averaging of neighbors’ set-points (doubly-stochastic mixing). This is optional if you have a strong anchor pilot; required for anchorless swarms.

### 3) Two-way time sync (PTP-style, four timestamps)
```
SYNC_REQ {                         // sent by A
  uint64 t1_tx_local               // A's HW TX time at send
  uint32 turnaround_hint_ns        // expected RESP latency at peer
}

SYNC_RESP {                        // sent by B in reply
  uint64 t2_rx_local               // B's HW RX time of SYNC_REQ
  uint64 t3_tx_local               // B's HW TX time of SYNC_RESP
  uint32 proc_latency_ns           // measured/kalibrated turnaround
}
SYNC_ACK {                         // optional ack from A (for stats)
  uint64 t4_rx_local               // A's HW RX time of SYNC_RESP
  float32 jitter_ns                // RX timestamp jitter est.
}
```
**Estimator on A (for pair A↔B):** from `(t1,t2,t3,t4)` compute offset \(\hat\beta\) and RTT; repeat K times; track **offset & skew** with a tiny Kalman filter per neighbor. Share \(\hat\beta\), variance to consensus (below) to build **network-wide time**.

### 4) RTT ranging burst (wideband)
```
RNG_REQ {                          // A -> B
  uint64 t1_tx_local               // A HW TX time
  uint8  waveform_id               // defines β (mean-square bandwidth)
  uint16 dwell_us                  // integration time τ
}

RNG_RESP {                         // B -> A
  uint64 t2_rx_local               // B HW RX time of RNG_REQ
  uint64 t3_tx_local               // B HW TX time of RNG_RESP
  float32 snr_db_post              // post-integration SNR estimate ρ
  float32 path_q                   // multipath/NLOS score 0..1
}
RNG_ACK {                          // A -> B (optional)
  uint64 t4_rx_local               // A HW RX time of RNG_RESP
}
```
**A** computes range and **per-edge variance**:
\[
\widehat{d}_{AB} \approx \tfrac{c}{2}\big[(t_4-t_1) - (t_3-t_2) - \Delta_\text{proc}\big],\quad
\sigma_{d,AB}^2 = \frac{c^2}{2\beta^2\rho} + (c\,\tau\,\sigma_y(\tau))^2
\]
Use `waveform_id→β`, `snr_db_post→ρ`, integration `dwell_us→τ`, and device Allan deviation table → \(\sigma_y(\tau)\). Keep `path_q` to gate/robustify later.

### 5) Time-consensus (offset field propagation)
Each node periodically shares its **current best** time parameters (relative to the elected reference):
```
TIME_STATE {
  float64 alpha    // skew (ppm → α≈1+ppm*1e-6); or send ln(α)
  float64 beta_ns  // offset (ns) in leader/anchor frame
  float32 var_beta // variance (ns^2) from KF
  float32 var_alpha
}
```
Neighbors run **weighted Jacobi/Gauss-Seidel** or **Covariance-Intersection** consensus to agree on \(\beta\) (and optionally \(\alpha\)) network-wide. Pin anchor’s \((\alpha,\beta)=(1,0)\).

### 6) Localization messages (decentralized solver)

**6a) Graph announcement (sparse and infrequent)**
```
NEIGHBOR_LINK {
  uint64 peer_id
  float32 d_hat_m       // latest range estimate
  float32 sigma_d_m     // std dev (from SNR/BW/clock)
  uint8   quality       // derived from path_q, SNR, recency
}
```
You can piggyback these in beacons or exchange on demand.

**6b) Distributed LM / ADMM updates (per edge)**
When running the solver, each node i maintains its state \(x_i \in \mathbb{R}^2/\mathbb{R}^3\). For neighbor j it sends linearized info:
```
LM_MSG {
  uint64 peer_id
  float32 JtWJ_ii[dim*dim]   // local Hessian block H_ii
  float32 JtWJ_ij[dim*dim]   // cross block H_ij  (optional if using ADMM)
  float32 JtWr_i[dim]        // gradient block g_i
  float32 damping            // λ (LM)
  uint8   robust_type        // 0=L2,1=Huber,2=Cauchy
  float32 robust_scale       // δ_ij
}
```
Nodes perform local solves, exchange with neighbors, and update via **ADMM consensus** or **Gauss–Seidel** until the **robust cost** stops decreasing. If anchors exist, they broadcast **priors**:
```
ANCHOR_PRIOR {
  float32 x_m[dim]           // (x,y[,z])
  float32 cov[dim*(dim+1)/2] // upper-triangular covariance
}
```

**6c) Optional: local PSD projection (guardrail)**
If you keep the lifted local matrix \(S_i\) (paper’s useful bit), nodes can exchange overlap entries for consistency; this can be encapsulated as:
```
PSD_OVERLAP {
  uint64  peer_id
  uint16  idx[]      // indices of shared lift entries
  float32 val[]      // averaged values post-projection
}
```

---

## Join procedure (node that boots cold)

1) **Listen to BEACONs** for ≤1 s → pick best anchor/leader (or trigger election if none).  
2) **Lock to PILOT** → run PLL for CFO/SRO; residual < ~1 ppm.  
3) **Two-way SYNC** with ≥2 neighbors (or anchor) → estimate \((\beta,\alpha)\) in a per-neighbor KF (K≈8–16 exchanges).  
4) **TIME_STATE consensus** → settle on network time (pin to anchor frame if present).  
5) **Ranging burst** in next RANGING slots with neighbors → compute `d_hat` and `sigma_d`.  
6) **Localization init** → run robust SMACOF locally using received `NEIGHBOR_LINK`s.  
7) **Distributed LM/ADMM** → send/receive `LM_MSG` with neighbors for a few iterations (e.g., 3–5 per second) until robust cost plateaus.  
8) **Track** → keep pilots, periodic SYNC, periodic RTT, incremental LM.

---

## Field sizes & overhead (typical)

- **BEACON**: ~64–96 B (w/ position).  
- **SYNC_REQ/RESP**: ~28–36 B each (plus 16 B AEAD tag).  
- **RNG_REQ/RESP**: ~28–40 B (plus tag).  
- **TIME_STATE / FREQ_CONSENSUS**: ~24–32 B.  
- **LM_MSG**: depends on `dim` (2D: H/G blocks are small; expect 48–96 B per neighbor per iteration).

Keep control traffic < **5–10 kbps/node** even with 10–20 neighbors and 5 Hz solver iterations.

---

## Security & sanity checks

- All control/ranging messages carry an **AEAD tag** (e.g., AES-GCM) and a rolling `auth_nonce` bound to superframe/seq to defeat replay.  
- Drop packets if **HW timestamp not present** or if PLL isn’t locked (pilot SNR too low).  
- Gate edges using innovation χ² in the LM; demote links with persistent multipath (`path_q` low).  
- Assert invariants: `diag(Z)==2`, `W·1=0` (if you use a consensus Laplacian), symmetry of Y after averaging, robust cost monotone non-increasing under damping.

---

## Implementation order (minimal viable prototype)

1) **BEACON + PILOT** (just transmit; PLL on receivers).  
2) **SYNC_REQ/RESP** with **hardware timestamps** and a tiny KF for \((\beta,\alpha)\).  
3) **RNG_REQ/RESP** and per-edge \(\sigma_{d,ij}\) from SNR/BW/clock.  
4) **NEIGHBOR_LINK** exchange + **robust SMACOF** init.  
5) **LM_MSG** loop with neighbors (ADMM or Gauss–Seidel) + damping + robust loss.

This message set stands up a working field prototype. Add optional frequency consensus (DFAC), PSD guardrail messages, and reduced SDP micro-solves out-of-band if you want certificates.
