# Integrated Spread Spectrum Design for Time/Frequency Sync and Ranging

This document describes how to co-design a **single wideband spread-spectrum waveform** that supports:  
1. **Frequency lock (syntonization)**  
2. **Time sync (offset/skew)**  
3. **High-precision ranging**  

All without requiring a second RF chain.

---

## 1) Frame & waveform (one RF, three jobs)

### A. Time-division framing (every node Tx burst)
```
| Preamble (pilot + coarse sync) | Ranging block (long PN @ chip rate R_c) | Payload (data+pilot) |
|      0.3–1.0 ms                |            0.2–1.0 ms                   |      0.3–2.0 ms      |
```
- **Preamble:** pilot tone/comb + short PN for CFO/SRO and coarse timing.  
- **Ranging block:** long PN sequence at full chip rate \(R_c pprox B\). Pure known code, no data.  
- **Payload:** DSSS data on orthogonal PN, with embedded pilots.

### B. Superposition variant (no hard TDM)
Transmit as:
\(
s(t) = a_p p(t) + a_r r(t) + a_d d(t)
\)
- \(p(t)\): pilot tone/comb for loops  
- \(r(t)\): wideband ranging PN  
- \(d(t)\): DSSS data PN  
- Power split: \(a_p \gg a_d\), \(a_r \ge a_d\)

**Choice:** TDM = best ranging SNR. Superposition = lower latency.

---

## 2) Receiver processing

1. **Frequency & sample-rate lock**
   - PLL on pilot/comb for CFO/SRO.  
   - Timing recovery from pilot sequence.

2. **Time sync (offset/skew)**
   - Exchange hardware TX/RX timestamps (PTP-style, 4 timestamps).  
   - Run per-neighbor KF for offset/skew.

3. **Fine ranging**
   - Matched filter to long PN.  
   - Peak-pick leading edge for TOA (robust to multipath).  
   - Sub-chip interpolation (parabolic/MUSIC/ESPRIT).  
   - Range variance:  
   \[
   \sigma_d^2 pprox rac{c^2}{2eta^2ho} + (c	au\sigma_y(	au))^2
   \]

4. **Data demodulation**
   - DSSS despread with data PN.  
   - Use pilots for coherent combining.

5. **Distributed updates**
   - Feed CFO/SRO to DFAC consensus.  
   - Feed offset/skew variance to time consensus.  
   - Feed ranges into distributed LM/ADMM localization.

---

## 3) Multiple access

- **TDMA superframe:** disjoint ranging slots, payload in CSMA/TDMA.  
- **Code-division:** orthogonal/low-Xcorr codes for simultaneous ranging.  
- **Near-far control:** cap payload power, keep pilots strong.

---

## 4) Example parameters

- **RF BW \(B\):** 80–160 MHz (UWB).  
- **Ranging PN:** Gold/m-seq length 1023 or 4095. At 100 Mcps, ~41 µs block.  
- **Resolution:** \(\Delta d pprox c/(2B)\). At 100 MHz → 1.5 m raw; with interpolation, 0.1–0.3 m.  
- **Pilots:** tone + comb (e.g., ±1, 3, 5 MHz).  
- **Payload:** DSSS at 1–5 Mcps, QPSK.

---

## 5) Why it works

- **Pilots** give reliable loops for CFO/SRO and phase.  
- **Wideband PN** gives sharp correlation for ranging.  
- **TDM/orthogonalization** isolates ranging from data.  
- **Robust processing** handles multipath.

---

## 6) Minimal transmitter pseudocode

```c
emit_preamble(pilot_cfg, coarse_pn);
emit_ranging_block(ranging_pn, chip_rate=B);
emit_payload(data_symbols, data_pn, pilots);
```

**Receiver:**
```c
pll_track(pilot);
toa = fine_toa_correlate(ranging_pn);
data = despread_demod(payload, data_pn, pilots);
```

---

## 7) Bottom line

Use **pilots + short PN** for time/frequency sync, and **long PN at full bandwidth** for ranging. This yields stable loops and high-precision TOA with one integrated spread-spectrum waveform.
