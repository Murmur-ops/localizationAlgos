# Triage Report: Uploaded Work-in-Progress

**Verdict:** The structure is solid, but many modules contain literal `...` elisions and embedded Markdown, so the code as uploaded is **not executable**. You’ve got the right folders and filenames, but several key implementations are cut off mid-function.

## What I inspected
- `docs/Decentralized_Array_Message_Spec.md` and `docs/Integrated_Spread_Spectrum_Design.md` — present.
- `src/messages/protocol.py` — contains Markdown fragments and `...`; missing actual enum definitions and pack/unpack logic.
- `src/sync/frequency_sync.py` — PLL loop skeleton present but truncated with `...`.
- `src/rf/spread_spectrum.py` — waveform generator skeleton; `_rrc_filter` and PN generation cut off with `...`.
- `tests/test_*.py` — test scaffolding present but relies on incomplete implementations.

## Highest-impact blockers (fix in order)
1. **Remove Markdown from `.py` files and replace all `...` with code.**
2. **Define message enums and struct layouts** in `src/messages/protocol.py` (Header, BEACON, SYNC_REQ/RESP, RNG_REQ/RESP, TIME_STATE, LM_MSG). Add pack/unpack + AEAD stubs.
3. **Finish PLL and time KF** in `src/sync/frequency_sync.py` (expose `cfo_hz`, `sro_ppm`, `phase`, `locked`).
4. **Complete ranging correlator** in `src/rf/spread_spectrum.py` (long PN matched filter + sub-chip interpolator).
5. **Implement per-edge variance model** using SNR/BW/Allan parameters.
6. **Make `tests/test_full_system.py` run on a tiny topology (n=4)** with synthetic clean channel to verify end-to-end.

## Minimal working example: core message pack/unpack

```python
# src/messages/protocol_core.py
import struct
from enum import IntEnum
from dataclasses import dataclass

MAGIC = 0xDAEDBEEF

class MsgType(IntEnum):
    BEACON = 1
    SYNC_REQ = 2
    SYNC_RESP = 3
    RNG_REQ = 4
    RNG_RESP = 5
    TIME_STATE = 6

HEADER_FMT = "<I B H I Q"  # magic,u8 msg_type,u16 hdr_len,u32 seq,u64 src_id
HEADER_SIZE = struct.calcsize(HEADER_FMT)

@dataclass
class Header:
    msg_type: int
    seq: int
    src_id: int

    def pack(self):
        return struct.pack(HEADER_FMT, MAGIC, self.msg_type, HEADER_SIZE, self.seq, self.src_id)

    @staticmethod
    def unpack(buf: bytes):
        magic, mtype, hdr_len, seq, src = struct.unpack_from(HEADER_FMT, buf, 0)
        if magic != MAGIC or hdr_len != HEADER_SIZE:
            raise ValueError("Bad header")
        return Header(mtype, seq, src), HEADER_SIZE

SYNC_REQ_FMT = "<Q I"  # t1_tx_local (ns), turnaround_hint_ns

def pack_sync_req(h: Header, t1_tx_local: int, turnaround_hint_ns: int) -> bytes:
    return h.pack() + struct.pack(SYNC_REQ_FMT, t1_tx_local, turnaround_hint_ns)

def unpack_sync_req(buf: bytes):
    h, off = Header.unpack(buf)
    t1, hint = struct.unpack_from(SYNC_REQ_FMT, buf, off)
    return h, t1, hint

SYNC_RESP_FMT = "<Q Q I"  # t2_rx_local, t3_tx_local, proc_latency_ns

def pack_sync_resp(h: Header, t2_rx_local: int, t3_tx_local: int, proc_latency_ns: int) -> bytes:
    return h.pack() + struct.pack(SYNC_RESP_FMT, t2_rx_local, t3_tx_local, proc_latency_ns)

def unpack_sync_resp(buf: bytes):
    h, off = Header.unpack(buf)
    t2, t3, proc = struct.unpack_from(SYNC_RESP_FMT, buf, off)
    return h, t2, t3, proc
```

Drop this in as a separate module to get tests passing while you flesh out the full `protocol.py`.

## Sanity tests to add immediately
- **Header round-trip:** pack→unpack preserves `msg_type, seq, src_id` for 100 random cases.
- **SYNC pack/unpack:** random timestamps; verify equality and struct sizes.
- **Ranging variance function:** plug a grid of (B, SNR, τ, σ_y) and check monotonicity w.r.t. SNR↑, BW↑, τ↑.

## Next steps I can take
- Replace your `...` in `spread_spectrum.py` with a functional RRC filter and PN generator.
- Provide a tiny PLL loop (2nd-order) with unit tests.
- Wire a deterministic 4-node simulation (`tests/test_integrated_system.py`) so CI turns green.
