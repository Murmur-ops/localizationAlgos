"""
Message Protocol Implementation
Based on the Decentralized Array Message Spec
"""

import struct
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import IntEnum
import numpy as np


class MessageType(IntEnum):
    """Message types from spec"""
    BEACON = 0x01
    SYNC_REQ = 0x02
    SYNC_RESP = 0x03
    RNG_REQ = 0x04
    RNG_RESP = 0x05
    LM_MSG = 0x06
    TIME_STATE = 0x07
    DATA = 0x08


class NodeState(IntEnum):
    """Node states"""
    DISCOVERING = 0
    SYNCING = 1
    RANGING = 2
    LOCALIZING = 3
    OPERATIONAL = 4


@dataclass
class MessageHeader:
    """Common message header"""
    msg_type: MessageType
    seq_num: int
    sender_id: int
    timestamp_ns: int
    flags: int = 0
    
    def pack(self) -> bytes:
        """Pack header into bytes"""
        return struct.pack('>BHIQB', 
                          self.msg_type, 
                          self.flags,
                          self.sender_id,
                          self.timestamp_ns,
                          self.seq_num & 0xFF)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'MessageHeader':
        """Unpack header from bytes"""
        msg_type, flags, sender_id, timestamp_ns, seq_num = struct.unpack('>BHIQB', data[:16])
        return cls(MessageType(msg_type), seq_num, sender_id, timestamp_ns, flags)


@dataclass
class BeaconMessage:
    """Node discovery beacon"""
    node_id: int
    position: Optional[np.ndarray] = None  # Known if anchor
    is_anchor: bool = False
    state: NodeState = NodeState.DISCOVERING
    neighbor_count: int = 0
    
    def pack(self) -> bytes:
        """Pack beacon message"""
        flags = (self.is_anchor << 0) | (self.state << 1)
        
        if self.position is not None:
            x, y = self.position[:2]
            z = self.position[2] if len(self.position) > 2 else 0
            return struct.pack('>IBBfff', 
                             self.node_id, flags, self.neighbor_count,
                             x, y, z)
        else:
            return struct.pack('>IBB', self.node_id, flags, self.neighbor_count)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'BeaconMessage':
        """Unpack beacon message"""
        node_id, flags, neighbor_count = struct.unpack('>IBB', data[:6])
        is_anchor = bool(flags & 0x01)
        state = NodeState((flags >> 1) & 0x0F)
        
        position = None
        if len(data) >= 18 and is_anchor:
            x, y, z = struct.unpack('>fff', data[6:18])
            position = np.array([x, y, z])
        
        return cls(node_id, position, is_anchor, state, neighbor_count)


@dataclass
class SyncMessage:
    """Time synchronization messages"""
    msg_type: MessageType  # SYNC_REQ or SYNC_RESP
    initiator_id: int
    responder_id: int
    t1: int = 0  # TX timestamp at initiator
    t2: int = 0  # RX timestamp at responder
    t3: int = 0  # TX timestamp at responder (response)
    t4: int = 0  # RX timestamp at initiator (response received)
    
    def pack(self) -> bytes:
        """Pack sync message"""
        return struct.pack('>BIIQQQQ',
                          self.msg_type,
                          self.initiator_id,
                          self.responder_id,
                          self.t1, self.t2, self.t3, self.t4)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'SyncMessage':
        """Unpack sync message"""
        msg_type, init_id, resp_id, t1, t2, t3, t4 = struct.unpack('>BIIQQQQ', data[:41])
        return cls(MessageType(msg_type), init_id, resp_id, t1, t2, t3, t4)


@dataclass
class RangingMessage:
    """Ranging request/response messages"""
    msg_type: MessageType  # RNG_REQ or RNG_RESP
    initiator_id: int
    responder_id: int
    seq_num: int
    tx_timestamp_ns: int = 0
    rx_timestamp_ns: int = 0
    measured_distance_m: float = 0.0
    quality_score: float = 0.0
    snr_db: float = 0.0
    
    def pack(self) -> bytes:
        """Pack ranging message"""
        return struct.pack('>BIIHQQFFF',
                          self.msg_type,
                          self.initiator_id,
                          self.responder_id,
                          self.seq_num,
                          self.tx_timestamp_ns,
                          self.rx_timestamp_ns,
                          self.measured_distance_m,
                          self.quality_score,
                          self.snr_db)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'RangingMessage':
        """Unpack ranging message"""
        (msg_type, init_id, resp_id, seq_num, 
         tx_ts, rx_ts, dist, quality, snr) = struct.unpack('>BIIHQQFFF', data[:38])
        return cls(MessageType(msg_type), init_id, resp_id, seq_num,
                  tx_ts, rx_ts, dist, quality, snr)


@dataclass
class LocalizationMessage:
    """Distributed localization message (LM_MSG)"""
    node_id: int
    iteration: int
    position_estimate: np.ndarray
    gradient: np.ndarray
    dual_variables: Dict[int, float] = field(default_factory=dict)
    neighbor_distances: Dict[int, float] = field(default_factory=dict)
    neighbor_qualities: Dict[int, float] = field(default_factory=dict)
    
    def pack(self) -> bytes:
        """Pack localization message"""
        # Header: node_id, iteration, dimension
        dim = len(self.position_estimate)
        data = struct.pack('>IHB', self.node_id, self.iteration, dim)
        
        # Position and gradient
        for val in self.position_estimate:
            data += struct.pack('>f', val)
        for val in self.gradient:
            data += struct.pack('>f', val)
        
        # Number of neighbors
        n_neighbors = len(self.neighbor_distances)
        data += struct.pack('>B', n_neighbors)
        
        # Neighbor data
        for neighbor_id in sorted(self.neighbor_distances.keys()):
            dist = self.neighbor_distances[neighbor_id]
            quality = self.neighbor_qualities.get(neighbor_id, 1.0)
            dual = self.dual_variables.get(neighbor_id, 0.0)
            data += struct.pack('>Ifff', neighbor_id, dist, quality, dual)
        
        return data
    
    @classmethod
    def unpack(cls, data: bytes) -> 'LocalizationMessage':
        """Unpack localization message"""
        offset = 0
        
        # Header
        node_id, iteration, dim = struct.unpack('>IHB', data[offset:offset+7])
        offset += 7
        
        # Position and gradient
        position = np.zeros(dim)
        gradient = np.zeros(dim)
        
        for i in range(dim):
            position[i] = struct.unpack('>f', data[offset:offset+4])[0]
            offset += 4
        
        for i in range(dim):
            gradient[i] = struct.unpack('>f', data[offset:offset+4])[0]
            offset += 4
        
        # Neighbors
        n_neighbors = struct.unpack('>B', data[offset:offset+1])[0]
        offset += 1
        
        neighbor_distances = {}
        neighbor_qualities = {}
        dual_variables = {}
        
        for _ in range(n_neighbors):
            nid, dist, quality, dual = struct.unpack('>Ifff', data[offset:offset+16])
            neighbor_distances[nid] = dist
            neighbor_qualities[nid] = quality
            dual_variables[nid] = dual
            offset += 16
        
        return cls(node_id, iteration, position, gradient,
                  dual_variables, neighbor_distances, neighbor_qualities)


@dataclass
class TimeStateMessage:
    """Time synchronization state broadcast"""
    node_id: int
    clock_offset_ns: float
    clock_skew_ppm: float
    sync_quality: float
    reference_node_id: int
    last_sync_timestamp_ns: int
    
    def pack(self) -> bytes:
        """Pack time state message"""
        return struct.pack('>IffFIQ',
                          self.node_id,
                          self.clock_offset_ns,
                          self.clock_skew_ppm,
                          self.sync_quality,
                          self.reference_node_id,
                          self.last_sync_timestamp_ns)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'TimeStateMessage':
        """Unpack time state message"""
        (node_id, offset, skew, quality, 
         ref_id, last_sync) = struct.unpack('>IffFIQ', data[:28])
        return cls(node_id, offset, skew, quality, ref_id, last_sync)


class SuperframeScheduler:
    """TDMA superframe scheduler"""
    
    def __init__(self, superframe_duration_ms: float = 100.0,
                 n_slots: int = 10):
        self.superframe_duration_ms = superframe_duration_ms
        self.n_slots = n_slots
        self.slot_duration_ms = superframe_duration_ms / n_slots
        self.epoch_time_ns = int(time.time() * 1e9)
        
        # Slot assignments (node_id -> slot_number)
        self.slot_assignments = {}
        
    def assign_slot(self, node_id: int) -> int:
        """Assign TDMA slot to node"""
        if node_id not in self.slot_assignments:
            # Simple hash-based assignment
            slot = node_id % self.n_slots
            
            # Check for collisions and resolve
            used_slots = set(self.slot_assignments.values())
            while slot in used_slots:
                slot = (slot + 1) % self.n_slots
            
            self.slot_assignments[node_id] = slot
        
        return self.slot_assignments[node_id]
    
    def get_current_slot(self) -> int:
        """Get current slot number"""
        current_time_ns = int(time.time() * 1e9)
        elapsed_ns = current_time_ns - self.epoch_time_ns
        elapsed_ms = elapsed_ns / 1e6
        
        superframe_num = int(elapsed_ms / self.superframe_duration_ms)
        time_in_superframe_ms = elapsed_ms % self.superframe_duration_ms
        current_slot = int(time_in_superframe_ms / self.slot_duration_ms)
        
        return current_slot
    
    def time_to_next_slot(self, slot_num: int) -> float:
        """Time in ms until specified slot"""
        current_slot = self.get_current_slot()
        
        if slot_num > current_slot:
            slots_to_wait = slot_num - current_slot
        else:
            slots_to_wait = (self.n_slots - current_slot) + slot_num
        
        return slots_to_wait * self.slot_duration_ms
    
    def can_transmit(self, node_id: int) -> bool:
        """Check if node can transmit in current slot"""
        if node_id not in self.slot_assignments:
            return False
        
        return self.get_current_slot() == self.slot_assignments[node_id]


class MessageAuthenticator:
    """Simple message authentication (placeholder for real crypto)"""
    
    def __init__(self, shared_key: bytes = b"test_key"):
        self.shared_key = shared_key
    
    def generate_mac(self, message: bytes) -> bytes:
        """Generate message authentication code"""
        h = hashlib.sha256()
        h.update(self.shared_key)
        h.update(message)
        return h.digest()[:8]  # Use first 8 bytes
    
    def verify_mac(self, message: bytes, mac: bytes) -> bool:
        """Verify message authentication code"""
        expected_mac = self.generate_mac(message)
        return mac == expected_mac


class MessageBuffer:
    """Buffer for out-of-order message handling"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.messages = {}  # (sender_id, seq_num) -> message
        self.expected_seq = {}  # sender_id -> next expected seq_num
    
    def add_message(self, sender_id: int, seq_num: int, message: bytes) -> List[bytes]:
        """Add message to buffer, return list of deliverable messages"""
        # Initialize expected sequence if needed
        if sender_id not in self.expected_seq:
            self.expected_seq[sender_id] = 0
        
        # Store message
        self.messages[(sender_id, seq_num)] = message
        
        # Check if we can deliver messages in order
        deliverable = []
        expected = self.expected_seq[sender_id]
        
        while (sender_id, expected) in self.messages:
            deliverable.append(self.messages.pop((sender_id, expected)))
            expected += 1
            self.expected_seq[sender_id] = expected
        
        # Clean up old messages if buffer is full
        if len(self.messages) > self.max_size:
            # Remove oldest messages
            sorted_keys = sorted(self.messages.keys(), key=lambda x: x[1])
            for key in sorted_keys[:len(self.messages) - self.max_size]:
                del self.messages[key]
        
        return deliverable


if __name__ == "__main__":
    # Test message packing/unpacking
    print("Testing Message Protocol...")
    print("=" * 50)
    
    # Test beacon
    beacon = BeaconMessage(
        node_id=1,
        position=np.array([10.0, 20.0, 0.0]),
        is_anchor=True,
        state=NodeState.OPERATIONAL,
        neighbor_count=3
    )
    
    packed = beacon.pack()
    unpacked = BeaconMessage.unpack(packed)
    print(f"\nBeacon Message:")
    print(f"  Original: Node {beacon.node_id}, Pos {beacon.position}, Anchor: {beacon.is_anchor}")
    print(f"  Unpacked: Node {unpacked.node_id}, Pos {unpacked.position}, Anchor: {unpacked.is_anchor}")
    
    # Test sync message
    sync = SyncMessage(
        msg_type=MessageType.SYNC_REQ,
        initiator_id=1,
        responder_id=2,
        t1=1000000000,
        t2=1000001000,
        t3=1000002000,
        t4=1000003000
    )
    
    packed = sync.pack()
    unpacked = SyncMessage.unpack(packed)
    print(f"\nSync Message:")
    print(f"  Original: {sync.initiator_id}->{sync.responder_id}, t1={sync.t1}")
    print(f"  Unpacked: {unpacked.initiator_id}->{unpacked.responder_id}, t1={unpacked.t1}")
    
    # Test localization message
    loc_msg = LocalizationMessage(
        node_id=4,
        iteration=10,
        position_estimate=np.array([50.0, 30.0]),
        gradient=np.array([-0.1, 0.2]),
        neighbor_distances={1: 58.3, 2: 58.3, 3: 56.6},
        neighbor_qualities={1: 0.9, 2: 0.8, 3: 0.95}
    )
    
    packed = loc_msg.pack()
    unpacked = LocalizationMessage.unpack(packed)
    print(f"\nLocalization Message:")
    print(f"  Original: Node {loc_msg.node_id}, Pos {loc_msg.position_estimate}")
    print(f"  Unpacked: Node {unpacked.node_id}, Pos {unpacked.position_estimate}")
    print(f"  Neighbors: {unpacked.neighbor_distances}")
    
    # Test superframe scheduler
    scheduler = SuperframeScheduler()
    for node_id in [1, 2, 3, 4]:
        slot = scheduler.assign_slot(node_id)
        print(f"\nNode {node_id} assigned slot {slot}")
    
    current_slot = scheduler.get_current_slot()
    print(f"\nCurrent slot: {current_slot}")
    print(f"Node 1 can transmit: {scheduler.can_transmit(1)}")
    
    print("\n✅ Message protocol working correctly!")