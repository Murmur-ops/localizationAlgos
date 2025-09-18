"""
Unit tests for consensus message types
Verify serialization, deserialization, and message integrity
"""

import numpy as np
import pytest
import time
import json
from ftl.consensus.message_types import (
    StateMessage, NetworkMessage, MeasurementMessage,
    ConvergenceStatus, MessageType
)


class TestStateMessage:
    """Test state message creation, serialization, and validation"""

    def test_state_message_creation(self):
        """Test creating a state message with valid data"""
        state = np.array([10.0, 20.0, 5.0, 0.1, 0.01])  # x, y, bias, drift, cfo
        msg = StateMessage(
            node_id=1,
            state=state,
            iteration=5,
            timestamp=1234567890.0
        )

        assert msg.node_id == 1
        assert np.allclose(msg.state, state)
        assert msg.iteration == 5
        assert msg.timestamp == 1234567890.0
        assert msg.covariance is None
        assert msg.auth_tag is None

    def test_state_validation(self):
        """Test that invalid state dimensions are rejected"""
        with pytest.raises(AssertionError, match="State must be 5D"):
            StateMessage(
                node_id=1,
                state=np.array([1.0, 2.0]),  # Wrong dimension
                iteration=0,
                timestamp=0.0
            )

    def test_state_conversion_from_list(self):
        """Test automatic conversion from list to numpy array"""
        state_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        msg = StateMessage(
            node_id=1,
            state=state_list,
            iteration=0,
            timestamp=0.0
        )

        assert isinstance(msg.state, np.ndarray)
        assert np.allclose(msg.state, np.array(state_list))

    def test_serialization_deserialization(self):
        """Test round-trip serialization"""
        original = StateMessage(
            node_id=42,
            state=np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
            iteration=10,
            timestamp=1234567890.123
        )

        # Serialize
        serialized = original.serialize()
        assert isinstance(serialized, bytes)

        # Deserialize
        recovered = StateMessage.deserialize(serialized)

        assert recovered.node_id == original.node_id
        assert np.allclose(recovered.state, original.state)
        assert recovered.iteration == original.iteration
        assert recovered.timestamp == original.timestamp

    def test_serialization_with_covariance(self):
        """Test serialization with covariance matrix"""
        cov = np.eye(5) * 0.1
        msg = StateMessage(
            node_id=1,
            state=np.zeros(5),
            iteration=0,
            timestamp=0.0,
            covariance=cov
        )

        serialized = msg.serialize()
        recovered = StateMessage.deserialize(serialized)

        assert recovered.covariance is not None
        assert np.allclose(recovered.covariance, cov)

    def test_authentication_tag(self):
        """Test authentication tag generation"""
        msg = StateMessage(
            node_id=1,
            state=np.ones(5),
            iteration=0,
            timestamp=0.0
        )

        serialized = msg.serialize()
        assert msg.auth_tag is not None
        assert len(msg.auth_tag) == 8  # 8 bytes as specified

        # Verify deserialized message has same auth tag
        recovered = StateMessage.deserialize(serialized)
        assert recovered.auth_tag == msg.auth_tag

    def test_message_age(self):
        """Test age calculation"""
        old_time = time.time() - 5.0
        msg = StateMessage(
            node_id=1,
            state=np.zeros(5),
            iteration=0,
            timestamp=old_time
        )

        age = msg.age()
        assert 4.9 < age < 5.1  # Allow small timing variance

    def test_property_accessors(self):
        """Test position and clock parameter properties"""
        state = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        msg = StateMessage(
            node_id=1,
            state=state,
            iteration=0,
            timestamp=0.0
        )

        assert np.allclose(msg.position, np.array([10.0, 20.0]))
        assert np.allclose(msg.clock_params, np.array([30.0, 40.0, 50.0]))

    def test_deserialization_error_handling(self):
        """Test deserialization with invalid data"""
        # Invalid JSON
        with pytest.raises(ValueError, match="Failed to deserialize"):
            StateMessage.deserialize(b"not valid json")

        # Missing fields
        incomplete_data = json.dumps({'node_id': 1}).encode()
        with pytest.raises(ValueError, match="Failed to deserialize"):
            StateMessage.deserialize(incomplete_data)


class TestNetworkMessage:
    """Test generic network message wrapper"""

    def test_network_message_creation(self):
        """Test creating a network message"""
        payload = {"data": "test"}
        msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=2,
            payload=payload
        )

        assert msg.msg_type == MessageType.STATE
        assert msg.sender_id == 1
        assert msg.receiver_id == 2
        assert msg.payload == payload
        assert msg.hop_count == 0

    def test_broadcast_detection(self):
        """Test broadcast message detection"""
        # Broadcast message
        broadcast_msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=-1,
            payload=None
        )
        assert broadcast_msg.is_broadcast()

        # Unicast message
        unicast_msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=2,
            payload=None
        )
        assert not unicast_msg.is_broadcast()

    def test_hop_increment(self):
        """Test hop count increment for routing"""
        msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=2,
            payload=None
        )

        assert msg.hop_count == 0
        msg.increment_hop()
        assert msg.hop_count == 1
        msg.increment_hop()
        assert msg.hop_count == 2

    def test_staleness_check(self):
        """Test message staleness detection"""
        # Fresh message
        fresh_msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=2,
            payload=None,
            timestamp=time.time()
        )
        assert not fresh_msg.is_stale(max_age=1.0)

        # Old message
        old_msg = NetworkMessage(
            msg_type=MessageType.STATE,
            sender_id=1,
            receiver_id=2,
            payload=None,
            timestamp=time.time() - 2.0
        )
        assert old_msg.is_stale(max_age=1.0)
        assert not old_msg.is_stale(max_age=3.0)


class TestMeasurementMessage:
    """Test measurement message"""

    def test_measurement_creation(self):
        """Test creating measurement message"""
        msg = MeasurementMessage(
            from_node=1,
            to_node=2,
            measurement_type="toa",
            value=10.5,
            variance=0.01,
            timestamp=1234567890.0
        )

        assert msg.from_node == 1
        assert msg.to_node == 2
        assert msg.measurement_type == "toa"
        assert msg.value == 10.5
        assert msg.variance == 0.01

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for serialization"""
        msg = MeasurementMessage(
            from_node=1,
            to_node=2,
            measurement_type="tdoa",
            value=5.0,
            variance=0.04,
            timestamp=1234567890.0
        )

        d = msg.to_dict()
        assert d['from_node'] == 1
        assert d['to_node'] == 2
        assert d['type'] == "tdoa"
        assert d['value'] == 5.0
        assert d['variance'] == 0.04
        assert d['timestamp'] == 1234567890.0


class TestConvergenceStatus:
    """Test convergence status message"""

    def test_convergence_creation(self):
        """Test creating convergence status"""
        status = ConvergenceStatus(
            node_id=1,
            iteration=10,
            converged=False,
            gradient_norm=1e-4,
            step_norm=1e-6,
            cost=100.0
        )

        assert status.node_id == 1
        assert status.iteration == 10
        assert not status.converged
        assert status.gradient_norm == 1e-4
        assert status.step_norm == 1e-6
        assert status.cost == 100.0

    def test_convergence_check(self):
        """Test convergence checking with tolerances"""
        # Not converged - high gradient
        status1 = ConvergenceStatus(
            node_id=1,
            iteration=10,
            converged=True,
            gradient_norm=1e-4,
            step_norm=1e-9,
            cost=100.0
        )
        assert not status1.has_converged(grad_tol=1e-6)

        # Not converged - high step
        status2 = ConvergenceStatus(
            node_id=1,
            iteration=10,
            converged=True,
            gradient_norm=1e-7,
            step_norm=1e-6,
            cost=100.0
        )
        assert not status2.has_converged(step_tol=1e-8)

        # Not converged - flag is False
        status3 = ConvergenceStatus(
            node_id=1,
            iteration=10,
            converged=False,
            gradient_norm=1e-7,
            step_norm=1e-9,
            cost=100.0
        )
        assert not status3.has_converged()

        # Converged
        status4 = ConvergenceStatus(
            node_id=1,
            iteration=10,
            converged=True,
            gradient_norm=1e-7,
            step_norm=1e-9,
            cost=100.0
        )
        assert status4.has_converged(grad_tol=1e-6, step_tol=1e-8)


class TestMessageTypes:
    """Test message type enum"""

    def test_message_type_values(self):
        """Test message type enum values"""
        assert MessageType.STATE.value == "state"
        assert MessageType.MEASUREMENT.value == "measurement"
        assert MessageType.CONVERGENCE.value == "convergence"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.REQUEST_STATE.value == "request_state"

    def test_message_type_comparison(self):
        """Test message type comparisons"""
        msg_type = MessageType.STATE
        assert msg_type == MessageType.STATE
        assert msg_type != MessageType.MEASUREMENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])