"""
Distributed matrix operations for MPI-based SNL
Implements efficient distributed L matrix multiplication and other operations
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class DistributedMatrixOps:
    """Efficient distributed matrix operations for sparse block matrices"""
    
    def __init__(self, comm: MPI.Comm, n_sensors: int):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.n_sensors = n_sensors
        
        # Pre-compute communication patterns
        self.send_neighbors = {}  # {proc: [sensor_ids to send]}
        self.recv_neighbors = {}  # {proc: [sensor_ids to receive]}
        
        # Persistent communication requests for efficiency
        self.persistent_send_reqs = {}
        self.persistent_recv_reqs = {}
        
    def setup_communication_pattern(self, local_sensors: List[int], 
                                  sensor_neighbors: Dict[int, Set[int]]):
        """Pre-compute communication patterns for L matrix multiplication"""
        
        # Determine which sensors need to be sent/received to/from each process
        sensors_per_proc = self.n_sensors // self.size
        
        # For each local sensor, determine which remote processes need its data
        proc_send_lists = {i: set() for i in range(self.size) if i != self.rank}
        
        for sensor_id in local_sensors:
            # Find all sensors that have this sensor as a neighbor
            for other_sensor in range(self.n_sensors):
                if other_sensor in local_sensors:
                    continue  # Skip local sensors
                    
                # Check if other_sensor needs data from sensor_id
                # This would be provided by the neighbor information
                other_proc = self._get_process_for_sensor(other_sensor)
                if other_proc != self.rank:
                    # In practice, we'd check if sensor_id is in other_sensor's neighbors
                    # For now, we'll use a simplified pattern
                    if sensor_id in sensor_neighbors.get(other_sensor, set()):
                        proc_send_lists[other_proc].add(sensor_id)
        
        # Convert to lists for consistent ordering
        self.send_neighbors = {proc: sorted(list(sensors)) 
                              for proc, sensors in proc_send_lists.items() 
                              if sensors}
        
        # Exchange information about what each process will send
        for proc in range(self.size):
            if proc == self.rank:
                continue
                
            # Send what we'll send to proc
            if proc in self.send_neighbors:
                send_list = self.send_neighbors[proc]
            else:
                send_list = []
            
            # Receive what proc will send to us
            recv_list = self.comm.sendrecv(
                send_list, dest=proc, sendtag=0,
                source=proc, recvtag=0
            )
            
            if recv_list:
                self.recv_neighbors[proc] = recv_list
    
    def distributed_L_multiply(self, L_blocks: Dict[int, Dict[int, float]], 
                             v_local: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Perform distributed matrix-vector multiplication: result = L * v
        
        Args:
            L_blocks: Local blocks of L matrix as {sensor_id: {neighbor_id: value}}
            v_local: Local portions of vector v as {sensor_id: array}
            
        Returns:
            Local portion of result vector
        """
        
        # Step 1: Send required local vector elements to other processes
        send_buffers = self._pack_vector_data(v_local)
        recv_buffers = self._allocate_recv_buffers()
        
        # Non-blocking send/receive
        send_requests = []
        recv_requests = []
        
        for proc, buffer in send_buffers.items():
            req = self.comm.Isend(buffer, dest=proc, tag=1)
            send_requests.append(req)
        
        for proc, buffer in recv_buffers.items():
            req = self.comm.Irecv(buffer, source=proc, tag=1)
            recv_requests.append(req)
        
        # Step 2: Perform local multiplication while communication happens
        result = {}
        for sensor_id, v_i in v_local.items():
            result[sensor_id] = np.zeros_like(v_i)
            
            # Local L block row
            L_row = L_blocks.get(sensor_id, {})
            
            # Multiply with local vector elements
            for neighbor_id, L_ij in L_row.items():
                if neighbor_id in v_local:
                    result[sensor_id] += L_ij * v_local[neighbor_id]
        
        # Step 3: Wait for communication to complete
        MPI.Request.Waitall(send_requests + recv_requests)
        
        # Step 4: Unpack received data and complete multiplication
        remote_v = self._unpack_vector_data(recv_buffers)
        
        for sensor_id in result:
            L_row = L_blocks.get(sensor_id, {})
            
            # Add contributions from remote vector elements
            for neighbor_id, L_ij in L_row.items():
                if neighbor_id in remote_v:
                    result[sensor_id] += L_ij * remote_v[neighbor_id]
        
        return result
    
    def distributed_matrix_update(self, L_old: Dict[int, Dict[int, float]], 
                                update_factor: float = 0.1) -> Dict[int, Dict[int, float]]:
        """
        Update distributed matrix entries for convergence
        
        This implements a distributed update rule for matrix entries,
        useful for iterative algorithms like Sinkhorn-Knopp
        """
        
        # Step 1: Compute local row sums
        local_row_sums = {}
        for sensor_id, L_row in L_old.items():
            local_row_sums[sensor_id] = sum(L_row.values())
        
        # Step 2: Compute local column sums
        local_col_sums = np.zeros(self.n_sensors)
        for sensor_id, L_row in L_old.items():
            for neighbor_id, value in L_row.items():
                local_col_sums[neighbor_id] += value
        
        # Step 3: Global reduction for column sums
        global_col_sums = np.zeros(self.n_sensors)
        self.comm.Allreduce(local_col_sums, global_col_sums, op=MPI.SUM)
        
        # Step 4: Update matrix entries
        L_new = {}
        for sensor_id, L_row in L_old.items():
            L_new[sensor_id] = {}
            row_sum = local_row_sums[sensor_id]
            
            for neighbor_id, value in L_row.items():
                col_sum = global_col_sums[neighbor_id]
                
                # Sinkhorn-Knopp style update
                if row_sum > 0 and col_sum > 0:
                    new_value = value / (row_sum * col_sum) ** 0.5
                    L_new[sensor_id][neighbor_id] = new_value
                else:
                    L_new[sensor_id][neighbor_id] = value
        
        # Step 5: Normalize to maintain matrix properties
        return self._normalize_matrix_blocks(L_new)
    
    def distributed_inner_product(self, u_local: Dict[int, np.ndarray], 
                                v_local: Dict[int, np.ndarray]) -> float:
        """Compute distributed inner product <u, v>"""
        
        local_sum = 0.0
        for sensor_id in u_local:
            if sensor_id in v_local:
                local_sum += np.dot(u_local[sensor_id], v_local[sensor_id])
        
        # Global reduction
        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        return global_sum
    
    def distributed_norm(self, v_local: Dict[int, np.ndarray]) -> float:
        """Compute distributed L2 norm ||v||"""
        
        local_sum = 0.0
        for sensor_id, v_i in v_local.items():
            local_sum += np.dot(v_i, v_i)
        
        # Global reduction
        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        return np.sqrt(global_sum)
    
    def _get_process_for_sensor(self, sensor_id: int) -> int:
        """Determine which process owns a sensor"""
        sensors_per_proc = self.n_sensors // self.size
        remainder = self.n_sensors % self.size
        
        if sensor_id < remainder * (sensors_per_proc + 1):
            return sensor_id // (sensors_per_proc + 1)
        else:
            return (sensor_id - remainder) // sensors_per_proc + remainder
    
    def _pack_vector_data(self, v_local: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Pack vector data for sending to other processes"""
        
        send_buffers = {}
        
        for proc, sensor_list in self.send_neighbors.items():
            # Calculate buffer size
            n_sensors = len(sensor_list)
            if n_sensors == 0:
                continue
                
            # Assume 2D vectors for now
            buffer_size = n_sensors * 3  # sensor_id + 2D vector
            buffer = np.zeros(buffer_size)
            
            idx = 0
            for sensor_id in sensor_list:
                if sensor_id in v_local:
                    buffer[idx] = sensor_id
                    buffer[idx+1:idx+3] = v_local[sensor_id]
                    idx += 3
            
            send_buffers[proc] = buffer
        
        return send_buffers
    
    def _allocate_recv_buffers(self) -> Dict[int, np.ndarray]:
        """Allocate receive buffers based on communication pattern"""
        
        recv_buffers = {}
        
        for proc, sensor_list in self.recv_neighbors.items():
            n_sensors = len(sensor_list)
            if n_sensors > 0:
                buffer_size = n_sensors * 3  # sensor_id + 2D vector
                recv_buffers[proc] = np.zeros(buffer_size)
        
        return recv_buffers
    
    def _unpack_vector_data(self, recv_buffers: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Unpack received vector data"""
        
        remote_v = {}
        
        for proc, buffer in recv_buffers.items():
            idx = 0
            while idx < len(buffer):
                sensor_id = int(buffer[idx])
                vector = buffer[idx+1:idx+3]
                remote_v[sensor_id] = vector
                idx += 3
        
        return remote_v
    
    def _normalize_matrix_blocks(self, L_blocks: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
        """Normalize matrix blocks to maintain doubly stochastic property"""
        
        # Simple normalization - in practice would use more sophisticated method
        normalized = {}
        
        for sensor_id, L_row in L_blocks.items():
            row_sum = sum(L_row.values())
            if row_sum > 0:
                normalized[sensor_id] = {
                    neighbor: value / row_sum 
                    for neighbor, value in L_row.items()
                }
            else:
                normalized[sensor_id] = L_row.copy()
        
        return normalized


class DistributedSinkhornKnopp:
    """Distributed Sinkhorn-Knopp algorithm for doubly stochastic matrices"""
    
    def __init__(self, matrix_ops: DistributedMatrixOps):
        self.matrix_ops = matrix_ops
        self.comm = matrix_ops.comm
        self.rank = matrix_ops.rank
        
    def compute(self, adjacency_blocks: Dict[int, Set[int]], 
                max_iter: int = 100, tol: float = 1e-6) -> Dict[int, Dict[int, float]]:
        """
        Compute doubly stochastic matrix from adjacency structure
        
        Args:
            adjacency_blocks: Local adjacency lists {sensor_id: set of neighbors}
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Local blocks of doubly stochastic matrix
        """
        
        # Initialize matrix from adjacency
        L_blocks = {}
        for sensor_id, neighbors in adjacency_blocks.items():
            L_blocks[sensor_id] = {}
            n_neighbors = len(neighbors) + 1  # Include self
            
            # Uniform initialization
            for neighbor in neighbors:
                L_blocks[sensor_id][neighbor] = 1.0 / n_neighbors
            L_blocks[sensor_id][sensor_id] = 1.0 / n_neighbors
        
        # Sinkhorn-Knopp iterations
        converged = False
        for iteration in range(max_iter):
            
            # Row normalization (local)
            for sensor_id, L_row in L_blocks.items():
                row_sum = sum(L_row.values())
                if row_sum > 0:
                    for neighbor in L_row:
                        L_row[neighbor] /= row_sum
            
            # Column normalization (distributed)
            col_sums = self._compute_column_sums(L_blocks)
            
            max_deviation = 0.0
            for sensor_id, L_row in L_blocks.items():
                for neighbor, value in L_row.items():
                    if col_sums[neighbor] > 0:
                        new_value = value / col_sums[neighbor]
                        max_deviation = max(max_deviation, abs(new_value - value))
                        L_row[neighbor] = new_value
            
            # Check convergence
            global_max_deviation = self.comm.allreduce(max_deviation, op=MPI.MAX)
            
            if global_max_deviation < tol:
                converged = True
                if self.rank == 0:
                    logger.info(f"Sinkhorn-Knopp converged in {iteration+1} iterations")
                break
        
        if not converged and self.rank == 0:
            logger.warning(f"Sinkhorn-Knopp did not converge in {max_iter} iterations")
        
        return L_blocks
    
    def _compute_column_sums(self, L_blocks: Dict[int, Dict[int, float]]) -> np.ndarray:
        """Compute column sums with distributed reduction"""
        
        local_col_sums = np.zeros(self.matrix_ops.n_sensors)
        
        for sensor_id, L_row in L_blocks.items():
            for neighbor, value in L_row.items():
                local_col_sums[neighbor] += value
        
        # Global reduction
        global_col_sums = np.zeros(self.matrix_ops.n_sensors)
        self.comm.Allreduce(local_col_sums, global_col_sums, op=MPI.SUM)
        
        return global_col_sums


def test_distributed_operations():
    """Test distributed matrix operations"""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # Test parameters
    n_sensors = 20
    sensors_per_proc = n_sensors // size
    
    # Create local sensors
    local_sensors = list(range(rank * sensors_per_proc, (rank + 1) * sensors_per_proc))
    
    # Create simple adjacency (ring topology)
    adjacency = {}
    for sensor in local_sensors:
        neighbors = {(sensor - 1) % n_sensors, (sensor + 1) % n_sensors}
        adjacency[sensor] = neighbors
    
    # Initialize distributed operations
    matrix_ops = DistributedMatrixOps(comm, n_sensors)
    
    # Test Sinkhorn-Knopp
    sk = DistributedSinkhornKnopp(matrix_ops)
    L_blocks = sk.compute(adjacency)
    
    if rank == 0:
        print(f"Computed doubly stochastic matrix blocks")
        print(f"Example block: {list(L_blocks.values())[0]}")
    
    # Test matrix-vector multiplication
    v_local = {sensor: np.random.randn(2) for sensor in local_sensors}
    result = matrix_ops.distributed_L_multiply(L_blocks, v_local)
    
    # Verify properties
    norm = matrix_ops.distributed_norm(result)
    if rank == 0:
        print(f"Norm of L*v: {norm:.6f}")


if __name__ == "__main__":
    test_distributed_operations()