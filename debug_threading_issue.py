"""
Debug the threading performance issue
"""

import threading
import queue
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)


def test_barrier_deadlock():
    """Test if barriers are causing deadlock"""
    print("\n=== Testing Barrier Potential Deadlock ===")
    
    n_threads = 4
    barrier = threading.Barrier(n_threads)
    results = []
    
    def worker(id, barrier, delay):
        logger.info(f"Worker {id} starting")
        time.sleep(delay)
        logger.info(f"Worker {id} reaching barrier")
        try:
            barrier.wait(timeout=2.0)  # Add timeout to detect deadlock
            logger.info(f"Worker {id} passed barrier")
            results.append(id)
        except threading.BrokenBarrierError:
            logger.error(f"Worker {id} barrier broken!")
            
    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i, barrier, i * 0.1))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
        
    print(f"Results: {results}")
    print(f"Success: {len(results) == n_threads}")


def test_queue_blocking():
    """Test if queue operations are blocking"""
    print("\n=== Testing Queue Blocking ===")
    
    q = queue.Queue()
    
    def producer(q, n_items):
        logger.info("Producer starting")
        for i in range(n_items):
            q.put(f"item_{i}")
            logger.info(f"Produced item_{i}")
            time.sleep(0.01)
        logger.info("Producer done")
    
    def consumer(q, timeout=1.0):
        logger.info("Consumer starting")
        items = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                item = q.get(timeout=0.1)
                items.append(item)
                logger.info(f"Consumed {item}")
            except queue.Empty:
                logger.debug("Queue empty")
                
        logger.info(f"Consumer done, got {len(items)} items")
        return items
    
    # Test
    prod_thread = threading.Thread(target=producer, args=(q, 5))
    cons_thread = threading.Thread(target=lambda: consumer(q, 2.0))
    
    prod_thread.start()
    cons_thread.start()
    
    prod_thread.join()
    cons_thread.join()
    
    print("Queue test completed")


def test_threadpool_overhead():
    """Test ThreadPoolExecutor overhead"""
    print("\n=== Testing ThreadPoolExecutor Overhead ===")
    
    def simple_task(x):
        """Minimal computation"""
        return x * x
    
    n_tasks = 100
    data = list(range(n_tasks))
    
    # Sequential
    start = time.time()
    seq_results = [simple_task(x) for x in data]
    seq_time = time.time() - start
    
    # Threaded
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        thread_results = list(executor.map(simple_task, data))
    thread_time = time.time() - start
    
    print(f"Sequential time: {seq_time:.4f}s")
    print(f"Threaded time: {thread_time:.4f}s")
    print(f"Overhead ratio: {thread_time / seq_time:.2f}x")
    

def diagnose_snl_threading():
    """Diagnose specific SNL threading issues"""
    print("\n=== Diagnosing SNL Threading Pattern ===")
    
    # Simulate the SNL communication pattern
    n_sensors = 10
    
    class SimpleCommunicator:
        def __init__(self, n_sensors):
            self.queues = {i: queue.Queue() for i in range(n_sensors)}
            self.barriers = {}
            self.lock = threading.Lock()
            
        def send(self, from_id, to_id, data):
            self.queues[to_id].put({'from': from_id, 'data': data})
            
        def receive_with_timeout(self, sensor_id, timeout=0.1):
            messages = []
            deadline = time.time() + timeout
            
            while time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    msg = self.queues[sensor_id].get(timeout=min(remaining, 0.01))
                    messages.append(msg)
                except queue.Empty:
                    break
                    
            return messages
            
        def barrier(self, name, n_participants):
            with self.lock:
                if name not in self.barriers:
                    self.barriers[name] = threading.Barrier(n_participants)
            self.barriers[name].wait()
    
    comm = SimpleCommunicator(n_sensors)
    
    def sensor_iteration(sensor_id, comm, iteration):
        logger.info(f"Sensor {sensor_id} starting iteration {iteration}")
        
        # Send to neighbors (simplified)
        neighbors = [(sensor_id + 1) % n_sensors, (sensor_id - 1) % n_sensors]
        for neighbor in neighbors:
            comm.send(sensor_id, neighbor, f"data_from_{sensor_id}")
            
        # Barrier 1
        logger.debug(f"Sensor {sensor_id} at barrier 1")
        comm.barrier(f"iter_{iteration}_phase1", n_sensors)
        
        # Receive
        messages = comm.receive_with_timeout(sensor_id, 0.1)
        logger.debug(f"Sensor {sensor_id} received {len(messages)} messages")
        
        # Barrier 2
        logger.debug(f"Sensor {sensor_id} at barrier 2")
        comm.barrier(f"iter_{iteration}_phase2", n_sensors)
        
        logger.info(f"Sensor {sensor_id} completed iteration {iteration}")
        return len(messages)
    
    # Run one iteration with all sensors
    with ThreadPoolExecutor(max_workers=n_sensors) as executor:
        futures = []
        for i in range(n_sensors):
            future = executor.submit(sensor_iteration, i, comm, 0)
            futures.append(future)
            
        results = [f.result() for f in futures]
        
    print(f"All sensors received messages: {results}")
    print(f"Pattern test {'PASSED' if all(r > 0 for r in results) else 'FAILED'}")


def profile_matrix_operations():
    """Profile the matrix operations that might be slow"""
    print("\n=== Profiling Matrix Operations ===")
    
    sizes = [10, 50, 100, 200]
    
    for n in sizes:
        # Generate random matrices
        A = np.random.randn(n, n)
        b = np.random.randn(n)
        
        # Time linear solve
        start = time.time()
        for _ in range(10):
            x = np.linalg.solve(A, b)
        solve_time = (time.time() - start) / 10
        
        # Time eigendecomposition
        start = time.time()
        for _ in range(10):
            eigvals = np.linalg.eigvalsh(A @ A.T)
        eig_time = (time.time() - start) / 10
        
        print(f"Size {n}x{n}: solve={solve_time:.4f}s, eig={eig_time:.4f}s")


if __name__ == "__main__":
    print("Debugging Threading Performance Issues")
    print("=" * 50)
    
    # Run diagnostics
    test_barrier_deadlock()
    test_queue_blocking()
    test_threadpool_overhead()
    diagnose_snl_threading()
    profile_matrix_operations()
    
    print("\n" + "=" * 50)
    print("Diagnosis complete!")
    
    print("\nLikely issues:")
    print("1. Too many barriers causing synchronization overhead")
    print("2. Queue timeout values too high, causing unnecessary waiting")
    print("3. ThreadPoolExecutor overhead for small tasks")
    print("4. Matrix operations in inner loop without caching")