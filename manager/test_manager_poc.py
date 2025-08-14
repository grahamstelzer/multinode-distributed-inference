# test_manager_poc.py
# This script ties together scheduler, dispatcher, and monitor
# Simulates a small tensor-parallel task across manager + pseudo worker nodes

from scheduler import Scheduler
from job_dispatcher import JobDispatcher
from monitor import Monitor

def main():
    # Initialize manager components
    monitor = Monitor()
    scheduler = Scheduler()
    dispatcher = JobDispatcher()
    
    # Step 1: Monitor reports ready workers
    ready_workers = monitor.get_ready_workers()
    print("[POC] Ready workers:", ready_workers)
    
    # Step 2: Create a pseudo job (e.g., tensor of shape 10x10)
    pseudo_job = {'job_id': 1, 'tensor_shape': (10, 10), 'operation': 'double'}
    scheduler.add_job(pseudo_job)
    
    # Step 3: Scheduler splits job into chunks
    scheduler.assign_jobs()
    print("[POC] Job chunks:", pseudo_job['chunks'])
    
    # Step 4: Dispatch chunks to workers
    for i, chunk in enumerate(pseudo_job['chunks']):
        worker_ip = ready_workers[i]  # simple round-robin assignment
        dispatcher.dispatch_job(worker_ip, chunk)
    
    # Step 5: Collect results
    results = []
    for i, chunk in enumerate(pseudo_job['chunks']):
        worker_ip = ready_workers[i]
        result = dispatcher.receive_result(worker_ip, chunk)
        results.append(result)
    
    print("[POC] Collected results:", results)
    
    # Step 6: Mark job complete
    scheduler.collect_results()
    print("[POC] Job status:", pseudo_job['status'])

if __name__ == "__main__":
    main()
