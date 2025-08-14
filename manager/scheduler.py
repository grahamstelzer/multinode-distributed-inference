# scheduler.py
# Responsible for splitting tasks into subtasks, assigning worker nodes, and collecting results

class Scheduler:
    def __init__(self):
        # Initialize job queue and list of available workers
        self.job_queue = []
        self.worker_registry = {}  # Worker IPs and status
    
    def add_job(self, job):
        """
        Add a pseudo job (e.g., a large tensor) to the queue.
        job: dict with job_id, tensor_shape, operation, etc.
        """
        self.job_queue.append(job)
    
    def assign_jobs(self):
        """
        Decide which worker gets which chunk of a job.
        For now, we'll just split the tensor into two pieces as a proof of concept.
        """
        # Example pseudo splitting
        for job in self.job_queue:
            job['chunks'] = ['chunk1', 'chunk2']  # Placeholder for tensor chunks
    
    def collect_results(self):
        """
        After jobs are dispatched, collect results from worker nodes.
        Currently pseudo: just mark as done
        """
        for job in self.job_queue:
            job['status'] = 'completed'
