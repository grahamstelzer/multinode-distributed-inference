# job_dispatcher.py
# Responsible for sending tasks to workers over network and receiving results

class JobDispatcher:
    def __init__(self):
        # Could store connections to worker daemons if desired
        self.connections = {}
    
    def dispatch_job(self, worker_ip, job_chunk):
        """
        Send a chunk of pseudo data to the worker daemon.
        For now, we just print debug info.
        """
        print(f"[DISPATCHER] Sending {job_chunk} to worker {worker_ip}")
        # Later: serialize tensor and send via comms.py
    
    def receive_result(self, worker_ip, job_chunk):
        """
        Receive result back from worker.
        For now: just return a placeholder
        """
        return f"result_of_{job_chunk}"
