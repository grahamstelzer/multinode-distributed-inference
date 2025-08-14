# monitor.py
# Responsible for tracking worker nodes' availability and GPU memory (pseudo)

class Monitor:
    def __init__(self):
        # Dictionary of workers and their status
        self.workers = {
            'worker1': 'ready',
            'worker2': 'ready',
            'manager_gpu': 'ready'
        }
    
    def heartbeat(self):
        """
        Periodically check status of workers.
        Currently pseudo: just print statuses
        """
        for worker, status in self.workers.items():
            print(f"[MONITOR] {worker} status: {status}")
    
    def get_ready_workers(self):
        """
        Return a list of available workers
        """
        return [w for w, s in self.workers.items() if s == 'ready']
