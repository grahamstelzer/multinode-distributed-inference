# monitor.py
# Responsible for tracking worker nodes' availability and GPU memory (pseudo)

import json

class Monitor:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            config = json.load(f)
        # store list of worker dicts with IPs and ports
        self.workers = config["workers"]
    
    def get_ready_workers(self):
        # Return list of IP:port tuples for ready workers
        return [(w["ip"], w["port"]) for w in self.workers if True] 
    
    def heartbeat(self):
        """
        Periodically check status of workers.
        Currently pseudo: just print statuses
        """
        pass
        # for worker in self.workers:
        #     print(f"[MONITOR] {worker['name']} status: {worker['status']}")

    def get_ready_workers(self):
        """
        Return a list of available workers
        """
        return [w for w in self.workers if w['status'] == 'ready']
