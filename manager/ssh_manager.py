"""
NOT implemented yet

though as of now, this is not quite needed 
using socket to send bytes via TCP(TODO: double check its TCP)
"""

import paramiko

class SSHManager:
    def __init__(self, hostname, username, key_path):
        self.hostname = hostname
        self.username = username
        self.key_path = key_path
        self.client = None

    def connect(self):
        """Establish SSH connection to a worker node."""
        key = paramiko.RSAKey.from_private_key_file(self.key_path)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"[MANAGER] Connecting to {self.hostname}...")
        self.client.connect(hostname=self.hostname, username=self.username, pkey=key)
        print(f"[MANAGER] Connected to {self.hostname}")

    def run_command(self, command):
        """Run a command on the worker node and return stdout/stderr."""
        if not self.client:
            raise ConnectionError("SSH client not connected.")
        stdin, stdout, stderr = self.client.exec_command(command)
        return stdout.read().decode(), stderr.read().decode()

    def close(self):
        """Close the SSH connection."""
        if self.client:
            self.client.close()
            print(f"[MANAGER] Disconnected from {self.hostname}")



# class SSHManager:
#     def __init__(self, hostname, port=22, username=None, password=None, key_filename=None):
#         self.hostname = hostname
#         self.port = port
#         self.username = username
#         self.password = password
#         self.key_filename = key_filename
#         self.client = None

#     def connect(self):
#         """Establish an SSH connection."""
#         try:
#             self.client = paramiko.SSHClient()
#             self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#             if self.key_filename:
#                 self.client.connect(self.hostname, port=self.port, username=self.username, key_filename=self.key_filename)
#             else:
#                 self.client.connect(self.hostname, port=self.port, username=self.username, password=self.password)
#             print("SSH connection established.")
#         except Exception as e:
#             print(f"Failed to connect: {e}")

#     def execute_command(self, command):
#         """Execute a command on the remote server."""
#         if not self.client:
#             raise Exception("SSH connection not established.")
        
#         stdin, stdout, stderr = self.client.exec_command(command)
#         return stdout.read().decode(), stderr.read().decode()

#     def close(self):
#         """Close the SSH connection."""
#         if self.client:
#             self.client.close()
#             print("SSH connection closed.")