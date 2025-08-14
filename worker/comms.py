# comms.py
# Low-level network communication utilities

import socket
import pickle

def send_result(conn, result):
    """
    Serialize and send result back to manager
    """
    data = pickle.dumps(result)
    conn.sendall(data)

def receive_task(conn):
    """
    Receive a task from manager and deserialize
    """
    data = conn.recv(4096)
    task = pickle.loads(data)
    return task
