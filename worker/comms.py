# comms.py
# Serialize and send/receive tensors over TCP robustly

import pickle

def send_result(conn, tensor_result):
    """
    Serialize tensor and send to manager
    """
    data = pickle.dumps({'tensor': tensor_result})
    # send length first so manager knows how many bytes to expect
    data_len = len(data)
    conn.sendall(data_len.to_bytes(8, byteorder='big'))  # 8-byte length prefix
    conn.sendall(data)

def receive_task(conn):
    """
    Receive a serialized tensor task from manager
    """
    # read length first
    length_bytes = conn.recv(8)
    if len(length_bytes) < 8:
        raise RuntimeError("Failed to read task length")
    data_len = int.from_bytes(length_bytes, byteorder='big')

    # now read exactly data_len bytes
    data = b""
    while len(data) < data_len:
        packet = conn.recv(min(4096, data_len - len(data)))
        if not packet:
            raise RuntimeError("Connection closed before all data received")
        data += packet

    task = pickle.loads(data)
    return task
