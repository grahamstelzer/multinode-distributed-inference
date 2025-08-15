import torch
import socket
import pickle
from monitor import Monitor


def send_tensor_chunk(worker_ip, worker_port, tensor_chunk):
    """
    Connect to worker, send tensor, and receive result robustly
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((worker_ip, worker_port))
        # Serialize tensor and send length prefix
        data = pickle.dumps({'tensor': tensor_chunk.numpy()})
        data_len = len(data)
        s.sendall(data_len.to_bytes(8, byteorder='big'))
        s.sendall(data)

        # Receive length prefix first
        length_bytes = s.recv(8)
        if len(length_bytes) < 8:
            raise RuntimeError("Failed to receive result length")
        result_len = int.from_bytes(length_bytes, byteorder='big')

        # Read exactly result_len bytes
        result_bytes = b""
        while len(result_bytes) < result_len:
            packet = s.recv(min(4096, result_len - len(result_bytes)))
            if not packet:
                raise RuntimeError("Connection closed before all result received")
            result_bytes += packet

        # Deserialize
        result = pickle.loads(result_bytes)
        return torch.tensor(result['tensor'])

def main():
    monitor = Monitor(config_path="config.json")
    ready_workers = monitor.get_ready_workers()
    print("[POC] Ready workers (IP, port):", ready_workers)

    # Create a small tensor for testing
    x = torch.arange(0, 20, dtype=torch.float32).reshape(2, 10)
    print("[POC] Original tensor:\n", x)

    # Split tensor into chunks based on number of workers
    chunks = torch.chunk(x, len(ready_workers), dim=0)
    results = []

    for i, chunk in enumerate(chunks):
        worker = ready_workers[i]
        worker_ip = worker["ip"]
        worker_port = worker["port"]
        print(f"[POC] Sending chunk {i} to {worker_ip}:{worker_port}")
        result = send_tensor_chunk(worker_ip, worker_port, chunk)
        results.append(result)

    final_result = torch.cat(results, dim=0)
    print("[POC] Final result:\n", final_result)

if __name__ == "__main__":
    main()
