"""
manager to workers proof of concept test

just makes a small tensor, splits it, sends to workers, collects results

manager methodology:
- monitor.py: find ready workers
- scheduler.py: split tensor into chunks
- job_dispatcher.py: send chunks to workers, receive results
- comms.py: handle serialization and network communication

worker:
- worker_daemon.py: listens for incoming tensor chunks, processes them, sends results back
- in this case, loaded dummy model via "model_loader.py" and ran a doubling calculation through
  "tensor_parallel_runner.py"

"""





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
    # get ready workers
    monitor = Monitor(config_path="config.json")
    ready_workers = monitor.get_ready_workers()
    print("[POC] Ready workers (IP, port):", ready_workers)

    # make small tensor
    x = torch.arange(0, 20, dtype=torch.float32).reshape(2, 10)
    print("[POC] Original tensor:\n", x)

    # chunk for each worker
    chunks = torch.chunk(x, len(ready_workers), dim=0)
    results = []

    # send each chunk to a different worker
    # NOTE: this could use optimization, since this loop runs serially
    # TODO: probably just parallize with threads or async IO
    for i, chunk in enumerate(chunks):
        worker = ready_workers[i]
        worker_ip = worker["ip"]
        worker_port = worker["port"]
        print(f"[POC] Sending chunk {i} to {worker_ip}:{worker_port}")
        result = send_tensor_chunk(worker_ip, worker_port, chunk)
        results.append(result)

    # concatenate results
    # NOTE: this will likely be an all-gather operation in a real system
    final_result = torch.cat(results, dim=0)
    print("[POC] Final result:\n", final_result)

if __name__ == "__main__":
    main()
