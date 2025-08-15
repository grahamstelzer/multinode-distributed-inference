import socket
import torch

from model_loader import ModelLoader
from tensor_parallel_runner import TensorParallelRunner
from comms import send_result, receive_task

def run_worker_daemon(port=5050):
    """
    Worker daemon: listens for incoming tensor chunks, processes them,
    and sends results back to manager.
    """
    # Load model (PoC: pseudo model)
    model_loader = ModelLoader()
    model = model_loader.load_model("pseudo_model")  # replace with real model later
    runner = TensorParallelRunner(model)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow quick restart
    s.bind(("0.0.0.0", port))
    s.listen(5)
    print(f"[WORKER] Listening on port {port}...")

    try:
        while True:
            conn, addr = s.accept()
            print(f"[WORKER] Connection from {addr}")

            try:
                # Receive tensor chunk
                task = receive_task(conn)
                tensor_chunk = torch.tensor(task['tensor']).to("cuda:0")
                print(f"[WORKER] Received tensor chunk of shape {tensor_chunk.shape}")
                print(f"[WORKER] Tensor chunk: {tensor_chunk}")

                # Process tensor (PoC: multiply by 2)
                result_tensor = runner.run(tensor_chunk)

                # Send result back
                send_result(conn, result_tensor.cpu().numpy())
                print(f"[WORKER] Sent result back to manager")

            except Exception as e:
                print(f"[WORKER] Error during processing: {e}")
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("[WORKER] Shutting down daemon")
    finally:
        s.close()


if __name__ == "__main__":
    run_worker_daemon()