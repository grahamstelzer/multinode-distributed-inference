import socket
import subprocess
import socket
from model_loader import ModelLoader
from tensor_parallel_runner import TensorParallelRunner
from comms import send_result, receive_task

def run_worker_daemon(port=5050):
    """
    Simple worker node daemon that listens for incoming 'ping' commands
    from the manager and responds with 'pong'.
    """

    model_loader = ModelLoader()
    model = model_loader.load_model("pseudo_model")
    runner = TensorParallelRunner(model)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    print(f"[WORKER] Listening on port {port}...")

    while True:
        conn, addr = s.accept()
        print(f"[WORKER] Connection from {addr}")
        data = conn.recv(1024).decode()
        print(f"[WORKER] Received command: {data}")



        # Receive a pseudo task
        task = receive_task(conn)
        chunk = task.get('chunk', None)
        print(f"[WORKER] Received task chunk: {chunk}")

        # Run tensor operation
        result = runner.run(chunk)

        # Send result back
        send_result(conn, result)



        if data.strip() == "shutdown":
            conn.sendall(b"shutting down\n")
            conn.close()
            break

        if data.strip() == "ping":
            conn.sendall(b"pong\n")
        elif data.strip() == "run date":
            # Example: run local shell command
            output = subprocess.check_output("date", shell=True)
            conn.sendall(output)
        else:
            conn.sendall(b"unknown command\n")

        conn.close()

if __name__ == "__main__":
    run_worker_daemon()
