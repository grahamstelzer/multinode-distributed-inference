import socket
import subprocess

def run_worker_daemon(port=5050):
    """
    Simple worker node daemon that listens for incoming 'ping' commands
    from the manager and responds with 'pong'.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(1)
    print(f"[WORKER] Listening on port {port}...")

    while True:
        conn, addr = s.accept()
        print(f"[WORKER] Connection from {addr}")
        data = conn.recv(1024).decode()
        print(f"[WORKER] Received command: {data}")

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
