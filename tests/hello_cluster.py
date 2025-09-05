# test ssh connection
# this can be used to ssh into worker node and run commands, ex: starting the daemon with python worker/worker_daemon.py
# or running a command like `date` to test the connection


import paramiko
import getpass

def run_remote_command(host, username, password, command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown hosts
    ssh.connect(hostname=host, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command(command)
    print(f"--- Output from {host} ---")
    print(stdout.read().decode())
    print(stderr.read().decode())
    ssh.close()

if __name__ == "__main__":
    # Worker details
    worker_host = "..."
    worker_user = "..."

    # Ask for password at runtime
    worker_pass = getpass.getpass(f"Enter password for {worker_user}@{worker_host}: ")

    # Run test command
    run_remote_command(worker_host, worker_user, worker_pass, "echo Hello World")
