#!/usr/bin/env python
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import time
import signal
import threading

# Set environment variables if .env file exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env loading")

# Function to run a command and continuously output its results
def run_and_output(command, cwd=None):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        cwd=cwd
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.poll()

# Commands to run
commands = [
    {"cmd": "uvicorn main:app --host 0.0.0.0 --port 8000 --reload", "cwd": "backend", "name": "Backend"},
    {"cmd": "streamlit run app.py", "cwd": "frontend", "name": "Frontend"}
]

# Create a event to signal processes to stop
stop_event = threading.Event()

# Set signal handlers
def signal_handler(sig, frame):
    print("\nStopping all services...")
    stop_event.set()
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Function to run in threads
def run_service(cmd_info):
    try:
        print(f"Starting {cmd_info['name']}...")
        run_and_output(cmd_info["cmd"], cmd_info["cwd"])
    except Exception as e:
        print(f"Error running {cmd_info['name']}: {e}")
    finally:
        if not stop_event.is_set():
            print(f"{cmd_info['name']} stopped unexpectedly")

# Main execution
if __name__ == "__main__":
    print("Starting all services...")
    
    # Make sure directories exist
    for cmd_info in commands:
        if not os.path.exists(cmd_info["cwd"]):
            print(f"Directory {cmd_info['cwd']} doesn't exist!")
            sys.exit(1)
    
    # Run commands in parallel
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_service, cmd_info) for cmd_info in commands]
        
        # Wait for stop event or completion
        try:
            while not stop_event.is_set() and any(not f.done() for f in futures):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping services...")
            stop_event.set()
        
    print("All services stopped") 