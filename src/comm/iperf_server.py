import os
import subprocess
import psutil
import threading

class IperfServer:
    def __init__(self, log_file, port=None):
        self._server_proc = None
        self._server_log_file = log_file
        self._port = port

    def start(self):
        if self._server_proc is not None:
            raise Exception("Server already running")
        print("[SERVER] Server log:", self._server_log_file)

         # Remove or truncate the log file to start with an empty file
        if os.path.exists(self._server_log_file):
            os.remove(self._server_log_file)
        if self._port != None:
            server_cmd = ["iperf3", "-s", "-p", str(self._port), "--logfile", self._server_log_file]
        else:
            server_cmd = ["iperf3", "-s", "-p", "5201", "--logfile", self._server_log_file]
        
        try:
            self._server_proc = subprocess.Popen(server_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        except Exception as e:
            # Kill all iperf3 processes if an error is raised
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'iperf3':
                    proc.kill()
            raise e

        print("Server started with PID:", self._server_proc.pid)

    def _capture_output(self, stream):
        for line in stream:
            print(line, end='')
    
    def stop(self):
        if self._server_proc is not None:
            self._server_proc.terminate()
            self._server_proc.wait()
            self._server_proc = None
            print("Server stopped")
        else:
            print("Server not running")

    def restart(self):
        self.stop()
        self.start()
