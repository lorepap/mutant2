"""
Author: Lorenzo Pappone
Year: 2023

Iperf Client Thread

In order to execute iperf in the context of mahimahi we run a python script after calling mm-link.
The python script run the iperf3 client with parameters.
To kill the process, the iperf runner in the script store its pid in a file and this client kills it
when the training is over.
"""

import os
import sys
import re
import threading
from concurrent.futures import thread
import traceback, signal

import utilities.utils as utils
import utilities.context as context
from subprocess_wrappers import Popen, call, check_output, print_output, check_call
import subprocess

class IperfClient(threading.Thread):

    def __init__(self, time=86400, log_file=None, rtt=20, bw=12, q_size=100, bw_factor=1) -> None:
        threading.Thread.__init__(self)

        self.ip = self._get_private_ip()
        self.time = time
        self.log_file = log_file
        self.ps = None
        self.rtt = rtt
        self.bw = bw
        self.bw_factor = bw_factor
        # self.bdp_mult = bdp_mult
        self.q_size = q_size
        # self._pid_file = "src/tmp/pid.txt"
        # os.makedirs(self._pid_file, exist_ok=True)

    def _get_private_ip(self):
        """
        Returns the private IP address of the host machine.
        """
        # Run the ifconfig command and capture the output
        output = subprocess.check_output(['ifconfig']).decode('utf-8')
        
        # Search the output for the private IP address using a regular expression
        # pattern = r'inet (192(?:\.\d{1,3}){2}\.\d{1,3})'
        pattern = r'inet (?:addr:)?(10\.0\.2\.15)'       
        match = re.search(pattern, output)

        if match:
            # If a match is found, return the IP address
            ip_address = match.group(1)
        else:
            # If no match is found, set the IP address to None
            ip_address = None
        return ip_address

    def _ip_forwarding_set(self) -> bool:
        cmd = ['sysctl', 'net.ipv4.ip_forward']

        res = check_output(cmd)

        val = res.strip().decode('utf-8')

        return val == 'net.ipv4.ip_forward = 1'

    def _set_ip_forwarding(self):

        if self._ip_forwarding_set():
            print('IP forwarding is already set\n')
            return

        cmd = ['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1']
        res = call(cmd)

        if res != 0:
            raise Exception("Unable to set ipv4 forwarding")

    def _get_mahimahi_cmd(self):

        # bdp = int((self.rtt/2 * self.bw)/8) # Convert bits to bytes
        # Compute the queue size in packets (1500 is the MTU; the bdp is expressed in bytes)
        # q_size = (self.bdp_mult * bdp) // 1500
        # Print client comm parameters
        
        if self.bw_factor == 1:
            trace_d = f'wired{int(self.bw)}'
            trace_u = trace_d
        else: 
            trace_d = f'wired{int(self.bw)}-{self.bw_factor}x-d'
            trace_u = f'wired{int(self.bw)}-{self.bw_factor}x-u'
        print(f"[IPERF CLIENT] Mahimahi network scenario:\n rtt(ms) = {self.rtt}\n bw(Mbps) = {self.bw}\n q_size (pkts) = {self.q_size}\n bw_factor = {self.bw_factor}\n")
        print(f"[IPERF CLIENT] Mahimahi traces:\n D: {trace_d}\n U: {trace_u}\n")
        
        cmd = ['mm-delay', str(int(self.rtt/2)),
               'mm-link', 
               f'{context.entry_dir}/traces/{trace_u}',
               f'{context.entry_dir}/traces/{trace_d}',
                '--uplink-queue=droptail',
                f'--uplink-queue-args="packets={self.q_size}"',
                '--downlink-queue=droptail', 
                f'--downlink-queue-args="packets={self.q_size}"'
               ]
        # print(cmd)
        return cmd

    def _get_iperf_cmd(self):

        return [
            'python3',
            'src/comm/iperf.py', #TODO: check if calling iperf.py is necessary
            self.ip,
            str(self.time),
            self.log_file,
            '5201'
        ]
    

    def run(self) -> None:
        try:

            self._set_ip_forwarding()

            cmd = self._get_mahimahi_cmd() + self._get_iperf_cmd()
            # print("[DEBUG] Command executing:", cmd)

            check_call(cmd)

            print("mahimahi experiment ran successfully\n")

        except Exception as _:
            print('\n')
            print(traceback.format_exc())

    # This stop function is set when total episodes time in training > iperf time, so I need to stop manually the client when training is finished
    # This temporarily solution let me run the training with a specific number of episodes (and steps per episodes), while iperf is always running
    # TODO: iperf process should check the moderator which could be stopped by the RL-module.
    # def stop(self) -> None:
    #     # Read the PID from the file
    #     if os.path.exists(self._pid_file):
    #         with open(self._pid_file, 'r') as f:
    #             # print("Getting pid from", self._pid_file)
    #             pid = int(f.read().strip())
    #             # print("Client PID:", pid)
    #         # Kill the client process if moderator is on
    #             if not(self.moderator.is_stopped()):
    #                 os.kill(pid, signal.SIGTERM)
    #                 print("Iperf client killed")
    #             # Remove the pid.txt file
    #             os.remove(self._pid_file)
        
