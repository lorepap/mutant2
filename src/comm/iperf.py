#!/usr/bin/env python3
# TODO: is this necessary?

import os
import signal
from ast import arg
import sys
import time
import shutil
import traceback
sys.path.append('src')
from subprocess_wrappers import call, check_output, Popen


class IperfRunner():

    def __init__(self, ip: str, time: int, log: str, port: str, scheme: str) -> None:
        self.ip = ip
        self.time = time
        self.log = log
        self.scheme = scheme
        self.ps = None
        os.makedirs('tmp', exist_ok=True)
        self.pid_file = 'tmp/pid.txt'
        self.port = port

    def run(self) -> None:

        trials = 1

        ss_cmd = ['iperf3']

        if self.scheme != None and self.scheme.strip() != '':
            ss_cmd = ['iperf3', '-C', str(self.scheme)]

        while trials <= 5:
            try:
                cmd = [
                    '-c',
                    self.ip,
                    '-p',
                    str(self.port),
                    '-t',
                    str(self.time),
                    '-J',
                    '--logfile',
                    self.log
                ]

                cmd = ss_cmd + cmd

                self.ps = Popen(cmd)

                # Write the process ID to the file if the pid_file is provided
                if self.pid_file is not None:
                    print("Writing in", self.pid_file)
                    with open(self.pid_file, 'w') as f:
                        f.write(str(self.ps.pid))
                
                # Wait 'til finishes
                self.ps.wait()
                
                shutil.rmtree('tmp')

                break
            except Exception as _:
                sys.stderr.write(traceback.format_exc())
                sys.stderr.write(
                    f'Trails #{trials}: Server is still busy, trying again after a second\n')
                time.sleep(10)
                trials += 1

    def stop(self) -> None:
        if self.ps is not None and self.ps.poll() is None:
            os.killpg(os.getpgid(self.ps.pid), signal.SIGTERM)
            self.ps.wait()
            self.ps = None
            print("Iperf Runner stopped")
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)



def main():
    scheme = None if len(sys.argv) == 5 else sys.argv[5]
    # runner = IperfRunner(sys.argv[1], str(sys.argv[2]), sys.argv[3], str(sys.argv[4]), str(sys.argv[5]), scheme)
    runner = IperfRunner(sys.argv[1], str(sys.argv[2]), sys.argv[3], str(sys.argv[4]), scheme)
    runner.run()

if __name__ == '__main__':
    main()
