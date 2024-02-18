import yaml
import time
import random
from test_switching.kernel_comm import TestCommManager
from comm.kernel_thread import KernelRequest

ACTION_FLAG = 2

CUBIC = 0
HYBLA = 1
BBR = 2
WESTWOOD = 3
VENO = 4
VEGAS = 5
YEAH = 6
CDG = 7
BIC = 8
HTCP = 9
HIGH_SPEED = 10
ILLINOIS = 11

class SwitchingTestRunner():
    """ Collector class
    The collector runs a data collection campaign by running a specific protocol for a predefined time period.
    It setup a communication with Mutant kernel module (client) to collect the traffic data (network statistics).
    The data collected are stored locally as a csv file.

    Inputs: protocol, data collection time (running_time).
    Output: csv file of data collected
    """

    def __init__(self, protocol='mutant', running_time=10):
        self.cm = TestCommManager(protocol, 'log/test_switching', client_time=running_time) #iperf_dir, time
        self.proto = protocol
        self.running_time = running_time
        # TODO: handle the params with a config file
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)

        self.num_fields_kernel = config['num_fields_kernel']
        self.initiated = False
        self._init_communication()

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag='test', server_log_dir='log/collection')

    def stop_communication(self):
        self.cm.stop_iperf_communication()
        self.cm.close_kernel_communication()
        self.kernel_thread.exit()

    def _init_communication(self):
        # Start thread to communicate with kernel

        if not self.initiated:
            print("Start kernel thread...")

            # Thread for kernel info
            self.kernel_thread = KernelRequest(
                self.cm.netlink_communicator, self.num_fields_kernel)

            self.kernel_thread.start()

            print("Kernel thread started.")
            self.initiated = True

    def _read_data(self):
        kernel_info = self.kernel_thread.queue.get()
        self.kernel_thread.queue.task_done()
        return kernel_info
    
    # def _recv_data(self):
    #     msg = self.cm.netlink_communicator.recv_msg()
    #     data = self.cm.netlink_communicator.read_netlink_msg(msg)
    #     split_data = data.decode(
    #         'utf-8').split(';')[:self.num_fields_kernel]
    #     return list(map(int, split_data))

    def run_collection(self):
        """ 
        TODO: we want to receive network parameters from the kernel side. In order to do that, we run a thread which is in charge of 
        communicating in real time with the kernel module. During the communication, the thread receive the "message" from the kernel 
        module, containing the network information, and store everything locally.
        """

        collected_data = {}
        cca_list = [CUBIC, HYBLA, BBR, WESTWOOD, VENO, VEGAS, YEAH, CDG, BIC, HTCP, HIGH_SPEED, ILLINOIS]
        current_cca_index = 0
        step_time = 1 # waiting time in seconds between each switch
        start_exp = time.time()
        while time.time() - start_exp < self.running_time:
            start = time.time()
            # Every time we send a random protocol
            current_cca = random.choice(cca_list)
            print(f"Switch to {current_cca}")
            msg = self.cm.netlink_communicator.create_netlink_msg(
                'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=current_cca)
            self.cm.netlink_communicator.send_msg(msg)
            while time.time() - start < step_time:
                data = self._read_data()
                # print("Data:", data)
                print("Data:", ", ".join(str(x) for x in data))
