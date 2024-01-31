import os
import sys
import traceback
from typing import Any
from subprocess_wrappers import call, Popen, check_output, print_output
# from moderator import Moderator
from comm.iperf_client import IperfClient
from comm.iperf_server import IperfServer
import utilities.utils as utils
from comm.netlink_communicator import NetlinkCommunicator
from utilities import context

TEST_FLAG = 4
INIT_SINGLE_PROT_TEST = 3
ACTION_FLAG = 2
INIT_COMM_FLAG = 1
END_COMM_FLAG = 0

# Base class for Trainer
class CommManager():

    def __init__(self, log_dir_name='log/iperf', client_time=None, rtt=20, bw=12, bdp_mult=10) -> None:
        
        self.time = client_time if client_time else 86400
        self.log_dir = log_dir_name
        self.min_rtt = rtt
        self.bw = bw
        # Compute the q_size (n. of packets)
        bdp = bw * rtt # Mbits
        mss = 1488 # bytes
        self.q_size = int(bdp_mult * bdp * 10**3 / (8*mss)) # packets

        self.init_proto()

        self.netlink_communicator = NetlinkCommunicator()
        self.client: IperfClient = None
        self.server: IperfServer = None
        # self.moderator: Moderator = Moderator(self.args.iperf == 1)

    def is_kernel_initialized(self) -> bool:
        cmd = ['cat', '/proc/sys/net/ipv4/tcp_congestion_control']

        res = check_output(cmd)

        protocol = res.strip().decode('utf-8')

        return protocol == 'mutant'

    def init_proto(self):

        if self.is_kernel_initialized():
            print('Kernel has already been initialized\n')
            return

        cmd = os.path.join(context.entry_dir, 'scripts/init_kernel.sh') 

        # make script runnable
        res = call(['chmod', '755', cmd])
        if res != 0:
            raise Exception('Unable to set run permission\n')

        res = call(cmd)
        if res != 0:
            raise Exception('Unable to init kernel\n')
        
    def start_server(self, server_log_dir='log/iperf'):
        base_path = os.path.join(context.entry_dir, server_log_dir)
        filename = f'server.log'
        log_filename = f'{base_path}/{filename}'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        self.server = IperfServer(log_filename)
        self.server.start()


    def start_client(self, tag: str) -> str:

        base_path = os.path.join(context.entry_dir, self.log_dir)
        utils.check_dir(base_path)

        filename = f'{tag}.{utils.time_to_str()}.json'
        
        log_filename = f'{base_path}/json/{filename}'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        self.client = IperfClient(self.time, log_filename, self.min_rtt, self.bw, self.q_size)

        try:
            self.client.start()
        # Continue with the main program logic here
        except RuntimeError as e:
            print(f"Error in IperfClient thread: {e}")
            # Terminate the main program or perform necessary cleanup
            sys.exit(1)
        return log_filename
    
    
    def start_communication(self, client_tag, server_log_dir='log/iperf'):
        self.start_server(server_log_dir)
        self.start_client(client_tag)
    
    def change_iperf_logfile_name(old_name: str, new_name: str) -> None:
        try:
            new_file = new_name.replace("csv", "json")
            os.rename(old_name, new_file)

        except Exception as _:
            print('\n')
            print(traceback.print_exc())

    def init_kernel_communication(self):
        print("Initiating communication...")

        msg = self.netlink_communicator.create_netlink_msg(
            'INIT_COMMUNICATION', msg_flags=INIT_COMM_FLAG)
 
        self.netlink_communicator.send_msg(msg)

        print("Communication initiated")

        # total_choices, total_prots = utils.get_number_of_actions(
        #     self.netlink_communicator)

        # print(f'\n\n----- Number of protocols available in kernel is {total_choices} ----- \n\n')

    def close_kernel_communication(self) -> None:

        msg = self.netlink_communicator.create_netlink_msg(
            'END_COMMUNICATION', msg_flags=END_COMM_FLAG)

        self.netlink_communicator.send_msg(msg)
        self.netlink_communicator.close_socket()

        print("Communication closed")

    def stop_iperf_communication(self):
        # Client stops by itself?
        self.server.stop()

    def create_netlink_msg(self, data, msg_type=0, msg_flags=0, msg_seq=0, msg_pid=os.getpid()):
        self.netlink_communicator.create_netlink_msg(data, msg_type, msg_flags, msg_seq, msg_pid=os.getpid())