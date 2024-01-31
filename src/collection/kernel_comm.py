import os
from subprocess import call
from comm.comm import CommManager
from utilities import context

INIT_COMM_FLAG = 1

class CollectionCommManager(CommManager):
    def __init__(self, log_dir_name='log/iperf', client_time=None, rtt=20, bw=12, bdp_mult=1):
        super().__init__(log_dir_name=log_dir_name, client_time=client_time, 
                         rtt=rtt, bw=bw, bdp_mult=bdp_mult)


    def init_kernel_communication(self):
        print("[COLLECTION MANAGER] Initiating communication...")

        msg = self.netlink_communicator.create_netlink_msg(
            'INIT_COMMUNICATION', msg_flags=INIT_COMM_FLAG)
 
        # Send init communication message to kernel: response is handled by kernel thread
        self.netlink_communicator.send_msg(msg)

        print("[COLLECTION MANAGER] Communication initiated")
