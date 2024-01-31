import os
from subprocess import call
from comm.comm import CommManager
from utilities import context

INIT_COMM_FLAG = 1

class TestCommManager(CommManager):
    def __init__(self, protocol, log_dir_name='log/iperf', client_time=None):
        self.proto = protocol
        super().__init__(log_dir_name=log_dir_name, client_time=client_time)


    def init_kernel_communication(self):
        print("[COLLECTION MANAGER] Initiating communication...")

        msg = self.netlink_communicator.create_netlink_msg(
            'INIT_COMMUNICATION', msg_flags=INIT_COMM_FLAG)
 
        # Send init communication message to kernel: response is handled by kernel thread
        self.netlink_communicator.send_msg(msg)

        print("[COLLECTION MANAGER] Communication initiated")

    def init_proto(self):

        if self.is_kernel_initialized():
            print('Kernel module has already been initialized\n')
            return

        # Convert the input protocol name to lowercas and appends "_mod"
        # protocol_mod = self.proto.lower() + '_mod' 
        # no need to append _mod since its handled in the insert_proto script

        cmd = os.path.join('/home/lorenzo/Desktop/mutant', 
            'src/test_switching/insert_proto.sh')

        # make script runnable
        res = call(['chmod', '755', cmd])
        if res != 0:
            raise Exception('Unable to set run permission\n')

        res = call([cmd, self.proto])
        if res != 0:
            raise Exception(f'Unable to init {self.proto} \n')