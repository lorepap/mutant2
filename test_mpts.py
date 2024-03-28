import os
# from src.mab.encoding_network import EncodingNetwork
from src.comm.kernel_thread import KernelRequest
from src.comm.netlink_communicator import NetlinkCommunicator
import src.utilities.utils as utils
from src.comm.comm import CommManager

import time
import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from argparse import ArgumentParser 

from src.utilities.feature_extractor import FeatureExtractor
from src.utilities.logger import Logger
from src.utilities import context
from src.utilities.change_detection import PageHinkley, ADWIN
from src.mab.mpts import MPTS

from collections import deque

class KernelThread(KernelRequest):
        def __init__(self, comm_manager, num_fields_kernel):
                super().__init__(comm_manager.netlink_communicator, num_fields_kernel)
                self._comm_manager: CommManager = comm_manager
                self._setup_communication()
                self._init_communication()

        def _init_communication(self):
                print("Initiating communication...")
                self.start()
                print("Communication initiated")
    
        def _setup_communication(self):
                # Set up iperf client-server communication
                # Now a single flow between client and server is running
                # We can now set up the runner and start training the RL model    
                self._comm_manager.init_kernel_communication()
                self._comm_manager.start_communication(client_tag='test', server_log_dir='log/collection')


def compute_reward(kappa, zeta, thr, loss_rate, rtt):
        # Reward is normalized if the normalize_rw is true, otherwise max_rw = 1
        return (pow(abs((thr - zeta * loss_rate)), kappa) / (rtt*10**-3) )  # thr in Mbps; rtt in s
    

def set_initial_protocol(netlink_comm, map_proto):
        """
          Set the initial protocol for the next reset.
          This action is necessary only if the _reset() function shouldn't be overridden.
        """
        msg = netlink_comm.create_netlink_msg(
                'SENDING ACTION', msg_flags=2, msg_seq=int(map_proto[0]))
        netlink_comm.send_msg(msg)

            
if __name__ == "__main__":
        parser = ArgumentParser()
        # Accept a list of policies to be used in the environment - if it's not passed, use all of them
        parser.add_argument('--rtt', '-r', default=20, type=int)
        parser.add_argument('--bw', '-b', default=12, type=int)
        parser.add_argument('--bdp_mult', '-q', default=1, type=int)
        parser.add_argument('--bw_factor', '-f', default=1, type=int)
        args = parser.parse_args()

        # Config
        config = utils.parse_training_config()
        proto_config = utils.parse_protocols_config()
        feature_settings = utils.parse_features_config()
        policies = proto_config.keys()
        
        # Reward
        zeta = config['reward']['zeta']
        kappa = config['reward']['kappa']

        # Change detection
        detector = ADWIN(delta=1e-8)
        
        # Communication setup (comm manager + kernel thread)
        logdir = 'test_mpts/log'
        comm_manager = CommManager(log_dir_name=logdir, rtt=args.rtt, bw=args.bw, bdp_mult=args.bdp_mult, bw_factor=args.bw_factor) #iperf_dir, time
        k_thread = KernelThread(comm_manager, config['num_fields_kernel'])

        # Loop
        step_wait = config['step_wait_seconds']
        step_cnt = 0
        
        map_proto = {i: proto_config[p]['id'] for i, p in enumerate(policies)}
        # MPTS
        mpts = MPTS(arms=map_proto, k=2, T=100, thread=k_thread, net_channel=comm_manager.netlink_communicator)
        set_initial_protocol(comm_manager.netlink_communicator, map_proto)
        mpts.initialize_protocols()
        while step_cnt < 1500:
                arms = mpts.mpts()      
                # Expected outcome: the arms selected should be the same for the same trace
                print(f"Selected arms: {arms}")
                step_cnt+=1
        
        comm_manager.stop_iperf_communication()
        # comm_manager.close_kernel_communication()