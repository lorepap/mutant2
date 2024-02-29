import os
import yaml
import time
import numpy as np

from collection.kernel_comm import CollectionCommManager
from comm.kernel_thread import KernelRequest
from utilities import utils
from collections import deque

# Protocol mapping
PROTOCOL_MAPPING = {
    "cubic": 0,
    "hybla": 1,
    "bbr": 2,
    "westwood": 3,
    "veno": 4,
    "vegas": 5,
    "yeah": 6,
    "cdg": 7,
    "bic": 8,
    "htcp": 9,
    "highspeed": 10,
    "illinois": 11,
    "base": 12,
    "base2": 13
}


class Collector():
    """ Collector class
    The collector runs a data collection campaign by running a specific protocol for a predefined time period.
    It setup a communication with Mutant kernel module (client) to collect the traffic data (network statistics).
    The data collected are stored locally as a csv file.

    Inputs: protocol, data collection time (running_time).
    Output: csv file of data collected
    """

    def __init__(self, protocol, n_steps, log_dir='log/collection', rtt=20, bw=12, bdp_mult=1, normalize=False):
        self.bw = int(bw)
        self.rtt = int(rtt)
        self.bdp_mult = round(bdp_mult, 1) if bdp_mult < 1 else int(bdp_mult)
        
        self.cm = CollectionCommManager(log_dir_name=log_dir, rtt=rtt,
                                bw=bw, bdp_mult=bdp_mult)
        self.log_dir = log_dir
        self.protocol = protocol
        self.proto_id = PROTOCOL_MAPPING.get(protocol.lower())  # Convert to lowercase for case-insensitivity
        self.n_steps = n_steps
        self.feature_settings = utils.parse_features_config()

        self.initiated = False
        self.prev_delivered = None
        self.normalize = normalize

        self.config = utils.parse_training_config()
        self.step_wait = self.config['step_wait_seconds']
        self.num_fields_kernel = self.config['num_fields_kernel']

        self._init_communication()

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'{self.protocol}', server_log_dir=os.path.join(self.log_dir))

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
    
    def set_protocol(self):
        msg = self.cm.netlink_communicator.create_netlink_msg(
            'SENDING ACTION', msg_flags=2, msg_seq=self.proto_id)
        self.cm.netlink_communicator.send_msg(msg)
    
    def write_data(self, collected_data, path_to_file):
        data_string = ",".join([str(collected_data[key]) for key in collected_data])
        header = ",".join([str(key) for key in collected_data])
        with open(path_to_file, 'a') as f:
            # Check if the file is empty, and if so, write the header
            if f.tell() == 0:
                f.write(f"{header}\n")
            f.write(f"{data_string}\n")


    def run_collection(self):
        """ 
        TODO: we want to receive network parameters from the kernel side. In order to do that, we run a thread which is in charge of 
        communicating in real time with the kernel module. During the communication, the thread receive the "message" from the kernel 
        module, containing the network information, and store everything locally.
        """

        # Check if the file already exists
        csv_dir = f'{self.log_dir}/csv/{int(self.n_steps)}/'
        os.makedirs(csv_dir, exist_ok=True)
        path_to_file = os.path.join(csv_dir, f'{self.protocol}.bw{self.bw}.rtt{self.rtt}.bdp_mult{self.bdp_mult}.csv')
        if os.path.exists(path_to_file):
            # Empty the existing file (we overwrite the file with the new collection but same parameters)
            with open(path_to_file, 'w'):
                pass

        data_dict = {}
        collected_data = {}
        feature_names = self.feature_settings['kernel_info'] + ['reward']
        collected_data = {feature: [] for feature in feature_names}
        collected_data['reward'] = []
        # start = time.time()
        self.set_protocol() # Communicate with kernel to set the protocol
        print(f"Running {self.protocol} for {self.n_steps} steps...")

        max_rw = pow(self.bw, self.config['reward']['kappa']) / (self.rtt*10**-3)
        step_cnt = 0
        while step_cnt < self.n_steps:
            _tmp = []
            self.kernel_thread.enable()
            step_start = time.time()
            while float(time.time() - step_start) <= float(self.step_wait):
                data = self._read_data()
                data_dict = {feature: data[i] for i, feature in enumerate(feature_names)}
                data_dict['loss_rate'] = data_dict['loss_rate'] / 100
                data_dict['thruput'] = data_dict['thruput'] / 1e6 # Convert to Mbps
                # print("[DEBUG] rtt", data_dict['rtt'])
                reward = pow(abs(data_dict['thruput'] - self.config['reward']['zeta'] * data_dict['loss_rate']),  self.config['reward']['kappa']) / (data_dict['rtt']*10**-6)
                
                normalized_reward = reward / max_rw            
                data_dict['reward'] = normalized_reward if self.normalize else reward

                # Filter corrupted samples (theoretically the reward will never go beyond max_rw)
                if data_dict['reward'] <= max_rw:  # Convert the given min_rtt to us
                    _tmp.append(data_dict)
            self.kernel_thread.disable()
            
            # Averaging
            _avg_data = {feature: np.mean([d[feature] for d in _tmp]) if feature not in ('crt_proto_id', 'prev_proto_id', 'now') else _tmp[-1][feature] for feature in feature_names}
            self.write_data(_avg_data, path_to_file)
            for feature in feature_names:
                collected_data[feature].append(_avg_data[feature])
            step_cnt+=1
            print(f'[STEP {step_cnt}]', 'Reward:', 
                  _avg_data['reward'], 'Thruput (Mbps):', 
                  _avg_data['thruput'], 'Loss rate:', 
                  _avg_data['loss_rate'], 'RTT (ms):', 
                  _avg_data['rtt'])    
        print(f"Collection of {self.protocol} completed.")
        print(f"Avg thr: {round(np.mean(collected_data['thruput']), 4)} Mbps, \
              Avg loss rate: {round(np.mean(collected_data['loss_rate']), 4)}, \
              Avg RTT: {round(np.mean(collected_data['rtt'])*10**-3, 4)}ms, \
              Avg reward: {round(np.mean(collected_data['reward']), 4)}")
