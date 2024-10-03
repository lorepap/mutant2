import os
import yaml
import time
import numpy as np

from collection.kernel_comm import CollectionCommManager
from comm.kernel_thread import KernelRequest
from utilities import utils
from collections import deque
from src.utilities.feature_extractor import FeatureExtractor

from utilities.logger import Logger
from utilities.utils import context

import sys

# Protocol mapping
PROTOCOL_MAPPING = {
    "cubic": 0,
    "hybla": 1,
    "bbr": 2,
    "westwood": 3,
    "veno": 4,
    "vegas": 5,
    "yeah": 6,
    # "cdg": 7,
    "bic": 8,
    "htcp": 9,
    "highspeed": 10,
    "illinois": 11,
    # "base": 12,
    # "base2": 13
}


class Collector():
    """ Collector class
    The collector runs a data collection campaign by running a specific protocol for a predefined time period.
    It setup a communication with Mutant kernel module (client) to collect the traffic data (network statistics).
    The data collected are stored locally as a csv file.

    Inputs: protocol, data collection time (running_time).
    Output: csv file of data collected
    """

    def __init__(self, protocol, n_steps, log_dir='log/collection', rtt=20, bw=12, bdp_mult=1, bw_factor=1, normalize=False, log_mahimahi=True):
        self.bw = int(bw)
        self.rtt = int(rtt)
        self.bdp_mult = round(bdp_mult, 1) if bdp_mult < 1 else int(bdp_mult)
        self.bw_factor = bw_factor

        self.cm = CollectionCommManager(log_dir_name=log_dir, rtt=rtt,
                                bw=bw, bdp_mult=bdp_mult, bw_factor=bw_factor, log_mahimahi=log_mahimahi)
        self.log_dir = log_dir
        self.protocol = protocol
        self.proto_id = PROTOCOL_MAPPING.get(protocol.lower())  # Convert to lowercase for case-insensitivity
        self.n_steps = n_steps
        self.feature_settings = utils.parse_features_config()

        self.initiated = False
        self.prev_delivered = None
        self.normalize = normalize

        self.config = utils.parse_training_config()
        self.proto_config = utils.parse_protocols_config()
        self.step_wait = self.config['step_wait_seconds']
        self.num_fields_kernel = self.config['num_fields_kernel']
        
        # Feature extractor
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.window_sizes = self.feature_settings['window_sizes']
        # self.training_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.training_features = utils.get_training_features(all_features=self.non_stat_features, 
                                stat_features=self.stat_features, pool_size=len(self.proto_config))
        self.feat_extractor = FeatureExtractor(self.stat_features, self.window_sizes) # window_sizes=(10, 200, 1000)

        self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
     
        # Logger
        self.logger = self.set_logger()

        # Start the kernel thread
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

        avg_data_list = []
        feature_names = self.feature_settings['kernel_info'] + ['reward']
        avg_data_list = {feature: [] for feature in feature_names}
        avg_data_list['reward'] = []
        # start = time.time()
        self.set_protocol() # Communicate with kernel to set the protocol
        print(f"Running {self.protocol} for {self.n_steps} steps...")

        # max_rw = pow(self.bw, self.config['reward']['kappa']) / (self.rtt*10**-3)
        step_cnt = 0
        while step_cnt < self.n_steps:
            _tmp = []
            self.kernel_thread.enable()
            step_start = time.time()
            while float(time.time() - step_start) <= float(self.step_wait):
                s_tmp = np.array([])
                _feat_averages = []
                _feat_min = []
                _feat_max = []
                data = self._read_data()
                collected_data = {feature: data[i] for i, feature in enumerate(feature_names)}
                # Some preprocessing on the raw data
                collected_data['thruput'] *= 1e-6  # bps -> Mbps
                collected_data['rtt_min'] *= 1e-3  # us -> ms
                collected_data['rtt'] *= 1e-3  # us -> ms
                
                # Filter corrupted samples
                if collected_data['thruput'] > 192 or collected_data['rtt'] < self.rtt: # Filter corrupted samples
                        continue
                
                collected_data['loss_rate'] *= 0.01  # percentage -> ratio
                for key in ['crt_proto_id', 'prev_proto_id', 'delivered', 'lost', 'in_flight']: 
                        collected_data[key] = int(collected_data[key])
                
                collected_data['loss_rate'] = collected_data['loss_rate'] / 100
                reward = pow(abs(collected_data['thruput'] - self.config['reward']['zeta'] * collected_data['loss_rate']),  self.config['reward']['kappa']) / (collected_data['rtt']*1e-3) #rtt to seconds
                # max_rw = pow(self.bw, self.config['reward']['kappa']) / (self.rtt*10**-3)
                # normalized_reward = reward / max_rw

                # Features stats
                self.feat_extractor.update([val for name, val in collected_data.items() if name in self.stat_features])
                self.feat_extractor.compute_statistics()
                feat_statistics = self.feat_extractor.get_statistics()
                for size in self.window_sizes:
                    for feature in self.stat_features:
                            _feat_averages.append(feat_statistics[size]['avg'][feature])
                            _feat_min.append(feat_statistics[size]['min'][feature])
                            _feat_max.append(feat_statistics[size]['max'][feature])

                self.feat_averages = np.array(_feat_averages)
                self.feat_min = np.array(_feat_min)
                self.feat_max = np.array(_feat_max)

                # Data preprocessing
                data_tmp_collected = collected_data.copy()
                # Regenerate curr_kernel_features but stack the one_hot_encoded version of crt_proto_id
                data_tmp, before_one_hot = self.preprocess_data(data_tmp_collected)
                self.logger.log(data_tmp[0])
                if s_tmp.shape[0] == 0:
                    s_tmp = np.array(data_tmp).reshape(1, -1)
                    # log_tmp = np.array(log_kernel_features).reshape(1, -1)
                else:
                    s_tmp = np.vstack((s_tmp, np.array(data_tmp).reshape(1, -1)))

                # Filter corrupted samples (theoretically the reward will never go beyond max_rw)
                # if data_dict['reward'] <= max_rw:  # Convert the given min_rtt to us
                _tmp.append(collected_data)
            
            self.kernel_thread.disable()
            self.kernel_thread.flush()
            # Averaging
            if not _tmp:
                print("No data collected. Error")
                sys.exit(1)
            _avg_data = {feature: np.mean([d[feature] for d in _tmp]) if feature not in ('crt_proto_id', 'prev_proto_id', 'now') else _tmp[-1][feature] for feature in feature_names}

            for feature in feature_names:
                avg_data_list[feature].append(_avg_data[feature])
            step_cnt+=1
            print(f'[STEP {step_cnt}]', 'Thruput (Mbps):',
                  _avg_data['thruput'], 'Loss rate:',
                  _avg_data['loss_rate'], 'RTT (ms):',
                  _avg_data['rtt'], 'CWND:',
                  int(_avg_data['cwnd']), )
            # print(f'[STEP {step_cnt}]',
            #        'Thruput (Mbps):',
            #       _avg_data['thruput'], 'Loss rate:',
            #       _avg_data['loss_rate'], 'RTT (ms):',
            #       _avg_data['rtt'])
        print(f"Collection of {self.protocol} completed.")
        print(f"Avg thr: {round(np.mean(avg_data_list['thruput']), 4)} Mbps, \
              Avg loss rate: {round(np.mean(avg_data_list['loss_rate']), 4)}, \
              Avg RTT: {round(np.mean(avg_data_list['rtt'])*10**-3, 4)}ms, \
              Avg reward: {round(np.mean(avg_data_list['reward']), 4)}")

    def set_logger(self):
        csv_dir = f'{self.log_dir}/csv/{int(self.n_steps)}/'
        os.makedirs(csv_dir, exist_ok=True)
        path_to_file = os.path.join(csv_dir, f'{self.protocol}.bw{self.bw}.rtt{self.rtt}.bdp_mult{self.bdp_mult}.bw_factor{self.bw_factor}.csv')
        if os.path.exists(path_to_file):
            # Empty the existing file (we overwrite the file with the new collection but same parameters)
            with open(path_to_file, 'w'):
                pass
        self.logger = Logger(csv_file=path_to_file, columns=self.training_features)
        return self.logger

    def preprocess_data(self, data):
        def one_hot_encode(id, nchoices):
            vector = np.zeros(nchoices, dtype=int)
            vector[id] = 1
            return vector
    
        """
        Replace the crt_proto_id feature with its one_hot_encoded version
        """
        # The one hot encoding will be a vector of the same length as the number of protocols in the pool
        # one_hot_proto_id = self.one_hot_encode(self._inv_map_actions[data['crt_proto_id']],
        #                         len(self._inv_map_actions)).reshape(1, -1) 
        self.map_all_proto = {int(i): self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)}
        self.inv_map_all_proto = {int(v): k for k, v in self.map_all_proto.items()} # protocol id -> action id
        one_hot_proto_id = one_hot_encode(self.inv_map_all_proto[data['crt_proto_id']], len(self.proto_config)).reshape(1, -1)
        # Index of crt_proto_id in the collected data dict
        crt_proto_id_idx = self.log_features.index('crt_proto_id')
        # Store the kernel feature to append to the state
        tmp = np.concatenate(([val for feat, val in data.items() if feat in self.non_stat_features], \
                        self.feat_averages, self.feat_min, self.feat_max)).reshape(1, -1)
        # Remove crt_proto_id from the array 
        # preprocessed_data = np.delete(tmp, crt_proto_id_idx)
        tmp_no_id = np.delete(tmp.copy(), crt_proto_id_idx, axis=1)
        # Insert the one_hot_encoded version of crt_proto_id in the collected data
        # preprocessed_data = np.hstack((preprocessed_data, one_hot_proto_id))
        # Concatenate the one_hot_proto_id with the rest of the features
        preprocessed_data = np.hstack((tmp_no_id, one_hot_proto_id))
        return preprocessed_data, tmp