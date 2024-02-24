import numpy as np
import os
from tf_agents.specs import array_spec
from mab.tf_environment import BanditPyEnvironment
from tf_agents.specs import array_spec
from gym import spaces
import time
import numpy as np
import traceback

from utilities import context, utils
from utilities.feature_extractor import FeatureExtractor
from comm.kernel_thread import KernelRequest
from comm.netlink_communicator import NetlinkCommunicator
from mab.moderator import Moderator
import yaml
from utilities.logger import Logger
from comm.comm import ACTION_FLAG

ACTION_FLAG = 2

class MabEnvironment(BanditPyEnvironment):

    def __init__(self, observation_spec, action_spec, policies_id=None):
        super(MabEnvironment, self).__init__(observation_spec, action_spec)
        # Load the configuration
        config = utils.parse_training_config()
        self.config = config

        # Step counter
        self.num_steps = config['num_steps']
        self.step_counter = 0
        self.global_step_counter = 0

        # Reward
        self.curr_reward = 0
        self.epoch = 0
        self.allow_save = False
        self.step_wait = config['step_wait_seconds']
        self.zeta = config['reward']['zeta']
        self.kappa = config['reward']['kappa']
        # self._reward_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.float32, minimum=0, maximum=np.inf, name='reward')

        # Feature extractor
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.window_sizes = self.feature_settings['window_sizes']
        self.training_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.feat_extractor = FeatureExtractor(self.stat_features, self.window_sizes) # window_sizes=(10, 200, 1000)

        # Define action and observation space
        self.proto_config = utils.parse_protocols_config()
        if policies_id:
            self.map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(policies_id)} # action: id mapping for subset of protocols
        else:
            self.map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)} # action: id mapping for all protocols (subset not specified in the input)
        
        # self.width_state = len(self.non_stat_features) #+ 3 * len(self.stat_features) * len(self.window_sizes)
        # self.action_space = spaces.Discrete(len(policies))
        # self.observation_space = spaces.Box(
        #     low=0, high=np.inf, shape=(10, self.width_state), dtype=int) # Check the height (None)

        # Netlink communicator
        self.netlink_communicator = NetlinkCommunicator()
        self.num_fields_kernel = config['num_fields_kernel']

        # Logger
        self.now_str = utils.time_to_str()
        csv_file = os.path.join(context.entry_dir, 'log', 'mab', 'run', f'run.{self.now_str}.csv')
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.logger = Logger(csv_file=csv_file, 
                    columns=['epoch', 'step']+self.log_features+['reward'])
        # self.log_traces = ""
        self.allow_save = False
        self.initiated = False
        self.curr_reward = 0

        # Thread for kernel info
        self.kernel_thread = KernelRequest(self.netlink_communicator, self.num_fields_kernel)

    def _observe(self):
        s_tmp = np.array([])
        log_tmp = np.array([])
        # state_n = np.array([])
        state = np.array([])
        _feat_averages = []
        _feat_min = []
        _feat_max = []
        received_jiffies = 0

        # Callbacks data
        self.features = []
        rws = []
        binary_rws = []
        collected_data = {}

        # Read kernel features
        data = self._read_data()
        collected_data = {feature: data[i] for i, feature in enumerate(self.feature_names)}
        collected_data['thruput'] *= 1e-6  # bps -> Mbps
        collected_data['loss_rate'] *= 0.01  # percentage -> ratio
        #  Convert prev_proto, delivered, lost, in_flight, crt_proto_id to an integer 
        for key in ['prev_proto_id', 'delivered', 'lost', 'in_flight', 'crt_proto_id']:
            collected_data[key] = int(collected_data[key])

        # print the collected data
        print(f'Collected data: {collected_data}')

        # Compute statistics for the features
        self.feat_extractor.update([val for name, val in collected_data.items() if name in self.stat_features])
        self.feat_extractor.compute_statistics()
        feat_statistics = self.feat_extractor.get_statistics()

        for size in self.window_sizes:
            for feature in self.stat_features:
                _feat_averages.append(feat_statistics[size]['avg'][feature])
                _feat_min.append(feat_statistics[size]['min'][feature])
                _feat_max.append(feat_statistics[size]['max'][feature])

        feat_averages = np.array(_feat_averages)
        feat_min = np.array(_feat_min)
        feat_max = np.array(_feat_max)

        # Store the kernel feature to append to the state
        curr_kernel_features = np.concatenate(([val for feat, val in collected_data.items() if feat in self.non_stat_features], 
                                feat_averages, feat_min, feat_max))
        curr_timestamp = collected_data['now']
        num_msg = 1
        start = time.time()

        # Read and record data for step_wait seconds
        while float(time.time() - start) <= float(self.step_wait):
            # Empty the stats
            _feat_averages = []
            _feat_min = []
            _feat_max = []
            collected_data = {}

            # Read kernel features
            data = self._read_data()
            collected_data = {feature: data[i] for i, feature in enumerate(self.feature_names)}
            collected_data['thruput'] *= 1e-6  # bps -> Mbps            # collected_data = {name: value for name, value in zip(self.feature_names, data)}
            collected_data['loss_rate'] *= 0.01  # percentage -> ratio
            #  Convert prev_proto, delivered, lost, in_flight, crt_proto_id to an integer 
            for key in ['prev_proto_id', 'delivered', 'lost', 'in_flight', 'crt_proto_id']:
                collected_data[key] = int(collected_data[key])
            
            # Compute statistics for the features
            self.feat_extractor.update([val for name, val in collected_data.items() if name in self.stat_features])
            self.feat_extractor.compute_statistics()
            feat_statistics = self.feat_extractor.get_statistics()

            for size in self.window_sizes:
                for feature in self.stat_features:
                    _feat_averages.append(feat_statistics[size]['avg'][feature])
                    _feat_min.append(feat_statistics[size]['min'][feature])
                    _feat_max.append(feat_statistics[size]['max'][feature])

            feat_averages = np.array(_feat_averages)
            feat_min = np.array(_feat_min)
            feat_max = np.array(_feat_max)

            if collected_data['now'] != curr_timestamp:
                curr_kernel_features = np.divide(curr_kernel_features, num_msg)

                if s_tmp.shape[0] == 0:
                    s_tmp = np.array(curr_kernel_features).reshape(1, -1)
                    # log_tmp = np.array(log_kernel_features).reshape(1, -1)
                else:
                    s_tmp = np.vstack((s_tmp, np.array(curr_kernel_features).reshape(1, -1)))
                    # log_tmp = np.vstack((log_tmp, np.array(log_kernel_features).reshape(1, -1)))

                # Store the kernel feature to append to the state
                curr_kernel_features = np.concatenate(([val for feat, val in collected_data.items() if feat in self.non_stat_features], 
                                feat_averages, feat_min, feat_max))
                # log_kernel_features = np.concatenate(([val for feat, val in collected_data.items() if feat in self.feature_names],
                #                 feat_averages, feat_min, feat_max))
                
                curr_timestamp = collected_data['now']

                num_msg = 1

                received_jiffies += 1

            else:
                # sum new reading to existing readings
                curr_kernel_features = np.add(curr_kernel_features,
                            np.concatenate((
                                [val for feat, val in collected_data.items() if feat in self.non_stat_features], 
                                feat_averages, feat_min, feat_max)))
                # log_kernel_features = np.add(log_kernel_features,
                #             np.concatenate(([val for feat, val in collected_data.items() if feat in self.feature_names],
                #                 feat_averages, feat_min, feat_max)))
                num_msg += 1

        # Observation as a mean of the samples
        # TODO check state averaging (not sure if we can do that anywhere else with the tf settings) 
        # self.curr_state = np.mean(s_tmp, axis=0)
        self.curr_state = s_tmp

        return self.curr_state

    def _apply_action(self, action):

        # Change the CCA
        msg = self.netlink_communicator.create_netlink_msg(
                'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=int(self.map_proto[action]))
        self.netlink_communicator.send_msg(msg)

            # Compute the mean of rewards
        rws = []
        for k_features in self.curr_state:
            # Extract individual features
            tmp_collected_data = {name: value for name, value in zip(self.training_features, k_features)}
            rw = self._compute_reward(tmp_collected_data['thruput'], tmp_collected_data['loss_rate'], tmp_collected_data['srtt'])
            rws.append(rw)
            # if self.allow_save:
            #     self.logger.log([self.epoch, self.step_counter] + [val for val in k_features] + [rw])
        reward = np.mean(rws)
        return reward
  
    def _compute_reward(self, thr, loss_rate, rtt):
        return pow(abs((thr - self.zeta * loss_rate)), self.kappa) / (rtt*10**-6)

    def enable_log_traces(self):
        self.allow_save = True

    def close(self):
        self._end_communication()