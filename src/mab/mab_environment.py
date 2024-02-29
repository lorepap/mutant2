import numpy as np
import os
from tf_agents.specs import array_spec
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec
from gym import spaces
import time
import numpy as np
import traceback

from utilities import context, utils
from utilities.feature_extractor import FeatureExtractor
from comm.kernel_thread import KernelRequest
from comm.netlink_communicator import NetlinkCommunicator
import yaml
from utilities.logger import Logger
from comm.comm import ACTION_FLAG

ACTION_FLAG = 2

class MabEnvironment(bandit_py_environment.BanditPyEnvironment):

    def __init__(self, observation_spec, action_spec, policies_id=None, net_params=None, batch_size=1, normalize_rw: bool = False, logger: bool = False):
        super(MabEnvironment, self).__init__(observation_spec, action_spec)
        self._action_spec = action_spec
        self._batch_size = batch_size
        # Load the configuration
        config = utils.parse_training_config()
        self.config = config

        # Step counter
        self.num_steps = config['num_steps']
        self.step_counter = 0

        # Reward
        self.curr_reward = 0
        self.epoch = 0
        self.allow_save = False
        self.step_wait = config['step_wait_seconds']
        self.zeta = config['reward']['zeta']
        self.kappa = config['reward']['kappa']
        self._params = net_params
        if normalize_rw:
            self.max_rw = pow(self._params['bw'], self.config['reward']['kappa']) / (self._params['rtt']*10**-6)
        else:
            self.max_rw = 1

        # Feature extractor
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.window_sizes = self.feature_settings['window_sizes']
        # self.training_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.training_features = utils.get_training_features(self.non_stat_features, self.stat_features, self._action_spec.maximum+1)
        self.feat_extractor = FeatureExtractor(self.stat_features, self.window_sizes) # window_sizes=(10, 200, 1000)

        # Define action and observation space
        self.proto_config = utils.parse_protocols_config()
        if policies_id:
            self._map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(policies_id)} # action: id mapping for subset of protocols
        else:
            self._map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)} # action: id mapping for all protocols (subset not specified in the input)
        self._inv_map_proto = {v: k for k, v in self._map_proto.items()}
        self.crt_action = None

        # Netlink communicator
        self.netlink_communicator = NetlinkCommunicator()
        self.num_fields_kernel = config['num_fields_kernel']

        # Logger
        self.now_str = utils.time_to_str()
        csv_file = os.path.join(context.entry_dir, 'log', 'mab', 'run', f'run.{self.now_str}.csv')
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.logger = Logger(csv_file=csv_file, 
                        columns=['epoch', 'step']+self.log_features+['reward']) if logger else None
        self.initiated = False
        self.curr_reward = 0

        # Thread for kernel info
        self.kernel_thread = KernelRequest(self.netlink_communicator, self.num_fields_kernel)
        self._init_communication()

    def _observe(self):
        s_tmp = np.array([])
        _log_tmp = []
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

        # Block to kernel thread to avoid samples from the next protocol
        self.kernel_thread.enable()
        
        # Read and record data for step_wait seconds
        start = time.time()
        while float(time.time() - start) <= float(self.step_wait):
            # Empty the stats
            _feat_averages = []
            _feat_min = []
            _feat_max = []

            # Read kernel features
            data = self._read_data()
            collected_data = {feature: data[i] for i, feature in enumerate(self.feature_names)}
            # print("[DEBUG] rtt", collected_data['rtt'])
            # print("[DEBUG] ACTION RECEIVED", collected_data['crt_proto_id'])
            # Discard samples of another protocol
            if self.crt_action:
                if collected_data['crt_proto_id'] != int(self._map_proto[self.crt_action[0]]):
                    continue
            
            collected_data['thruput'] *= 1e-6  # bps -> Mbps
            
            # Filter corrupted samples
            if collected_data['thruput'] > 10*self._params['bw'] or collected_data['rtt'] < self._params['rtt']*10**3: # Filter corrupted samples
                continue
            
            collected_data['loss_rate'] *= 0.01  # percentage -> ratio
            for key in ['crt_proto_id', 'prev_proto_id', 'delivered', 'lost', 'in_flight']: 
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

            self.feat_averages = np.array(_feat_averages)
            self.feat_min = np.array(_feat_min)
            self.feat_max = np.array(_feat_max)

            # if collected_data['now'] != curr_timestamp:
            # curr_kernel_features = np.divide(curr_kernel_features, num_msg)

            data_tmp_collected = collected_data.copy()
            # Regenerate curr_kernel_features but stack the one_hot_encoded version of crt_proto_id
            data_tmp, before_one_hot = self.preprocess_data(data_tmp_collected)
            _log_tmp.append(before_one_hot) # Store the features before one_hot_encoding (only for logging)
            
            if s_tmp.shape[0] == 0:
                s_tmp = np.array(data_tmp).reshape(1, -1)
                # log_tmp = np.array(log_kernel_features).reshape(1, -1)
            else:
                s_tmp = np.vstack((s_tmp, np.array(data_tmp).reshape(1, -1)))
                # log_tmp = np.vstack((log_tmp, np.array(log_kernel_features).reshape(1, -1)))

        self.kernel_thread.disable()
        # Observation as a mean of the samples
        # Check if the crt_proto_id is the same for all the samples (it must be, otherwise is an error))
        self.log_values = np.mean(np.array(_log_tmp), axis=0).reshape(1, -1)
        # print("[DEBUG] Current proto id", self.log_values[0][7])
        # crt_proto_id_idx = self.non_stat_features.index('crt_proto_id')
        # self.log_values[0][crt_proto_id_idx] = int(collected_data['crt_proto_id'])

        self._observation = np.array(np.mean(s_tmp, axis=0), dtype=np.float32).reshape(1, -1)
        # self._observation = np.array(s_tmp)

        if self._observation.shape[1] != self._observation_spec.shape[0]:
            raise ValueError('The number of features in the observation should match the observation spec.')

        return self._observation

    def _apply_action(self, action):
        # TODO Apply the action and get the reward. Here the reward is not the reward of the action selected, the the previous one
        self.crt_action = action
        # Change the CCA
        msg = self.netlink_communicator.create_netlink_msg(
                'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=int(self._map_proto[action[0]]))
        self.netlink_communicator.send_msg(msg)
        self._observation = self._observe() # Update the observation to get the fresh reward

        # Compute the reward given the mean of the collected samples as the observation (shape: (1, num_features))

        data = {name: value for name, value in zip(self.training_features, self._observation[0])}
        reward = self._compute_reward(data['thruput'], data['loss_rate'], data['rtt'])
        reward = np.array(reward).reshape(self.batch_size)
        if self.logger:
            to_save = [self.epoch, self.step_counter] + [val for val in self.log_values[0]] + [reward[0]]
            self.logger.log(to_save)
            self.step_counter+=1
        return reward

        
    def _compute_reward(self, thr, loss_rate, rtt):
        # Reward is normalized if the normalize_rw is true, otherwise max_rw = 1
        return (pow(abs((thr - self.zeta * loss_rate)), self.kappa) / (rtt*10**-6) ) / self.max_rw

    def enable_log_traces(self):
        self.allow_save = True

    def close(self):
        self._end_communication()

    def _recv_data(self):
      msg = self.netlink_communicator.receive_msg()
      data = self.netlink_communicator.read_netlink_msg(msg)
      split_data = data.decode(
          'utf-8').split(';')[:self.num_fields_kernel]
      return list(map(int, split_data))

    def _read_data(self):
      kernel_info = self.kernel_thread.queue.get()
      self.kernel_thread.queue.task_done()
      return kernel_info
    
    def _init_communication(self):
        if not self.initiated:
            print("Initiating communication...")
            self.kernel_thread.start()
            print("Communication initiated")
            self.initiated = True

    def _end_communication(self):
      try:
          print("Closing communication...")

          # Close thread
          self.kernel_thread.exit()
          self.initiated = False

          print("Communication closed")

      except Exception as err:
          print(traceback.format_exc())

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    def batched(self) -> bool:
        return True

    def one_hot_encode(self, id, nchoices):
        vector = np.zeros(nchoices, dtype=int)
        vector[id] = 1
        return vector
    
    def preprocess_data(self, data):
        """
        Replace the crt_proto_id feature with its one_hot_encoded version
        """
        one_hot_proto_id = self.one_hot_encode(self._inv_map_proto[str(data['crt_proto_id'])],
                                self._action_spec.maximum+1).reshape(1, -1)
        # Index of crt_proto_id in the collected data dict
        crt_proto_id_idx = self.log_features.index('crt_proto_id')
        # Store the kernel feature to append to the state
        tmp = np.concatenate(([val for feat, val in data.items() if feat in self.non_stat_features], 
                        self.feat_averages, self.feat_min, self.feat_max)).reshape(1, -1)
        # Remove crt_proto_id from the array 
        # preprocessed_data = np.delete(tmp, crt_proto_id_idx)
        tmp_no_id = np.delete(tmp.copy(), crt_proto_id_idx, axis=1)
        # Insert the one_hot_encoded version of crt_proto_id in the collected data
        # preprocessed_data = np.hstack((preprocessed_data, one_hot_proto_id))
        # Concatenate the one_hot_proto_id with the rest of the features
        preprocessed_data = np.hstack((tmp_no_id, one_hot_proto_id))
        return preprocessed_data, tmp
            
    def set_initial_protocol(self):
        """
          Set the initial protocol for the next reset.
          This action is necessary only if the _reset() function shouldn't be overridden.
        """
        msg = self.netlink_communicator.create_netlink_msg(
                'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=int(self._map_proto[0]))
        self.netlink_communicator.send_msg(msg)