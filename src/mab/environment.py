import configparser
import os
import sys
import time
import traceback
from datetime import datetime

import gym
import numpy as np
from gym import spaces
from utilities import context, utils
from utilities.feature_extractor import FeatureExtractor
from comm.kernel_thread import KernelRequest
from comm.netlink_communicator import NetlinkCommunicator
from mab.moderator import Moderator
import yaml
from utilities.logger import Logger
from comm.comm import ACTION_FLAG

from collections import deque

import math


class MabEnvironment(gym.Env):
    '''Kernel Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, comm: NetlinkCommunicator, 
                 policies,
                 config
                 ):

        # Load the configuration
        self.config = config

        # Step counter
        self.steps_per_episode = config['steps_per_episode']
        self.step_counter = 0
        self.global_step_counter = 0

        # Reward
        self.curr_reward = 0
        self.epoch = 0
        self.allow_save = False
        self.step_wait = config['step_wait_seconds']
        self.zeta = config['reward']['zeta']
        self.kappa = config['reward']['kappa']
        # self.rw_win = deque(maxlen=500)

        # Feature extractor
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.window_sizes = self.feature_settings['window_sizes']
        # self.training_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.training_features = self.non_stat_features
        self.feat_extractor = FeatureExtractor(self.stat_features, self.window_sizes) # window_sizes=(10, 200, 1000)

        # Define action and observation space
        self.proto_config = utils.parse_protocols_config()
        if policies:
            self.map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(policies)} # action: id mapping for subset of protocols
        else:
            self.map_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)} # action: id mapping for all protocols (subset not specified in the input)
        self.width_state = len(self.non_stat_features) #  + len(self.stat_features) * len(self.window_sizes) + 3
        self.action_space = spaces.Discrete(len(policies))
        max_samples = int(self.step_wait / 0.01) # 0.1 is the sampling frequency of the kernel
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(max_samples, self.width_state), dtype=int) # Check the height (None)


        # Netlink communicator
        self.netlink_communicator = NetlinkCommunicator()
        self.num_fields_kernel = config['num_fields_kernel']
        
        # Logger
        self.now_str = utils.time_to_str()
        csv_file = os.path.join(context.entry_dir, 'log', 'mab', 'run', f'run.{self.now_str}.csv')
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.logger = Logger(csv_file=csv_file, 
                    columns=['epoch', 'step']+self.log_features+['crt_protocol_id', 'reward'])
        # self.log_traces = ""
        self.allow_save = False
        self.initiated = False
        self.curr_reward = 0


    def _get_state(self):
        """
        Get the state from the mab environment.
        This function is called after the action has been taken.
        It gets network statistics from the kernel, normalizes them and compute the reward for that action.
        To get the state, we designed our algorithm such that the RL module is able to keep up with the speed of the network.
        In order to achieve this goal, we "slow down" our algorithm. It means that each observation is stored into an array which is filled 
        for step_wait seconds (the time we keep a single protocol executing within Mimic). 
        For each observation we compute the associated reward for the action taken. 
        After step_wait seconds, we compute the mean reward.

        The function returns:

        state_n: the normalized observation from the environment
        action: the action taken
        rws: the list of rewards (one for each observed state during the step_wait time)
        binary_rws: as above but the reward values are binary (1 if improved w.r.t the previous step, 0 o.w.)
        """
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
        # self.feat_extractor.update([val for name, val in collected_data.items() if name in self.stat_features])
        # self.feat_extractor.compute_statistics()
        # feat_statistics = self.feat_extractor.get_statistics()
    
        # for size in self.window_sizes:
        #     for feature in self.stat_features:
        #         _feat_averages.append(feat_statistics[size]['avg'][feature])
        #         _feat_min.append(feat_statistics[size]['min'][feature])
        #         _feat_max.append(feat_statistics[size]['max'][feature])

        # feat_averages = np.array(_feat_averages)
        # feat_min = np.array(_feat_min)
        # feat_max = np.array(_feat_max)

        # Store the kernel feature to append to the state
        # curr_kernel_features = np.concatenate(([val for feat, val in collected_data.items() if feat in self.non_stat_features], 
        #                         feat_averages, feat_min, feat_max))
        # Simple case: only non stat features
        curr_kernel_features = np.array([val for feat, val in collected_data.items() if feat in self.non_stat_features])
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
            # self.feat_extractor.update([val for name, val in collected_data.items() if name in self.stat_features])
            # self.feat_extractor.compute_statistics()
            # feat_statistics = self.feat_extractor.get_statistics()
        
            # for size in self.window_sizes:
            #     for feature in self.stat_features:
            #         _feat_averages.append(feat_statistics[size]['avg'][feature])
            #         _feat_min.append(feat_statistics[size]['min'][feature])
            #         _feat_max.append(feat_statistics[size]['max'][feature])

            # feat_averages = np.array(_feat_averages)
            # feat_min = np.array(_feat_min)
            # feat_max = np.array(_feat_max)

            if collected_data['now'] != curr_timestamp:
                curr_kernel_features = np.divide(curr_kernel_features, num_msg)

                if s_tmp.shape[0] == 0:
                    s_tmp = np.array(curr_kernel_features).reshape(1, -1)
                    # log_tmp = np.array(log_kernel_features).reshape(1, -1)
                else:
                    s_tmp = np.vstack((s_tmp, np.array(curr_kernel_features).reshape(1, -1)))
                    # log_tmp = np.vstack((log_tmp, np.array(log_kernel_features).reshape(1, -1)))

                # Store the kernel feature to append to the state
                # curr_kernel_features = np.concatenate(([val for feat, val in collected_data.items() if feat in self.non_stat_features], 
                #                 feat_averages, feat_min, feat_max))
                
                curr_kernel_features = np.array([val for feat, val in collected_data.items() if feat in self.non_stat_features])
                
                curr_timestamp = collected_data['now']

                num_msg = 1

                received_jiffies += 1

            else:
                # sum new reading to existing readings
                # curr_kernel_features = np.add(curr_kernel_features,
                #             np.concatenate((
                #                 [val for feat, val in collected_data.items() if feat in self.non_stat_features], 
                #                 feat_averages, feat_min, feat_max)))
                curr_kernel_features = np.add(curr_kernel_features,
                                              [val for feat, val in collected_data.items() if feat in self.non_stat_features])
                num_msg += 1

        # Kernel features for callbacks
        self.features = s_tmp

        # Compute the mean of rewards
        for k_features in s_tmp:
            # Extract individual features
            tmp_collected_data = {name: value for name, value in zip(self.training_features, k_features)}
            rw = self.compute_reward(tmp_collected_data['thruput'], tmp_collected_data['loss_rate'], tmp_collected_data['srtt'])
            rws.append(rw)
            if self.allow_save:
                self.logger.log([self.epoch, self.step_counter] + [val for val in k_features] + [collected_data['crt_proto_id'], rw])

        self.curr_state = self.features
        return (self.curr_state, collected_data['crt_proto_id'], rws)

    
    def compute_reward(self, thr: float, loss_rate: float, rtt: float):
        # TODO: this is single-flow reward, no friendliness. Multi-flow scenario to be considered.
        reward = pow(abs((thr - self.zeta * loss_rate)), self.kappa) / (rtt*10**-6) # rtt us -> ms
        # self.rw_win.append(reward)
        # min_rw = min(self.rw_win)
        # max_rw = max(self.rw_win)
        # We test the reward in the case we know the min rtt and the bandwidth
        # The best reward possible is when the thruput is equal to the bandwidth, loss rate is 0 and rtt is the min rtt
        max_rw = pow(self.config['bw'], self.kappa) / (self.config['min_rtt']*10**-3)
        # The best rewards are the closest to 1
        reward = reward / max_rw 
        # if max_rw - min_rw != 0:
        #     reward = (reward - min_rw) / (max_rw - min_rw)
        # else:
        #     reward = 0
        return reward

    def step(self, action):
        self._change_cca(int(action))
        observation, action, rewards = self._get_state()

        done = False if self.step_counter != self.steps_per_episode-1 else True
        self.step_counter = (self.step_counter+1) % self.steps_per_episode
        self.global_step_counter += 1

        avg_reward = round(np.mean(rewards), 4)
        self.curr_reward = avg_reward
        # avg_binary_reward = np.bincount(binary_rewards).argmax()

        info = {'avg_reward': avg_reward} 
        data = {'rewards': rewards, # Here we replace the binary rewards with the actual rewards for TS
                "features": self.features, 
                'obs': observation}

        print(f'\nStep: {self.step_counter} Sent Action: {action} Epoch: {self.epoch} | Reward: {avg_reward}')

        counter = self.step_counter if self.step_counter != 0 else self.steps_per_episode

        step = self.epoch * self.steps_per_episode + counter

        obs_mean = np.mean(observation, axis=0)

        # if not self.moderator.can_start() and step > 1:
        #     done = True

        return data, avg_reward, done, info

    def reset(self):
        self._init_communication()
        self._change_cca(0)
        observation, _, _ = self._get_state()
        self.epoch += 1

        data = {'obs': observation}

        return data  # reward, done, info can't be included


    def _init_communication(self):

        if not self.initiated:
            print("Initiating communication...")

            # Thread for kernel info
            self.kernel_thread = KernelRequest(
                self.netlink_communicator, self.num_fields_kernel)

            self.kernel_thread.start()

            print("Communication initiated")
            self.initiated = True

    def _change_cca(self, action):

        msg = self.netlink_communicator.create_netlink_msg(
            'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=int(self.map_proto[action]))

        self.netlink_communicator.send_msg(msg)

    def _end_communication(self):
        try:
            print("Closing communication...")

            # Close thread
            self.kernel_thread.exit()
            self.initiated = False

            print("Communication closed")

        except Exception as err:
            print(traceback.format_exc())

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
    
    def enable_log_traces(self):
        self.allow_save = True

    def close(self):
        self._end_communication()
 
