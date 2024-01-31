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
from comm.kernel_thread import KernelRequest
from comm.netlink_communicator import NetlinkCommunicator
from mab.moderator import Moderator
from comm.comm import ACTION_FLAG
import math


class MabEnvironment(gym.Env):
    '''Kernel Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, comm: NetlinkCommunicator, 
                 moderator,
                 config
                 ):

        # self.moderator = moderator
        self.width_state = config['num_features'] # state dimension
        self.height_state = config['window_len']

        # Kernel state
        self.jiffies_per_state = config['jiffies_per_state']
        self.last_delivered = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(config['num_actions'])
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.height_state, self.width_state), dtype=int)

        # Step counter
        self.steps_per_episode = config['steps_per_episode']
        self.step_counter = 0

        # Keep current state
        # self.curr_state = np.zeros((self.height_state, self.width_state))
        # print( "[DEBUG] self.curr_state.shape: ", self.curr_state.shape)

        # Reward
        self.rws = dict() # list of rws for each step
        # self.delta = config['delta']
        self.curr_reward = 0
        self.last_rtt = 0
        self.min_thr = 0
        self.min_rtt = sys.maxsize
        self.last_cwnd = 0
        self.epoch = -1
        self.allow_save = False
        self.step_wait = config['step_wait_seconds']
        self.zeta = config['reward']['zeta']
        self.kappa = config['reward']['kappa']
        self.mss = None
        self.max_bw = 0.0
        self.num_features_tmp = self.width_state + 2 # all features to be normalized (this is not the state dimension)
        # self.normalizer = Normalizer(self.num_features_tmp)
        self.cwnd = []


        # Netlink communicator
        self.netlink_communicator = NetlinkCommunicator()

        self.with_kernel_thread = True
        self.num_fields_kernel = config['num_fields_kernel']

        self.now_str = utils.time_to_str()
        self.log_traces = ""
        self.allow_save = False
        self.floating_error = 1e-12
        self.nb = 1e6
        self.initiated = False
        self.curr_reward = 0
        self.moderator = moderator


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
        # state_n = np.array([])
        state = np.array([])
        received_jiffies = 0

        # Callbacks data
        self.features = []
        rws = []
        binary_rws = []


        if self.with_kernel_thread:
            print("[DEBUG] reading data from kernel...")
            timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action,  = self._read_data()
        else:
            timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action = self._recv_data()


        # Compute thr and loss rate from kernel statistics
        rtt = rtt if rtt > 0 else 1e-5
        # thr = cwnd * self.mss / (rtt/self.nb)
        loss_rate = 0 if (lost + delivered) == 0 else lost / (lost + delivered)

        # print(f"[DEBUG] {delivered}, {self.mss}, {cwnd}, {rtt}, {thr}, {cwnd*self.mss/(rtt/self.nb)}")

        # Print kernel statistics
        print(f"[DEBUG] cwnd: {cwnd}, rtt: {rtt}, rtt_dev: {rtt_dev}, rtt_min: {rtt_min}, \
              mss: {self.mss}, delivered: {delivered}, lost: {lost}, in_flight: {in_flight}, \
              retrans: {retrans}, action: {action}, thruput: {thr}")

        delivered_diff = delivered - self.last_delivered
        self.last_delivered = delivered
        self.update_rtt(rtt)

        # to ms TODO: check this value (kernel rtt should be in ms so it's already correct)
        # rtt = rtt/1000
        # rtt_dev = rtt_dev/1000
        # rtt_min = rtt_min/1000

        curr_kernel_features = np.array(
            [cwnd, rtt, rtt_dev, rtt_min, delivered, delivered_diff, loss_rate, in_flight, retrans, thr])
        
        self.features.append(curr_kernel_features)
        
        # Number of total features which will be extracted during the step_wait time (switching time)
        num_features_tmp = self.num_features_tmp

        curr_timestamp = timestamp

        num_msg = 1

        start = time.time()

        # Read and record data for step_wait seconds
        while float(time.time() - start) <= float(self.step_wait):
            if self.with_kernel_thread:
                timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action, thr = self._read_data()
            else:
                timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action, thr = self._recv_data()

            # thruput bit/s
            rtt = rtt if rtt > 0 else 1e-5
            # thr = (cwnd * self.mss) / (rtt/self.nb)
            loss_rate = 0 if (lost + delivered) == 0 else lost / (lost + delivered)

            
            delivered_diff = delivered - self.last_delivered
            self.last_delivered = delivered
            self.update_rtt(rtt)

            if timestamp != curr_timestamp:
                curr_kernel_features = np.divide(curr_kernel_features, num_msg)

                if s_tmp.shape[0] == 0:
                    s_tmp = np.array(curr_kernel_features).reshape(
                        1, num_features_tmp)
                else:
                    s_tmp = np.vstack(
                        (s_tmp, np.array(curr_kernel_features).reshape(1, num_features_tmp))
                    )

                curr_kernel_features = np.array(
                    [cwnd, rtt, rtt_dev, rtt_min, delivered, delivered_diff, loss_rate, in_flight, retrans, thr]
                    )
                
                curr_timestamp = timestamp

                num_msg = 1

                received_jiffies += 1

            else:
                # sum new reading to existing readings
                curr_kernel_features = np.add(curr_kernel_features,
                            np.array([cwnd, rtt, rtt_dev, rtt_min, delivered, delivered_diff, loss_rate, in_flight, retrans, thr]))
                num_msg += 1

        # Kernel features for callbacks
        self.features = s_tmp

        # Compute the mean of rewards
        for _, k_features in enumerate(s_tmp):
            # Extract individual features
            cwnd, rtt, rtt_dev, rtt_min, delivered, delivered_diff, loss_rate, in_flight, retrans, thr = k_features
            rw = self.compute_reward(thr, loss_rate, rtt)
            rws.append(rw)
            binary_rw = 0 if rw <= self.curr_reward else 1
            binary_rws.append(binary_rw)

        # TODO: mean of rewards; could we do better?
        # The following aggregated value refers to the mean of the rewards computed during step_wait (switching time) within the step
        reward = np.mean(rws)
        self.curr_reward = reward # current reward
        self.curr_state = self.features
        return (self.curr_state, action, rws, binary_rws)

    
    def compute_reward(self, thr: float, loss_rate: float, rtt: float):
        # TODO: this is single-flow reward, no friendliness. Multi-flow scenario to be considered.
        reward_base = (thr - self.zeta * loss_rate) / rtt # rtt should never be 0 since min_rtt is set to be > 0
        reward = pow(abs(reward_base), self.kappa) 
        return reward

    def update_rtt(self, rtt: float) -> None:
        if rtt > 0:
            self.last_rtt = rtt
    
    def record(self, state, reward, step, action):
        cwnd, rtt, rtt_dev, rtt_min, delivered, delivered_diff, lost, in_flight, retrans, thr= state

        if cwnd == 0:
            return

        cwnd_diff = cwnd - self.last_cwnd

        self.last_cwnd = cwnd

        self.log_traces = f'{self.log_traces}\n {action}, {cwnd}, {rtt}, {rtt_dev}, {delivered}, {delivered_diff}, {lost}, {in_flight}, {retrans}, {cwnd_diff}, {thr}' \
                 f'{step}, {round(self.curr_reward, 4)}, {round(reward, 4)}'

    def step(self, action):
        self._change_cca(int(action))
        observation, observed_action, rewards, binary_rewards = self._get_state()

        done = False if self.step_counter != self.steps_per_episode-1 else True
        self.step_counter = (self.step_counter+1) % self.steps_per_episode

        avg_reward = round(np.mean(rewards), 4)
        avg_binary_reward = np.bincount(binary_rewards).argmax()

        info = {'avg_reward': avg_reward}
        data = {'binary_rewards': binary_rewards, 
                'rewards': rewards, 
                "features": self.features, 
                'obs': observation}

        print(f'\nStep: {self.step_counter} \t 
              Sent Action: {action} \t 
              Received Action: {observed_action} \t 
              Epoch: {self.epoch} | Reward: {avg_reward} ({np.mean(avg_binary_reward)})  | Data Size: {observation.shape[0]}')

        counter = self.step_counter if self.step_counter != 0 else self.steps_per_episode

        step = self.epoch * self.steps_per_episode + counter

        obs_mean = np.mean(observation, axis=0)

        if self.allow_save:
            self.record(obs_mean, avg_binary_reward,
                        step, observed_action)

        # if not self.moderator.can_start() and step > 1:
        #     done = True

        return data, avg_reward, done, info

    def reset(self):
        self._init_communication()
        self._change_cca(0)
        observation, _, _, _ = self._get_state()

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
            'SENDING ACTION', msg_flags=ACTION_FLAG, msg_seq=action)

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

    def save_log(self, model_name: str, log_path: str) -> str:

        log_file_name = f'{model_name}.{self.now_str}.csv'
        log_fullpath = os.path.join(context.entry_dir, log_path, log_file_name)

        with open(log_fullpath, 'w') as writer:
            writer.write(self.log_traces)

        return log_file_name, log_fullpath

    def close(self):
        self._end_communication()



# class Normalizer():
#     def __init__(self, input_dim):
#         # self.params = params
#         # self.config = config
#         self.n = 1e-5
#         num_inputs = input_dim
#         self.mean = np.zeros(num_inputs)
#         self.mean_diff = np.zeros(num_inputs)
#         self.var = np.zeros(num_inputs)
#         self.dim = num_inputs
#         self.min = np.zeros(num_inputs)

#     def observe(self, x):
#         self.n += 1
#         last_mean = np.copy(self.mean)
#         self.mean += (x-self.mean)/self.n
#         self.mean_diff += (x-last_mean)*(x-self.mean)
#         self.var = self.mean_diff/self.n
#         # Check for zero standard deviation and set it to a small value
#         for i in range(self.dim):
#             if self.var[i] == 0:
#                 self.var[i] = 1e-5

#     def normalize(self, inputs):
#         obs_std = np.sqrt(self.var)
#         a=np.zeros(self.dim)
#         if self.n > 2:
#             a=(inputs - self.mean)/obs_std
#             for i in range(0,self.dim):
#                 if a[i] < self.min[i]:
#                     self.min[i] = a[i]
#             return a
#         else:
#             return np.zeros(self.dim)

#     def normalize_delay(self,delay):
#         obs_std = math.sqrt(self.var[0])
#         if self.n > 2:
#             return (delay - self.mean[0])/obs_std
#         else:
#             return 0

#     def stats(self):
#         return self.min    
