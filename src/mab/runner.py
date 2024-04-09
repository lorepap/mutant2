import json
import os
import time
from typing import Any

import numpy as np
from utilities import context, utils
from utilities.callbacks import TrainingCallback, ModelCheckpoint

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.utils.common import Checkpointer

from src.mab.mab_agent import NeuralUCBMabAgent, LinTSMabAgent
from src.mab.mab_environment import MabEnvironment
from src.mab.mpts import MPTS

from comm.comm import CommManager
import plots as plt

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import yaml

from src.mab.encoding_network import EncodingNetwork
from tqdm import tqdm


class MabRunner():
    def __init__(self, policies, min_rtt, bw, bdp_mult, bw_factor, log=True, restore=False, k=4):

        # List of CCA IDs to map the policies to the action 
        self.restore = restore

        # Input params
        self.bw = bw
        self.min_rtt = min_rtt
        self.bdp_mult = bdp_mult
        self.bw_factor = bw_factor
        self.nchoices = k

        # TF Environment
        self.batch_size = 1
        
        self.proto_config = utils.parse_protocols_config()

        # Config params
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        self.timestamp = utils.time_to_str()
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.obs_size = len(self.non_stat_features) + len(self.stat_features)*3*3 + len(self.proto_config) - 1 # one hot encoding of crt_proto_id
        self.steps_per_loop = self.config['steps_per_loop']

        # dir
        self.log_dir = "log/mab"
        self.history_dir = "log/mab/history"
        self.model_dir = "log/mab/model"
        self.ckpt_dir = "log/mab/checkpoint"
        self._mahimahi_dir = "log/mab/mahimahi"
        self._iperf_dir = "log/mab/iperf"
        self.make_paths()
        
        # Set up communication with kernel
        self.cm = CommManager(k, log_dir_name='log/iperf', rtt=min_rtt, bw=bw, bdp_mult=bdp_mult, bw_factor=bw_factor,
                              mahimahi_dir=self._mahimahi_dir, iperf_dir=self._iperf_dir)

        if 'lr' in config:
            self.lr = config['lr']
        self.num_features = config['num_features']

        self.base_config_dir = os.path.join(context.entry_dir, 'log/mab/config')
        self.model_path = os.path.join(
            context.entry_dir, self.log_dir, 'model')

        self.training_time = None
        self.step_wait_time = config['step_wait_seconds']
        self.num_steps = config['num_steps']
        self.total_steps = config['num_steps'] * config["steps_per_loop"]
        
        # Define observation and action space
        self.step_wait = config['step_wait_seconds']
        max_samples = int(self.step_wait / 0.01) # 0.1 is the sampling frequency of the kernel
        observation_spec = tensor_spec.TensorSpec(
            shape=(self.obs_size,), dtype=tf.float32, name='observation')
        time_step_spec = ts.time_step_spec(observation_spec)
        action_spec = tensor_spec.BoundedTensorSpec(
            dtype=tf.int32, shape=(), minimum=0, maximum=self.nchoices-1, name='action')

        # TF Agent
        # encoding_net = EncodingNetwork(observation_spec=observation_spec, encoding_dim=16)
        encoding_dim = 16
        encoding_net = tf.keras.models.Sequential(
            [   tf.keras.layers.Dense(self.obs_size, activation='relu'),
                tf.keras.layers.Reshape((1, self.obs_size)),
                tf.keras.layers.GRU(256, return_sequences=True),
                tf.keras.layers.Dense(encoding_dim, activation='relu')
            ])
        optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
        encoding_net.compile(optimizer=optimizer, loss='mse')
        encoding_net.build(input_shape=(None, self.obs_size))
        encoding_net.load_weights(f'log/mab/weights/weights.bw{bw}x{bw_factor}.rtt{min_rtt}.bdp{bdp_mult}.steps300.h5')
        self.agent = NeuralUCBMabAgent(time_step_spec=time_step_spec, 
            action_spec=action_spec,
            alpha=0.1,
            gamma=0.9,
            encoding_network=encoding_net,
            encoding_network_num_train_steps=-1,
            encoding_dim=encoding_net.encoding_dim,
            # optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
        )

        # self.agent = LinTSMabAgent(
        #     time_step_spec=time_step_spec,
        #     action_spec=action_spec,
        #     alpha=0.01,
        #     gamma=0.75,
        # )

        # Environment
        self._net_params = {'bw': self.bw, 'rtt': self.min_rtt, 'bdp_mult': self.bdp_mult, "bw_factor": self.bw_factor}
        self.environment = MabEnvironment(observation_spec, action_spec, self._net_params, normalize_rw=True, change_detection=True)
        self.environment.allow_save = log

        # MPTS
        mpts_config = utils.parse_mpts_config()
        self.map_all_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)}
        self.inv_map_all_proto = {v: k for k, v in self.map_all_proto.items()} # protocol id -> action id
        self.mpts = MPTS(arms=self.map_all_proto, k=self.nchoices, T=int(mpts_config['T']), 
                thread=self.environment.kernel_thread, net_channel=self.cm.netlink_communicator, step_wait=mpts_config['step_wait'])
        

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        # self.environment.set_initial_protocol()
        self.cm.start_communication(client_tag=f'mab')

    def stop_communication(self):
        self.cm.stop_iperf_communication()
        # self.cm.close_kernel_communication()
        self.environment.close()

    def make_paths(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def _initialize_protocols(self):
        """
        We leave the protocol to run for a short period of time to update its internal parameters.
        """
        self.proto_config = utils.parse_protocols_config() # for debug (protocol names)
        self.proto_names = {int(self.proto_config[p]['id']): p for p in self.proto_config.keys()}
        print("Initializing protocols...")
        for _, proto_id in self.map_all_proto.items():
            print("Initializing protocol: ", self.proto_names[int(proto_id)])
            start = time.time()
            while time.time() - start < 0.5:
                msg = self.cm.netlink_communicator.create_netlink_msg(
                        'SENDING ACTION', msg_flags=2, msg_seq=int(proto_id))
                self.cm.netlink_communicator.send_msg(msg)
    
    def train(self) -> Any:   

        # Let's run all the protocols once to skip the cold start
        self._initialize_protocols()

        # Let's calculate the policies in the pool with MPTS
        pool = self.mpts.run()
        # id -> proto name
        proto_names = {val['id']: name for name, val in self.proto_config.items()}
        settings_map_proto = {action: proto_names[p_id] for action, p_id in enumerate(pool)} # action: proto name (for logging)
        print("Selected pool: ", settings_map_proto)
        env_map_proto = {action: int(p_id) for action, p_id in enumerate(pool)} # action: kernel proto id (for the environment)
 
        self.environment.map_actions = env_map_proto

        # Check if the checkpoint dir has to be restored
        self.settings = {'timestamp': self.timestamp, **self._net_params, **self.config, 'action_mapping': settings_map_proto, **self.feature_settings}
        
        # Checkpoint
        self.restored_timestamp = None
        if self.restore:
            # Here we get the latest checkpoint dir for this experiment
            self.ckpt_filename = utils.get_latest_ckpt_dir(self.settings)
            if not self.ckpt_filename or os.path.isdir(self.ckpt_filename) is False:
                print("No checkpoint found for this experiment, checkpoint set to current timestamp")
                self.ckpt_filename = os.path.join(self.ckpt_dir, self.timestamp)
            else:
                print(f"Restoring checkpoint from {self.ckpt_filename}...")
                self.restored_timestamp = self.ckpt_filename.split('/')[-1]
                self.environment.timestamp = self.restored_timestamp
        else:
            self.ckpt_filename = os.path.join(self.ckpt_dir, self.timestamp)
        self.settings['checkpoint_dir'] = self.ckpt_filename
        
        # Log the settings
        utils.log_settings(os.path.join(self.log_dir, 'settings.json'), self.settings, 'failed')

        # Enable logging in the enviroment
        self.environment.set_logger()

        # Set initial protocol
        print("Set initial protocol...")
        self.environment.set_initial_protocol()

        time.sleep(1)

        self.environment: MabEnvironment = tf_py_environment.TFPyEnvironment(self.environment)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=self.batch_size, #TODO adjust batch_size (what it refers to?)
            max_length=self.steps_per_loop*self.batch_size
        )

        # regret_metric = tf_metrics.RegretMetric(compute_optimal_reward)
        regret_values = []

        observers = [replay_buffer.add_batch]
        driver = dynamic_step_driver.DynamicStepDriver(
            env=self.environment,
            policy=self.agent.collect_policy,
            num_steps=self.steps_per_loop * self.batch_size,
            observers=observers
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
        os.makedirs(self.ckpt_filename, exist_ok=True)

        train_checkpointer = Checkpointer(
            ckpt_dir=self.ckpt_filename,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        
        train_checkpointer.initialize_or_restore()
        self.environment.step_counter = global_step.numpy()

        print("Start experiment...")
        print("Training encoder for", self.agent._encoding_network_num_train_steps, "steps")
        start = time.time()
        total_steps = self.num_steps + self.agent._encoding_network_num_train_steps
        for step in tqdm(range(global_step.numpy(), total_steps)): # Number of total steps is num_steps * steps_per_loop
            driver.run()
            sel_actions = replay_buffer.gather_all().action.numpy()[0]
            rewards = replay_buffer.gather_all().reward.numpy()[0]
            for a, r in zip(sel_actions, rewards):
                if step < self.agent._encoding_network_num_train_steps:
                    print(f"[Encoder Train Step {step}] Action: {settings_map_proto[a]} | Reward: {r} | (DEBUG) Max rw: {self.environment._max_rw}\n") 
                else:
                    print(f"[Step {step}] Action: {settings_map_proto[a]} | Reward: {r} | (DEBUG) Max rw: {self.environment._max_rw}\n") 
            self.agent.train(replay_buffer.gather_all())
            replay_buffer.clear()
        
        train_checkpointer.save(global_step)
        self.training_time = time.time() - start
        # self.save_figs()
        utils.update_log(os.path.join(self.log_dir, 'settings.json'), self.settings, 'success', self.training_time, int(global_step.numpy()))

    def save_figs(self):
        exp = {'bw': self.bw, 'rtt': self.min_rtt, 'bdp_mult': self.bdp_mult}
        plt.plot_training_reward_single(self.timestamp, exp)
        plt.plot_training_reward_multi(self.timestamp, self.policies_id, exp, str(self.total_steps))
        plt.plot_rtt_multi(self.timestamp, self.policies_id, exp, str(self.total_steps))
        plt.plot_thr_multi(self.timestamp, self.policies_id, exp, str(self.total_steps))

    def compute_optimal_reward(observation):
       pass

    def shut_down_env(self) -> None:
        self.environment.close()
    
    @property
    def mahimahi_dir(self):
        return self._mahimahi_dir
    
    @property
    def iperf_dir(self):
        return self._iperf_dir
    
    @property
    def net_params(self):
        return self._net_params

    @mahimahi_dir.setter
    def mahimahi_dir(self, value):
        self._mahimahi_dir = value

    @iperf_dir.setter
    def iperf_dir(self, value):
        self._iperf_dir = value

    @net_params.setter
    def net_params(self, value):
        self._net_params = value