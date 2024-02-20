import json
import os
import time
from typing import Any

import numpy as np
from utilities import context, utils
from utilities.callbacks import TrainingCallback, ModelCheckpoint

import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts

from mab.mab_tf_agent import MabAgent
from mab.mab_tf_environment import MabEnvironment

from comm.comm import CommManager

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from mab.moderator import Moderator
import yaml


class MabRunner():
    def __init__(self, policies, min_rtt, bw, bdp_mult, checkpoint_filepath: str = None, log=True):

        # List of CCA IDs to map the policies to the action 
        self.policies_id = policies
        
        # running params
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        self.timestamp = utils.time_to_str()
        self.feature_settings = utils.parse_features_config()
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.obs_size = len(self.non_stat_features) + len(self.stat_features)*3*3

        # dir
        self.log_dir = "log/mab"
        self.history_dir = "log/mab/history"
        self.model_dir = "log/mab/model"
        self.make_paths()
        
        # Set up communication with kernel
        self.cm = CommManager(log_dir_name='log/iperf', rtt=min_rtt, bw=bw, bdp_mult=bdp_mult) #iperf_dir, time

        if not policies:
            self.proto_config = utils.parse_protocols_config()
            self.policies_id = list(self.proto_config.keys())

        self.nchoices = len(self.policies_id)
        if 'lr' in config:
            self.lr = config['lr']
        self.moderator = Moderator()
        self.num_features = config['num_features']

        self.base_config_dir = os.path.join(context.entry_dir, 'log/mab/config')
        self.model_path = os.path.join(
            context.entry_dir, self.log_dir, 'model')

        self.training_time = None
        self.step_wait_time = config['step_wait_seconds']
        self.num_steps = config['num_steps']
        
        # Define observation and action space
        self.step_wait = config['step_wait_seconds']
        max_samples = int(self.step_wait / 0.01) # 0.1 is the sampling frequency of the kernel
        observation_spec = tensor_spec.BoundedTensorSpec(
            shape=(self.obs_size,), dtype=tf.float32, minimum=0, maximum=np.inf)
        time_step_spec = ts.time_step_spec(observation_spec)
        action_spec = tensor_spec.BoundedTensorSpec(
            dtype=tf.int32, shape=(), minimum=0, maximum=self.nchoices)

        
        # TF Agent
        self.agent = MabAgent(time_step_spec=time_step_spec, action_spec=action_spec)
        
        # TF Environment
        self.environment = MabEnvironment(observation_spec, action_spec, self.policies_id)
        self.environment.allow_save = log
        self.now = time.time()

        # Settings
        if self.policies_id:
            map_proto = {p: action for action, p in zip(self.environment.map_proto.keys(), self.policies_id)}
        else:
            map_proto = self.environment.map_proto
        self.settings = {'timestamp': self.timestamp, **config, 'action_mapping': map_proto, **self.environment.feature_settings}
        utils.log_settings(os.path.join(self.log_dir, 'settings.json'), self.settings, 'failed')


    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'mab.{self.nchoices}actions.{self.num_steps}steps')

    def stop_communication(self):
        self.cm.stop_iperf_communication()
        self.cm.close_kernel_communication()
        self.environment.close()

    def make_paths(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self) -> Any:
        steps_per_loop = 1 # @param
        batch_size = 20

        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=batch_size, #TODO adjust batch_size (what it refers to?)
            max_length=steps_per_loop)

        observers = [replay_buffer.add_batch]

        driver = dynamic_step_driver.DynamicStepDriver(
            env=self.environment,
            policy=self.agent.collect_policy,
            num_steps=steps_per_loop * batch_size,
            observers=observers)

        step = self.environment.reset()
        print("observation:", step.observation)

        for _ in range(self.num_steps):
            action_step = self.agent.collect_policy.action(step)
            next_step = self.environment.step(action_step.action)
            experience = self.trajectory_for_bandit(step, action_step, next_step)
            print(experience)
            self.agent.train(replay_buffer.gather_all())
            step = next_step
        replay_buffer.clear()


    def test(self, episodes: int) -> None:

        now = self.now

        cb: TrainingCallback = TrainingCallback(
            log_file_path=os.path.join(
            context.entry_dir, 
            f'log/mab/history/debug_{self.agent.get_model_name()}.json'
            )
        )
        
        # self.environment.enable_log_traces()
        
        self.agent.test(self.environment,
                        nb_episodes=episodes, 
                        visualize=False, 
                        callbacks=[cb])

        # save logs
        # log_name, log_path = self.environment.save_log(model_id, 'log/mab/trace') #TODO: logging

    # We need to add another dimension here because the agent expects the
    # trajectory of shape [batch_size, time, ...], but in this tutorial we assume
    # that both batch size and time are 1. Hence all the expand_dims.

    def trajectory_for_bandit(self, initial_step, action_step, final_step):
        return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                                action=tf.expand_dims(action_step.action, 0),
                                policy_info=action_step.info,
                                reward=tf.expand_dims(final_step.reward, 0),
                                discount=tf.expand_dims(final_step.discount, 0),
                                step_type=tf.expand_dims(initial_step.step_type, 0),
                                next_step_type=tf.expand_dims(final_step.step_type, 0))


    def save_history(self) -> None:
        path = os.path.join(
            context.entry_dir, 
            self.history_dir,
            f'episode_history_{self.agent.get_model_name()}.json'
            )

        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_json(path)
    
    def save_model(self, reset_model: bool = True) -> str:
        path = os.path.join(
            context.entry_dir, f'log/mab/model/{self.agent.get_model_name()}.h5')
        self.agent.save_weights(path, True)

        print(f"Saving model...")
        print("[DEBUG] model weights saved successfully in", path)
    
    def get_optimizer(self) -> optimizer_v2.OptimizerV2:
        return Adam(lr=self.lr)
    
    def calculate_score(self) -> float:
        return np.mean(self.history['episode_reward']) if self.history != None else 0

    def shut_down_env(self) -> None:
        self.environment.close()
    