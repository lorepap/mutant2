import json
import os
import time
from typing import Any

import numpy as np
from utilities import context, utils
from utilities.callbacks import TrainingCallback, ModelCheckpoint
from keras.optimizers import Adam
from mab.mabagent import MabAgent
from mab.environment import MabEnvironment
from comm.comm import CommManager

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from mab.moderator import Moderator
import yaml
import logging


class MabRunner():
    def __init__(self, checkpoint_filepath: str = None, log=True):

        # running params
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        self.timestamp = utils.time_to_str()

        # dir
        self.log_dir = "log/mab"
        self.history_dir = "log/mab/history"
        self.model_dir = "log/mab/model"
        self.make_paths()
        
        # Set up communication with kernel
        self.cm = CommManager(log_dir_name='log/iperf', rtt=config['min_rtt'], bw=config['bw'], bdp_mult=config['bdp_mult']) #iperf_dir, time

        self.nchoices = config['num_actions']
        self.lr = config['lr']
        self.moderator = Moderator()
        self.num_features = config['num_features']

        self.base_config_dir = os.path.join(context.entry_dir, 'log/mab/config')
        self.model_path = os.path.join(
            context.entry_dir, self.log_dir, 'model')

        self.training_time = None
        self.step_wait_time = config['step_wait_seconds']
        self.steps_per_episode = config['steps_per_episode']
        self.num_fields_kernel = config['num_fields_kernel']
        self.training_steps = config['train_episodes'] * self.steps_per_episode

        self.model = MabAgent(self.nchoices, self.moderator)
        
        # TODO: checkpoint filepath to be changed for a better naming convention
        self.model.set_model_name(name=f'mab.{self.nchoices}actions.{self.training_steps}steps.{self.timestamp}')
        if not checkpoint_filepath:
            self.checkpoint_filepath = os.path.join(
                context.entry_dir, self.log_dir, 'checkpoint', f'{self.model_path}.{self.model.get_model_name()}')

        self.environment = MabEnvironment(self.cm, self.moderator, config)
        self.environment.allow_save = log
        self.training_steps = config['train_episodes'] * self.steps_per_episode
        self.now = time.time()

        # Settings
        self.settings = {'timestamp': self.timestamp, **config, **self.environment.feature_settings}
        utils.log_settings(os.path.join(self.log_dir, 'settings.json'), self.settings, 'failed')


    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'mab.{self.nchoices}actions.{self.training_steps}steps')

    def stop_communication(self):
        self.cm.stop_iperf_communication()
        self.cm.close_kernel_communication()
        self.environment.close()

    def make_paths(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self) -> Any:

        checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath
        )

        cb: TrainingCallback = TrainingCallback(
            log_file_path=os.path.join(
                context.entry_dir, 
                self.history_dir,
                f'{self.model.get_model_name()}.json'
            )
        )

        start = time.time()

        self.model.compile()
        self.train_res = self.model.fit(self.environment, nb_steps=self.training_steps, 
                    callbacks=[cb],
            visualize=False, verbose=2)

        self.training_time = time.time() - start
        
        self.history = self.train_res.history

        # log the completed training
        utils.update_log(os.path.join(self.log_dir, 'settings.json'), self.settings, 'success', self.training_time)

        return self.history

    def test(self, episodes: int) -> None:

        now = self.now

        cb: TrainingCallback = TrainingCallback(
            log_file_path=os.path.join(
            context.entry_dir, 
            f'log/mab/history/debug_{self.model.get_model_name()}.json'
            )
        )
        
        self.environment.enable_log_traces()
        
        self.model.test(self.environment,
                        nb_episodes=episodes, 
                        visualize=False, 
                        callbacks=[cb])

        # save logs
        # log_name, log_path = self.environment.save_log(model_id, 'log/mab/trace') #TODO: logging

    def save_history(self) -> None:
        path = os.path.join(
            context.entry_dir, 
            self.history_dir,
            f'episode_history_{self.model.get_model_name()}.json'
            )

        import pandas as pd
        df = pd.DataFrame(self.history)
        df.to_json(path)
    
    def save_model(self, reset_model: bool = True) -> str:
        path = os.path.join(
            context.entry_dir, f'log/mab/model/{self.model.get_model_name()}.h5')
        self.model.save_weights(path, True)

        print(f"Saving model...")
        print("[DEBUG] model weights saved successfully in", path)
    
    def get_optimizer(self) -> optimizer_v2.OptimizerV2:
        return Adam(lr=self.lr)
    
    def calculate_score(self) -> float:
        return np.mean(self.history['episode_reward']) if self.history != None else 0

    def shut_down_env(self) -> None:
        self.environment.close()
    