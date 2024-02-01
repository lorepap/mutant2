import json
import os
import time
from typing import Any

import numpy as np
from utilities import context, utils
from utilities.callbacks import TrainingCallback
from keras.optimizers import Adam
from mab.mabagent import MabAgent
from mab.environment import MabEnvironment
from comm.comm import CommManager

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.keras.callbacks import ModelCheckpoint
from mab.moderator import Moderator
import yaml


class MabRunner():
    def __init__(self, checkpoint_filepath: str = None):

        # running params
        with open('config/train.yml', 'r') as file:
            config = yaml.safe_load(file)
        
        self.now = utils.time_to_str()
        # Set up communication with kernel
        self.cm = CommManager(log_dir_name='log/iperf', rtt=config['min_rtt'], bw=config['bw'], bdp_mult=config['bdp_mult']) #iperf_dir, time

        self.nchoices = config['num_actions']
        self.lr = config['lr']
        self.moderator = Moderator()
        self.num_features = config['num_features']

        self.base_config_dir = os.path.join(context.entry_dir, 'log/mab/config')
        self.model_path = os.path.join(
            context.entry_dir, f'log/mab/model')

        self.training_time = None
        self.step_wait_time = config['step_wait_seconds']
        self.steps_per_episode = config['steps_per_episode']
        self.num_fields_kernel = config['num_fields_kernel']
        self.training_steps = config['train_episodes'] * self.steps_per_episode

        self.model = MabAgent(self.nchoices, self.moderator)
        
        # TODO: checkpoint filepath to be changed for a better naming convention
        self.model.set_model_name(name=f'mab.{self.nchoices}actions.{self.training_steps}steps.{self.now}')
        if not checkpoint_filepath:
            self.checkpoint_filepath = os.path.join(
                context.entry_dir, f'{self.model_path}.{self.model.get_model_name()}.h5')

        self.environment = MabEnvironment(self.cm, self.moderator, config)
        self.training_steps = config['train_episodes'] * self.steps_per_episode
        self.now = time.time()

        # dir
        self.history_dir = "log/mab/history"
        self.model_dir = "log/mab/model"
        self.make_paths()

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'mab.{self.nchoices}actions.{self.training_steps}steps')

    def stop_communication(self):
        self.cm.stop_iperf_communication()

    def make_paths(self):
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self) -> Any:
        now = self.now

        checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_best_only=False,
            verbose=1
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
                    callbacks=[cb, checkpoint_callback],
            visualize=False, verbose=2)

        self.training_time = time.time() - start
        
        self.history = self.train_res.history

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

        # with open(path, 'w+') as file:
        #     json.dump(df, file)

        # update config
        # self.config['runs'].append({
        #     'model_name': self.model.get_model_name(),
        #     'path': path,
        #     'timestamp': self.now,
        #     'training_time': self.training_time,
        #     'trace': self.trace_name,
        #     'actions': self.nchoices,
        #     'step_wait': self.step_wait_time,
        #     'num_features': self.num_features,
        #     'num_kernel_fields': self.num_fields_kernel,
        #     'steps_per_episode': self.steps_per_episode,
        #     'reward': self.environment.reward_name
        # })
        # self.save_config(self.config_path, self.config)
    
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
    