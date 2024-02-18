from datetime import datetime
from typing import Any
import warnings
from copy import deepcopy
import pickle
from sklearn.linear_model import SGDClassifier
from contextualbandits.online import _BasePolicy, LinTS

import numpy as np
import tensorflow as tf
from utilities import utils
from rl.core import Agent
from tensorflow.keras.callbacks import History
from mab.moderator import Moderator
import statistics

from collections import Counter

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

class MabAgent(Agent):

    def __init__(self, nchoices, moderator=None, **kwargs):
        super(MabAgent, self).__init__(**kwargs)
        # TODO: build id string for model
        self.moderator = moderator
        self.nchoices = nchoices
        # self.base_algorithm = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, shuffle=False, verbose=0)
        self.model = self.get_policy()
        self.prev_action = 0

    def get_model(self) -> _BasePolicy:
        return self.model

    def get_policy(self) -> _BasePolicy:

        beta_prior = ((1, 1), 5)
        p = LinTS(nchoices=self.nchoices, beta_prior='auto', v_sq=0.25, method='chol')
        return p

    def load_weights(self, filepath) -> None:

        with open(filepath, 'rb') as file:
            self.model = pickle.load(file)

    def save_weights(self, filepath, overwrite=False) -> None:

        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

    def reset_states(self) -> None:
        self.recent_observation = None

    def compile(self) -> None: # TODO: optimizer? metrics?
        self.compiled = True

    def forward(self, observation) -> np.ndarray:

        if observation['obs'].shape[0] == 0:
            print('Observation is empty. Cannot predict')
            return self.prev_action

        # Select an action.
        actions = self.model.predict(observation['obs'])
    
        # Book-keeping.
        self.recent_observation = observation
        
        # select the action with the highest occurrence
        action_counts = Counter(actions)
        print("[DEBUG] action_counts: ", action_counts)

        # Get the action that occurs the most
        action = action_counts.most_common(1)[0][0]
        # action = actions[len(actions) - 1]

        # The action we will use later for the backward pass, is an array of n_samples, where
        # the selected action is repeated.
        # This little trick will allow the model to understand that the observed environment is related to 
        # the most occurring action selected during the forward pass.
        self.actions = [action] * observation['obs'].shape[0]
        
        self.prev_action = action

        # Last predicted action
        # action = actions[N-1]

        return action

    def backward(self, reward, terminal) -> None:

        if not self.training or 'rewards' not in self.recent_observation:
            return {}

        # np.random.seed(self.counter)

        N = len(self.actions)
        # print("[DEBUG] N: ", N)

        # Here, since we want data in batches, we pull the rewards from the observation
        # Note that we could customize the fit method in core.py to handle the rewards as an array
        # The reward in the fit function (output of step) is an average - can be modified since we don't use it
        self.model.partial_fit(
            self.recent_observation['obs'],
            np.array(self.actions),
            np.array(self.recent_observation['rewards']))

    def alter_x(self, x: np.ndarray) -> np.ndarray:
        return x

    def can_exploit(self) -> bool:
        return True

    def set_model_name(self, name: str) -> None:
        self.model_name = name

    def get_model_name(self) -> str:
        return self.model_name

    def reset_name(self) -> None:
        self.model_name = self.orig_name
    
    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            # Edit: check if iperf simulation has done (moderator class)
            done = False
            try:
                while not done:
                    if self.moderator.is_stopped():
                        print("exiting rl-module....\n")
                        raise StopIteration

                    callbacks.on_step_begin(episode_step)

                    action = self.forward(observation)
                    print(action)
                    if self.processor is not None:
                        action = self.processor.process_action(action)
                    reward = 0.
                    accumulated_info = {}
                    for _ in range(action_repetition):
                        callbacks.on_action_begin(action)
                        observation, r, d, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, r, d, info = self.processor.process_step(observation, r, d, info)
                        callbacks.on_action_end(action)
                        reward += r
                        for key, value in info.items():
                            if not np.isreal(value):
                                continue
                            if key not in accumulated_info:
                                accumulated_info[key] = np.zeros_like(value)
                            accumulated_info[key] += value
                        if d:
                            done = True
                            break
                    if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                        done = True
                    self.backward(reward, terminal=done)
                    episode_reward += reward

                    step_logs = {
                        'action': action,
                        'observation': observation,
                        'reward': reward,
                        'episode': episode,
                        'info': accumulated_info,
                    }
                    callbacks.on_step_end(episode_step, step_logs)
                    episode_step += 1
                    self.step += 1

            except StopIteration:
                break

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history
