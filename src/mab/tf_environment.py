import abc
import numpy as np
import tensorflow as tf
import traceback
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec, tensor_spec 

nest = tf.nest

class BanditPyEnvironment(py_environment.PyEnvironment):

  def __init__(self, observation_spec, action_spec, batch_size):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._batch_size = batch_size
    super(BanditPyEnvironment, self).__init__()

  # Helper functions.
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  
  def reward_spec(self):
     return tensor_spec.TensorSpec(shape=(), dtype=np.float32, name='reward')

  def _empty_observation(self):
    return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype),
                                 self.observation_spec())

  # These two functions below should not be overridden by subclasses.
  def _reset(self):
    """Returns a time step containing an observation."""
    self._init_communication()
    return ts.restart(self._observe(), batch_size=self.batch_size)
  
  @property
  def batch_size(self) -> int:
    return self._batch_size

  def _step(self, action):
    """Returns a time step containing the reward for the action taken."""
    proto_id = action.numpy()[0]
    reward = self._apply_action(proto_id)
    return ts.termination(self._observe(), reward)

  # These two functions below are to be implemented in subclasses.
  @abc.abstractmethod
  def _observe(self):
    """Returns an observation."""

  @abc.abstractmethod
  def _apply_action(self, action):
    """Applies `action` to the Environment and returns the corresponding reward.
    """

