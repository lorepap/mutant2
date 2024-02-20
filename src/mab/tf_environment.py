import abc
import numpy as np
import tensorflow as tf
import traceback
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec, tensor_spec 

nest = tf.nest

class BanditPyEnvironment(py_environment.PyEnvironment):

  def __init__(self, observation_spec, action_spec):
    self._observation_spec = observation_spec
    self._action_spec = action_spec 
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

  def _step(self, action):
    """Returns a time step containing the reward for the action taken."""
    reward = self._apply_action(action)
    return ts.termination(self._observe(), reward)

  # These two functions below are to be implemented in subclasses.
  @abc.abstractmethod
  def _observe(self):
    """Returns an observation."""

  @abc.abstractmethod
  def _apply_action(self, action):
    """Applies `action` to the Environment and returns the corresponding reward.
    """

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

