import os
import tensorflow as tf
import time

from utilities import utils
import src.utilities.utils as utils
from mab.runner import MabRunner
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import time_step as ts
from tf_agents.utils.common import Checkpointer
from src.mab.mab_environment import MabEnvironment
from comm.comm import CommManager
from tqdm import tqdm

class TestRunner(MabRunner):
    def __init__(self, rtt: int, bw: int, bdp_mult: int, bw_factor: int, k=4):
        super().__init__(policies=None, min_rtt=rtt, bw=bw, bdp_mult=bdp_mult, bw_factor=bw_factor, k=k)
        self._mahimahi_dir = 'log/mab/mahimahi-test' # for evaluation post-training
        # Set up communication with kernel
        self.cm = CommManager(k, log_dir_name='log/iperf', rtt=rtt, bw=bw, bdp_mult=bdp_mult, bw_factor=bw_factor,
                              mahimahi_dir=self._mahimahi_dir, iperf_dir=self._iperf_dir)

    def setup_communication(self):
        # Set up iperf client-server communication
        # Now a single flow between client and server is running
        # We can now set up the runner and start training the RL model    
        self.cm.init_kernel_communication()
        # self.environment.set_initial_protocol()
        self.cm.start_communication(client_tag=f'mab')

    def test(self) -> None:

        #TODO test

        # Let's run all the protocols once to skip the cold start
        self._initialize_protocols()
        time.sleep(1)

        # Load the settings and the pool from the latest training
        self.ckpt_filename = utils.get_latest_ckpt_dir(self._net_params)
        if not self.ckpt_filename or os.path.isdir(self.ckpt_filename) is False:
            raise Exception("No checkpoint found for this experiment")
        

        proto_action_names = utils.get_actions_from_experiment(self.ckpt_filename.split('/')[-1])
        # Convert keys in integer
        proto_action_names = {int(k): v for k, v in proto_action_names.items()}
        # Get the policies ids from the config
        map_actions = {int(a): int(self.proto_config[p_name]['id']) for a, p_name in proto_action_names.items()}
        self.environment.map_actions = map_actions

        self.environment.set_initial_protocol()
        time.sleep(1)
        
        self.environment: MabEnvironment = tf_py_environment.TFPyEnvironment(self.environment)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=self.batch_size, #TODO adjust batch_size (what it refers to?)
            max_length=self.steps_per_loop*self.batch_size
        )

        observers = [replay_buffer.add_batch]
        driver = dynamic_step_driver.DynamicStepDriver(
            env=self.environment,
            policy=self.agent.collect_policy,
            num_steps=self.steps_per_loop * self.batch_size,
            observers=observers
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
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

        for step in tqdm(range(50), desc="Testing"):
            driver.run()
            sel_actions = replay_buffer.gather_all().action.numpy()[0]
            rewards = replay_buffer.gather_all().reward.numpy()[0]
            for a, r in zip(sel_actions, rewards):
                print(f"[Step {step}] Action: {proto_action_names[a]} | Reward: {r} | (DEBUG) Max rw: {self.environment._max_rw}\n")
            self.agent.train(replay_buffer.gather_all())
            replay_buffer.clear()
    