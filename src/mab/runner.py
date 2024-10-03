import os
import time
from typing import Any, Dict

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils.common import Checkpointer

from mab.mab_agent import NeuralUCBMabAgent
from mab.mab_environment import MabEnvironment
from mab.mpts import MPTS
from utilities.selection_strategy import MPTSStrategy, ManualSelectionStrategy, ProtocolSelectionStrategy
from comm.comm import CommManager
import utilities.utils as utils
import plots as plt

from tqdm import tqdm

class MabRunner:
    def __init__(self, min_rtt: float, bw: float, bdp_mult: float, bw_factor: float, log: bool = True, restore: bool = False, k: int = 4, protocol_selection_strategy: ProtocolSelectionStrategy = None):
        self.restore = restore
        self.net_params = {'bw': bw, 'rtt': min_rtt, 'bdp_mult': bdp_mult, 'bw_factor': bw_factor}
        self.nchoices = k
        
        self._load_config()
        self._setup_directories()
        self._setup_communication()
        self._setup_environment()
        self._setup_agent()
        self._setup_mpts()

        self.protocol_selection_strategy = protocol_selection_strategy or self._default_strategy() # Run MPTS or Manual selection

    def _default_strategy(self) -> ProtocolSelectionStrategy:
        return ManualSelectionStrategy([0, 1, 2, 3])

    def _load_config(self):
        self.config = utils.parse_training_config()
        self.feature_settings = utils.parse_features_config()
        self.proto_config = utils.parse_protocols_config()
        self.timestamp = utils.time_to_str()

    def _setup_directories(self):
        self.log_dir = "log/mab"
        self.history_dir = f"{self.log_dir}/history"
        self.model_dir = f"{self.log_dir}/model"
        self.ckpt_dir = f"{self.log_dir}/checkpoint"
        self._mahimahi_dir = f"{self.log_dir}/mahimahi"
        self._iperf_dir = f"{self.log_dir}/iperf"
        
        for directory in [self.log_dir, self.history_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)

    def _setup_communication(self):
        self.cm = CommManager(
            self.nchoices,
            log_dir_name='log/iperf',
            rtt=self.net_params['rtt'],
            bw=self.net_params['bw'],
            bdp_mult=self.net_params['bdp_mult'],
            bw_factor=self.net_params['bw_factor'],
            mahimahi_dir=self._mahimahi_dir,
            iperf_dir=self._iperf_dir
        )
        self.cm.init_kernel_communication()
        self.cm.start_communication(client_tag=f'mab') # start iper server and client

    def _stop_communication(self):
        self.cm.stop_iperf_communication() # Stops the server
        self.environment.close() # Kill the kernel thread

    def _setup_environment(self):
        observation_spec = self._create_observation_spec()
        action_spec = self._create_action_spec()
        time_step_spec = ts.time_step_spec(observation_spec)
        
        self.environment = MabEnvironment(observation_spec, action_spec, self.net_params, normalize_rw=True, change_detection=True)
        self.environment = tf_py_environment.TFPyEnvironment(self.environment)

    def _create_observation_spec(self):
        obs_size = self._calculate_observation_size()
        return tensor_spec.TensorSpec(shape=(obs_size,), dtype=tf.float32, name='observation')

    def _create_action_spec(self):
        return tensor_spec.BoundedTensorSpec(dtype=tf.int32, shape=(), minimum=0, maximum=self.nchoices-1, name='action')

    def _calculate_observation_size(self):
        stat_features = self.feature_settings['train_stat_features']
        non_stat_features = self.feature_settings['train_non_stat_features']
        return len(non_stat_features) + len(stat_features)*3*3 + len(self.proto_config) - 1

    def _setup_agent(self):
        encoding_net = self._create_encoding_network()
        self.agent = NeuralUCBMabAgent(
            time_step_spec=self.environment.time_step_spec(),
            action_spec=self.environment.action_spec(),
            alpha=0.1,
            gamma=0.9,
            encoding_network=encoding_net,
            encoding_network_num_train_steps=-1,
            encoding_dim=encoding_net.encoding_dim,
        )

    def _create_encoding_network(self):
        encoding_dim = 16
        encoding_net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self._calculate_observation_size(), activation='relu'),
            tf.keras.layers.Reshape((1, self._calculate_observation_size())),
            tf.keras.layers.GRU(256, return_sequences=True),
            tf.keras.layers.Dense(encoding_dim, activation='relu')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        encoding_net.compile(optimizer=optimizer, loss='mse')
        encoding_net.build(input_shape=(None, self._calculate_observation_size()))
        weights_file = f'log/mab/weights/weights.bw{self.net_params["bw"]}x{self.net_params["bw_factor"]}.rtt{self.net_params["rtt"]}.bdp{self.net_params["bdp_mult"]}.steps300.h5'
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"File {weights_file} not found. Please train the encoding network first.")
        encoding_net.load_weights(weights_file)
        return encoding_net

    def _setup_mpts(self):
        mpts_config = utils.parse_mpts_config()
        self.map_all_proto = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)}
        self.inv_map_all_proto = {v: k for k, v in self.map_all_proto.items()}
        self.mpts = MPTS(
            arms=self.map_all_proto,
            k=self.nchoices,
            T=int(mpts_config['T']),
            thread=self.environment.kernel_thread,
            net_channel=self.cm.netlink_communicator,
            step_wait=mpts_config['step_wait']
        )

    def train(self) -> Any:
        self._initialize_protocols()
        pool = self._select_protocol_pool()
        self._setup_training_environment(pool)
        
        replay_buffer = self._create_replay_buffer()
        driver = self._create_driver(replay_buffer)
        
        global_step = tf.compat.v1.train.get_or_create_global_step()
        train_checkpointer = self._setup_checkpointer(replay_buffer, global_step)
        
        self._run_training_loop(driver, replay_buffer, global_step, train_checkpointer)
        self._stop_communication()

    def _initialize_protocols(self):
        print("Initializing protocols...")
        for _, proto_id in self.map_all_proto.items():
            print(f"Initializing protocol: {self.proto_names[int(proto_id)]}")
            start = time.time()
            while time.time() - start < 0.5:
                msg = self.cm.netlink_communicator.create_netlink_msg('SENDING ACTION', msg_flags=2, msg_seq=int(proto_id))
                self.cm.netlink_communicator.send_msg(msg)

    def _select_protocol_pool(self):
        # pool = self.mpts.run()
        # proto_names = {val['id']: name for name, val in self.proto_config.items()}
        # settings_map_proto = {action: proto_names[p_id] for action, p_id in enumerate(pool)}
        # print("Selected pool: ", settings_map_proto)
        # return {action: int(p_id) for action, p_id in enumerate(pool)}

        all_protocols = {i: self.proto_config[p]['id'] for i, p in enumerate(self.proto_config)}
        pool = self.protocol_selection_strategy.select_protocols(all_protocols)

        proto_names = {val['id']: name for name, val in self.proto_config.items()}
        settings_map_proto = {action: proto_names[p_id] for action, p_id in enumerate(pool)}
        print("Selected pool: ", settings_map_proto)

        return pool

    def _setup_training_environment(self, pool):
        self.environment.map_actions = pool
        self.environment.set_logger()
        self.environment.set_initial_protocol()
        time.sleep(1)

    def _create_replay_buffer(self):
        return tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.policy.trajectory_spec,
            batch_size=1,
            max_length=self.config['steps_per_loop']
        )

    def _create_driver(self, replay_buffer):
        return dynamic_step_driver.DynamicStepDriver(
            env=self.environment,
            policy=self.agent.collect_policy,
            num_steps=self.config['steps_per_loop'],
            observers=[replay_buffer.add_batch]
        )

    def _setup_checkpointer(self, replay_buffer, global_step):
        train_checkpointer = Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.initialize_or_restore()
        self.environment.step_counter = global_step.numpy()
        return train_checkpointer

    def _run_training_loop(self, driver, replay_buffer, global_step, train_checkpointer):
        print("Start experiment...")
        print(f"Training encoder for {self.agent._encoding_network_num_train_steps} steps")
        start = time.time()
        total_steps = self.config['num_steps'] + self.agent._encoding_network_num_train_steps
        
        for step in tqdm(range(global_step.numpy(), total_steps)):
            driver.run()
            self._log_step_info(replay_buffer, step)
            self.agent.train(replay_buffer.gather_all())
            replay_buffer.clear()
        
        train_checkpointer.save(global_step)
        self.training_time = time.time() - start
        self._update_logs(global_step)

    def _log_step_info(self, replay_buffer, step):
        sel_actions = replay_buffer.gather_all().action.numpy()[0]
        rewards = replay_buffer.gather_all().reward.numpy()[0]
        for a, r in zip(sel_actions, rewards):
            step_type = "Encoder Train Step" if step < self.agent._encoding_network_num_train_steps else "Step"
            print(f"[{step_type} {step}] Action: {self.settings_map_proto[a]} | Reward: {r} | (DEBUG) Max rw: {self.environment._max_rw}\n")

    def _update_logs(self, global_step):
        utils.update_log(
            os.path.join(self.log_dir, 'settings.json'),
            self.settings,
            'success',
            self.training_time,
            int(global_step.numpy())
        )

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