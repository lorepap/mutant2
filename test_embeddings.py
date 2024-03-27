import os
from src.mab.encoding_network import EncodingNetwork
from src.comm.kernel_thread import KernelRequest
from src.comm.netlink_communicator import NetlinkCommunicator
import src.utilities.utils as utils
from src.comm.comm import CommManager

import time
import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from argparse import ArgumentParser 

from src.utilities.feature_extractor import FeatureExtractor
from src.utilities.logger import Logger
from src.utilities import context
from src.utilities.change_detection import PageHinkley, ADWIN

from collections import deque

class KernelThread(KernelRequest):
        def __init__(self, comm_manager, num_fields_kernel):
                super().__init__(comm_manager.netlink_communicator, num_fields_kernel)
                self._comm_manager: CommManager = comm_manager
                self._setup_communication()
                self._init_communication()

        def _read_data(self):
                kernel_info = self.queue.get()
                # self.kernel_thread.queue.task_done()
                return kernel_info

        def _init_communication(self):
                print("Initiating communication...")
                self.start()
                print("Communication initiated")
    
        def _setup_communication(self):
                # Set up iperf client-server communication
                # Now a single flow between client and server is running
                # We can now set up the runner and start training the RL model    
                self._comm_manager.init_kernel_communication()
                self._comm_manager.start_communication(client_tag='test', server_log_dir='log/collection')

def preprocess_data(data, action_spec, map_proto, log_features, f_averages, f_min, f_max):
        def one_hot_encode(id, nchoices):
                vector = np.zeros(nchoices, dtype=int)
                vector[id] = 1
                return vector
        """
        Replace the crt_proto_id feature with its one_hot_encoded version
        """
        
        _inv_map_proto = {v: k for k, v in map_proto.items()}
        one_hot_proto_id = one_hot_encode(_inv_map_proto[str(data['crt_proto_id'])],
                                action_spec.maximum+1).reshape(1, -1)
        # Index of crt_proto_id in the collected data dict
        crt_proto_id_idx = log_features.index('crt_proto_id')
        # Store the kernel feature to append to the state
        tmp = np.concatenate(([val for feat, val in data.items() if feat in non_stat_features], 
                        f_averages, f_min, f_max)).reshape(1, -1)
        # Remove crt_proto_id from the array 
        # preprocessed_data = np.delete(tmp, crt_proto_id_idx)
        tmp_no_id = np.delete(tmp.copy(), crt_proto_id_idx, axis=1)
        # Insert the one_hot_encoded version of crt_proto_id in the collected data
        # preprocessed_data = np.hstack((preprocessed_data, one_hot_proto_id))
        # Concatenate the one_hot_proto_id with the rest of the features
        preprocessed_data = np.hstack((tmp_no_id, one_hot_proto_id))
        if preprocessed_data.shape[1] > 57:
                print("Preprocessed data shape: ", preprocessed_data.shape)
        return preprocessed_data, tmp

def compute_reward(kappa, zeta, thr, loss_rate, rtt):
        # Reward is normalized if the normalize_rw is true, otherwise max_rw = 1
        return (pow(abs((thr - zeta * loss_rate)), kappa) / (rtt*10**-3) )  # thr in Mbps; rtt in s
    

def set_initial_protocol(netlink_comm, map_proto):
        """
          Set the initial protocol for the next reset.
          This action is necessary only if the _reset() function shouldn't be overridden.
        """
        msg = netlink_comm.create_netlink_msg(
                'SENDING ACTION', msg_flags=2, msg_seq=int(map_proto[0]))
        netlink_comm.send_msg(msg)

            

if __name__ == "__main__":
        parser = ArgumentParser()
        # Accept a list of policies to be used in the environment - if it's not passed, use all of them
        parser.add_argument('--proto', '-p', nargs='+', default=None, type=str)
        parser.add_argument('--rtt', '-r', default=20, type=int)
        parser.add_argument('--bw', '-b', default=12, type=int)
        parser.add_argument('--bdp_mult', '-q', default=1, type=int)
        parser.add_argument('--bw_factor', '-f', default=1, type=int)
        parser.add_argument('--train_steps', '-n', default=250, type=int)
        parser.add_argument('--embeddings', '-e', action='store_true', default=True)
        args = parser.parse_args()
        policies = args.proto

        # Config
        config = utils.parse_training_config()
        proto_config = utils.parse_protocols_config()
        feature_settings = utils.parse_features_config()
        
        # Features stats
        feature_names = feature_settings['kernel_info'] # List of features
        stat_features= feature_settings['train_stat_features']
        non_stat_features = feature_settings['train_non_stat_features']
        log_features = utils.extend_features_with_stats(non_stat_features, stat_features)
        obs_size = len(non_stat_features) + len(stat_features)*3*3 + len(policies)-1 # one hot encoding of crt_proto_id
        window_sizes = feature_settings['window_sizes']

        # Logger
        now_str = utils.time_to_str()
        if args.embeddings:
                filename = f'{args.proto[0]}.bw{args.bw}x{args.bw_factor}.rtt{args.rtt}.bdp{args.bdp_mult}.steps{args.train_steps}.{now_str}.csv'
        else:
                filename = f'{args.proto[0]}.bw{args.bw}x{args.bw_factor}.rtt{args.rtt}.bdp{args.bdp_mult}_no_embeddings.{now_str}.csv'
        csv_file = os.path.join(context.entry_dir, 'test_embeddings', 'log', 'loss', filename)
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        logger = Logger(csv_file=csv_file, 
                columns=['step', 'reward', 'predicted', 'loss', 'thruput', 'loss_rate', 'rtt', 'proto'])
        
        # Reward
        zeta = config['reward']['zeta']
        kappa = config['reward']['kappa']

        # Change detection
        detector = ADWIN(delta=1e-8)
        
        # Communication setup (comm manager + kernel thread)
        logdir = 'test_embeddings/log'
        os.makedirs(logdir, exist_ok=True)
        comm_manager = CommManager(log_dir_name=logdir, rtt=args.rtt, bw=args.bw, bdp_mult=args.bdp_mult, bw_factor=args.bw_factor) #iperf_dir, time
        k_thread = KernelThread(comm_manager, config['num_fields_kernel'])
        
        # Encoding network
        nchoices=len(policies)
        action_spec = tensor_spec.BoundedTensorSpec(
            dtype=tf.int32, shape=(), minimum=0, maximum=nchoices-1, name='action')
        observation_spec = tensor_spec.TensorSpec(
            shape=(obs_size,), dtype=tf.float32, name='observation')
        encoding_dim = 32
        encoding_net = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Reshape((1, 64)),
            tf.keras.layers.GRU(128, return_sequences=True),
            tf.keras.layers.Dense(encoding_dim, activation='relu')
            ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        encoding_net.compile(optimizer=optimizer, loss='mse')
        encoding_net.build(input_shape=(None, obs_size))
        encoding_net.summary()
        reward_predictor = tf.keras.layers.Dense(1)  # Output a single reward value
        training_features = utils.get_training_features(non_stat_features, stat_features, action_spec.maximum+1)

        # Loop
        step_cnt = 0
        feat_extractor = FeatureExtractor(stat_features, window_sizes) # window_sizes=(10, 200, 1000)
        map_proto = {i: proto_config[p]['id'] for i, p in enumerate(policies)}
        set_initial_protocol(comm_manager.netlink_communicator, map_proto)
        train = True
        train_steps = args.train_steps
        thr_history = deque(maxlen=2000)
        rtt_history = deque(maxlen=2000)
        while step_cnt < 1500:
                k_thread.enable()
                start = time.time()
                while time.time() - start < 0.2:
                        # Empty the stats
                        s_tmp = np.array([])
                        _feat_averages = []
                        _feat_min = []
                        _feat_max = []
                        data = k_thread._read_data()
                        collected_data = {feature: data[i] for i, feature in enumerate(feature_names)}

                        # Some preprocessing on the raw data
                        collected_data['thruput'] *= 1e-6  # bps -> Mbps
                        collected_data['rtt'] *= 10**-3  # us -> ms
                        collected_data['rtt_min'] *= 10**-3  # us -> ms
                        
                        # Filter corrupted samples
                        if collected_data['thruput'] > 10*args.bw or collected_data['rtt'] < args.rtt: # Filter corrupted samples
                                continue
                        
                        collected_data['loss_rate'] *= 0.01  # percentage -> ratio
                        for key in ['crt_proto_id', 'prev_proto_id', 'delivered', 'lost', 'in_flight']: 
                                collected_data[key] = int(collected_data[key])

                        # Features stats
                        feat_extractor.update([val for name, val in collected_data.items() if name in stat_features])
                        feat_extractor.compute_statistics()
                        feat_statistics = feat_extractor.get_statistics()

                        for size in window_sizes:
                                for feature in stat_features:
                                        _feat_averages.append(feat_statistics[size]['avg'][feature])
                                        _feat_min.append(feat_statistics[size]['min'][feature])
                                        _feat_max.append(feat_statistics[size]['max'][feature])

                        feat_averages = np.array(_feat_averages)
                        feat_min = np.array(_feat_min)
                        feat_max = np.array(_feat_max)

                        # Add elements to detector
                        thr_history.append(collected_data['thruput'])
                        detector.add_element(collected_data['thruput'])

                        # Data preprocessing
                        data_tmp_collected = collected_data.copy()
                        # Regenerate curr_kernel_features but stack the one_hot_encoded version of crt_proto_id
                        data_tmp, before_one_hot = preprocess_data(data_tmp_collected, action_spec, map_proto, log_features, feat_averages, feat_min, feat_max)

                        if s_tmp.shape[0] == 0:
                                s_tmp = np.array(data_tmp).reshape(1, -1)
                                # log_tmp = np.array(log_kernel_features).reshape(1, -1)
                        else:
                                s_tmp = np.vstack((s_tmp, np.array(data_tmp).reshape(1, -1)))
                
                k_thread.disable()
                
                if len(s_tmp) == 0:
                        continue
                _observation = np.array(np.mean(s_tmp, axis=0), dtype=np.float32).reshape(1, -1)
                
                # Compute reward
                data = {name: value for name, value in zip(training_features, _observation[0])}
                reward = compute_reward(kappa, zeta, data['thruput'], data['loss_rate'], data['rtt'])
                # if len(thr_history) > 100 or step_cnt==0:
                max_thr = max(thr_history)
                # min_rtt = min(rtt_history) # I'm not using this - for now let's set a min rtt in mahimahi
                max_rw = compute_reward(thr=max_thr, loss_rate=0, rtt=args.rtt, zeta=zeta, kappa=kappa)
                norm_rw = reward / max_rw

                # Train the encoding network online for N steps
                if args.embeddings:
                        with tf.GradientTape() as tape:
                                embedding = encoding_net(_observation)
                                predicted_reward = reward_predictor(embedding)
                                loss = tf.keras.losses.mean_squared_error(norm_rw, predicted_reward)
                else:
                        with tf.GradientTape() as tape:
                                predicted_reward = reward_predictor(_observation)
                                loss = tf.keras.losses.mean_squared_error(norm_rw, predicted_reward)                
                
                pred_rw = predicted_reward.numpy().reshape(-1, 1)[0][0]
                loss_ = loss.numpy()[0][0]
                
                # Train the encoder only for train_steps steps
                if step_cnt > train_steps:
                        train = False
                if train:
                        gradients = tape.gradient(loss, encoding_net.trainable_variables + reward_predictor.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, encoding_net.trainable_variables + reward_predictor.trainable_variables))

                # Detect change
                if detector.detected_change():
                        train = True
                        train_steps = step_cnt + 100
                        thr_history.clear()
                        print("\n-----Change detected-----")
                        print("Change detected at step: ", step_cnt, "Throughput: ", collected_data['thruput'])
                        print("\n\n")
                        # Let's read the kernel data for a bit to enque the new data
                        # This will avoid that the max reward will be computed with too fresh data (potential reward > 1)
                        k_thread.empty_queue() # empty the queue


                print(f"[STEP {step_cnt}] Avg Thr: ", data['thruput'], "| Max Thr: ", max_thr, "| Loss: ", data['loss_rate'], "| RTT: ", data['rtt'], "| Reward: ", norm_rw, 
                "| Predicted Reward", predicted_reward.numpy().reshape(-1, 1)[0][0], "| Loss", loss.numpy()[0][0], "\n")
                step_cnt += 1

                # Save the loss and reward for analysis
                to_save = [step_cnt, norm_rw, pred_rw, loss_, data['thruput'], data['loss_rate'], data['rtt'], collected_data['crt_proto_id']]
                logger.log(to_save)

                # Save the embeddings for analysis

        comm_manager.stop_iperf_communication()
        comm_manager.close_kernel_communication()