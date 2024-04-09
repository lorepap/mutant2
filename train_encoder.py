"""
Let's train the encoder from the collection data in log/collection/100
"""

import os
import numpy as np

import src.utilities.utils as utils
import tensorflow as tf
import pandas as pd
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.log_dir = 'log/collection/csv/50'
        self.proto_config = utils.parse_protocols_config()
        self.feature_settings = utils.parse_features_config()
        self.config = utils.parse_training_config()
        self.kappa = self.config['reward']['kappa']
        self.zeta = self.config['reward']['zeta']
        self.pool = [p for p in self.proto_config.keys()]

        # Features stats
        self.feature_names = self.feature_settings['kernel_info'] # List of features
        self.stat_features= self.feature_settings['train_stat_features']
        self.non_stat_features = self.feature_settings['train_non_stat_features']
        self.log_features = utils.extend_features_with_stats(self.non_stat_features, self.stat_features)
        self.obs_size = len(self.non_stat_features) + len(self.stat_features)*3*3 + len(self.proto_config)-1 # one hot encoding of crt_proto_id
        self.dataset = None
            

    def build_dataset(self):
        # Read the csv files in the log directory
        datasets = []
        for f in tqdm(os.listdir(self.log_dir)):
            print("Processing file", f, "...")
            if f.endswith('.csv'):
                print(f)
                rewards = []
                normalized_rewards = []
                bw = int(f.split('.')[1].split('bw')[1])
                rtt = int(f.split('.')[2].split('rtt')[1])
                bw_factor = int(f.split('.')[4].split('bw_factor')[1])
                data = pd.read_csv(os.path.join(self.log_dir, f))
                for idx, row in data.iterrows():
                    # print("Thruput:", row['thruput'], "Loss Rate", row['loss_rate'], "RTT", row['rtt'])
                    # print("Reward", (pow((row['thruput'] - self.zeta*row['loss_rate']), self.kappa) / ((row['rtt'])*1e-3)))
                    max_rw = pow(bw, self.kappa)/(rtt*1e-3)
                    if row['thruput'] > bw + 1:
                        max_rw = pow(bw*bw_factor, self.kappa)/(rtt*1e-3)
                    rw = pow((row['thruput'] - self.zeta*row['loss_rate']), self.kappa) / ((row['rtt'])*1e-3)
                    normalized_rw = rw/max_rw
                    rewards.append(rw)
                    normalized_rewards.append(normalized_rw)
                # debug the reward
                # Add reward and normalized reward columns to the dataset
                data['reward'] = rewards
                data['normalized_rw'] = normalized_rewards

            print(data.tail())
            datasets.append(data)
            # break # to remove after debug
        
        print("Finished processing the dataset")
        self.dataset = pd.concat(datasets, ignore_index=True)
         # Save the concatenated dataset to a CSV file
        output_file = 'collection_dataset.csv'
        # dataset.to_csv(output_file, index=False)  # Set index=False to avoid saving the DataFrame index
        print("Dataset shape", self.dataset.shape)
        print("Dataset columns", self.dataset.columns)
        print("Dataset head", self.dataset.head())


    def run(self):
        encoding_dim = 16
        encoding_net = tf.keras.models.Sequential(
            [   tf.keras.layers.Dense(self.obs_size, activation='relu'),
                tf.keras.layers.Reshape((1,self.obs_size)),
                tf.keras.layers.GRU(256, return_sequences=True),
                tf.keras.layers.Dense(encoding_dim, activation='relu')
            ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # encoding_net.compile(optimizer=optimizer, loss='mse')
        # encoding_net.build(input_shape=(None, self.obs_size))
        # encoding_net.summary()
        reward_predictor = tf.keras.layers.Dense(1)  # Output a single reward value

        _observation = self.dataset
        # Remove the reward and normalized_rw columns from the dataset
        norm_rw = _observation.pop('normalized_rw')
        rw = _observation.pop('reward')

         # Define reward predictor model
        reward_predictor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1)  # Single dense layer for reward prediction
        ])
        
        # Combine encoder and reward predictor into a single model
        combined_model = tf.keras.models.Sequential([
            encoding_net,
            reward_predictor
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        combined_model.compile(optimizer=optimizer, loss='mse')  # Compile the combined model with MSE loss
        combined_model.build(input_shape=(None, self.obs_size))  # Build the model
        combined_model.summary()

        # Train the combined model (encoder and reward predictor)
        combined_model.fit(_observation, norm_rw, epochs=100, batch_size=32)

        # Extract and save the encoder model
        encoder_model = tf.keras.models.Sequential(encoding_net.layers)
        encoder_model.build(input_shape=(None, self.obs_size))  # Build the model
        encoder_model.summary()  # Optional: Print the summary of the encoder model

        # Save the weights of the encoder model
        encoder_model.save_weights('encoder_weights.h5')

    def preprocess_data(self,data, one_hot_length, map_proto, log_features, f_averages, f_min, f_max):
            def one_hot_encode(id, nchoices):
                    vector = np.zeros(nchoices, dtype=int)
                    vector[id] = 1
                    return vector
            """
            Replace the crt_proto_id feature with its one_hot_encoded version
            """
            
            _inv_map_proto = {v: k for k, v in map_proto.items()}
            one_hot_proto_id = one_hot_encode(_inv_map_proto[str(data['crt_proto_id'])],
                                    one_hot_length).reshape(1, -1)
            # Index of crt_proto_id in the collected data dict
            crt_proto_id_idx = log_features.index('crt_proto_id')
            # Store the kernel feature to append to the state
            tmp = np.concatenate(([val for feat, val in data.items() if feat in self.non_stat_features], 
                            f_averages, f_min, f_max)).reshape(1, -1)
            # Remove crt_proto_id from the array 
            # preprocessed_data = np.delete(tmp, crt_proto_id_idx)
            tmp_no_id = np.delete(tmp.copy(), crt_proto_id_idx, axis=1)
            # Insert the one_hot_encoded version of crt_proto_id in the collected data
            # preprocessed_data = np.hstack((preprocessed_data, one_hot_proto_id))
            # Concatenate the one_hot_proto_id with the rest of the features
            preprocessed_data = np.hstack((tmp_no_id, one_hot_proto_id))
            return preprocessed_data, tmp


if __name__ == '__main__':
    trainer = Trainer()
    trainer.build_dataset()
    trainer.run()