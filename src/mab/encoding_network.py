import tensorflow as tf
from tf_agents.networks import network, encoding_network

class EncodingNetwork(encoding_network.EncodingNetwork):
    def __init__(self, observation_spec, encoding_dim):

        self._encoding_dim = encoding_dim
        preprocessing_layers = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Reshape((1, 64)),
            tf.keras.layers.GRU(256, return_sequences=True),
            tf.keras.layers.Dense(self._encoding_dim, activation='relu')
            ]
        )
        super(EncodingNetwork, self).__init__(
                input_tensor_spec=observation_spec, 
                preprocessing_layers=preprocessing_layers,
                name='EncodingNetwork'
        )

    @property
    def encoding_dim(self):
        return self._encoding_dim

    @encoding_dim.getter
    def encoding_dim(self):
        return self._encoding_dim
