import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras

def build_bnn_model(input_shape, output_shape):
    model = tfk.Sequential([
        tfpl.DenseVariational(input_shape=input_shape,
                              units=64,
                              make_prior_fn=lambda t: tfd.Normal(loc=tf.zeros(t), scale=tf.ones(t)),
                              make_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
                              kl_weight=1/X_train.shape[0],
                              activation='relu'),
        tfpl.DenseVariational(units=tfpl.OneHotCategorical.params_size(output_shape),
                              make_prior_fn=lambda t: tfd.Normal(loc=tf.zeros(t), scale=tf.ones(t)),
                              make_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
                              kl_weight=1/X_train.shape[0]),
        tfpl.OneHotCategorical(event_size=output_shape),
    ])
    return model

def compile_and_train(model, X_train, y_train, epochs=100):
    model.compile(optimizer='adam',
                  loss=lambda y, model: -model.log_prob(y),
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    
# Assuming X_train, y_train are your data and labels
input_shape = X_train.shape[1:]
output_shape = y_train.shape[1]  # For classification, this would be the number of classes
bnn_model = build_bnn_model(input_shape, output_shape)
compile_and_train(bnn_model, X_train, y_train)
