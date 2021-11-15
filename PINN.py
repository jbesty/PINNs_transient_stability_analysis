import tensorflow as tf
from power_system_functions import create_system_matrices
import numpy as np


class PinnModel(tf.keras.models.Model):
    """
    The PINN that incorporates the simple DenseCoreNetwork and adds the physics.
    """
    def __init__(self, neurons_in_hidden_layer, power_system, case='normal',
                 seed=12345, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        tf.random.set_seed(seed)

        self.DenseCoreNetwork = DenseCoreNetwork(n_states=power_system['n_states'],
                                                 neurons_in_hidden_layer=neurons_in_hidden_layer,
                                                 output_offset=power_system['output_offset'])
        self.n_states = power_system['n_states']
        self.n_generators = power_system['n_generators']
        self.n_buses = power_system['n_buses']
        self.loss_terms = 3 * self.n_states
        self.bus_disturbance = 4
        self.epoch_count = 0
        self.seed = seed

        self.A, self.B, self.C, self.D, self.F, self.G, self.u_0, self.x_0, = create_system_matrices(
            power_system=power_system, case=case)

        self.build(input_shape=[(None, 1), (None, self.n_buses)])

    def call(self, inputs, training=None, mask=None):
        x_time, x_power = inputs
        x_power_selective = x_power[:, self.bus_disturbance:self.bus_disturbance+1]

        network_output, network_output_t = self.calculate_time_derivatives(inputs=[x_time, x_power_selective])

        FCX = tf.sin(network_output @ self.C.T) @ self.D.T

        u_disturbance = x_power @ self.G.T
        u = u_disturbance + self.u_0.T

        network_output_physics = network_output_t - (network_output @ self.A.T + FCX @ self.F.T + u @ self.B.T)

        return_variables_splits = tf.split(network_output, num_or_size_splits=self.n_states, axis=1) + tf.split(
            network_output_t, num_or_size_splits=self.n_states, axis=1) + tf.split(network_output_physics,
                                                                                   num_or_size_splits=self.n_states,
                                                                                   axis=1)

        return return_variables_splits

    def predict_states(self, inputs):

        x_time_np, x_power_np = inputs

        x_time = tf.convert_to_tensor(x_time_np, dtype=tf.float32)
        x_power = tf.convert_to_tensor(x_power_np, dtype=tf.float32)
        x_power_selective = x_power[:, self.bus_disturbance:self.bus_disturbance+1]

        network_output, network_output_t = self.calculate_time_derivatives(inputs=[x_time, x_power_selective])

        FCX = tf.sin(network_output @ self.C.T) @ self.D.T

        u_disturbance = x_power @ self.G.T
        u = u_disturbance + self.u_0.T

        network_output_physics = network_output_t - (network_output @ self.A.T + FCX @ self.F.T + u @ self.B.T)

        return network_output.numpy(), network_output_t.numpy(), network_output_physics.numpy()

    def calculate_time_derivatives(self, inputs, **kwargs):
        time_input, _ = inputs

        list_network_output = []
        list_network_output_t = []

        for state in range(self.n_states):
            with tf.GradientTape(watch_accessed_variables=False,
                                 persistent=False) as grad_t:
                grad_t.watch(time_input)
                network_output_single = self.DenseCoreNetwork.call_inference(inputs=inputs, **kwargs)[:, state:state + 1]

                network_output_t_single = grad_t.gradient(network_output_single,
                                                          time_input,
                                                          unconnected_gradients='zero')

            list_network_output.append(network_output_single)
            list_network_output_t.append(network_output_t_single)

        network_output = tf.concat(list_network_output, axis=1)
        network_output_t = tf.concat(list_network_output_t, axis=1)

        return network_output, network_output_t

    def update_test_data_with_prediction(self, test_data):
        X_testing = [test_data['time'],
                     test_data['power']]

        prediction_split = self.predict(X_testing)
        network_states, network_states_t, network_physics = np.split(np.concatenate(prediction_split, axis=1),
                                                                     indices_or_sections=3,
                                                                     axis=1)

        test_data['states_prediction'] = network_states
        test_data['states_t_prediction'] = network_states_t
        test_data['physics_prediction'] = network_physics

        return test_data


class DenseCoreNetwork(tf.keras.models.Model):
    """
    This constitutes the core neural network with the PINN model. It outputs the angle and frequency for each
    generator and laod based on the disturbance. Additionally a common time input represents the time instance that
    shall be predicted.
    """

    def __init__(self, n_states, neurons_in_hidden_layer, output_offset):

        super(DenseCoreNetwork, self).__init__()

        self.input_normalisation = tf.keras.layers.experimental.preprocessing.Normalization(axis=1,
                                                                                            name='input_normalisation',
                                                                                            mean=np.array([0.162, 3.]),
                                                                                            variance=np.array([
                                                                                                0.022, 3.4]))

        self.hidden_layer_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layer[0],
                                                            activation=tf.keras.activations.tanh,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                                            bias_initializer=tf.keras.initializers.zeros,
                                                    name='first_layer')
        self.layer_0_normalisation = tf.keras.layers.BatchNormalization(name='layer_0_normalisation', trainable=True)
        self.hidden_layer_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layer[1],
                                                            activation=tf.keras.activations.tanh,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                                            bias_initializer=tf.keras.initializers.zeros,
                                                    name='hidden_layer_1')

        self.layer_1_normalisation = tf.keras.layers.BatchNormalization(name='layer_1_normalisation', trainable=True)

        self.dense_output_layer = tf.keras.layers.Dense(units=n_states,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal,
                                                        bias_initializer=tf.constant_initializer(-output_offset),
                                                        name='output_layer')

    def call(self, inputs, training=None, mask=None):
        concatenated_input = tf.concat(inputs, axis=1, name='input_concatenation')

        hidden_layer_0_input = self.input_normalisation(concatenated_input)
        hidden_layer_0_output = self.hidden_layer_0(hidden_layer_0_input)
        hidden_layer_1_input = self.layer_0_normalisation(hidden_layer_0_output)
        hidden_layer_1_output = self.hidden_layer_1(hidden_layer_1_input)
        output_layer_input = self.layer_1_normalisation(hidden_layer_1_output)
        network_output = self.dense_output_layer(output_layer_input)

        return network_output

    def call_inference(self, inputs, training=None, mask=None):
        concatenated_input = tf.concat(inputs, axis=1, name='input_concatenation')

        hidden_layer_0_input = self.input_normalisation(concatenated_input, training=False)
        hidden_layer_0_output = self.hidden_layer_0(hidden_layer_0_input)
        hidden_layer_1_input = self.layer_0_normalisation(hidden_layer_0_output, training=False)
        hidden_layer_1_output = self.hidden_layer_1(hidden_layer_1_input)
        output_layer_input = self.layer_1_normalisation(hidden_layer_1_output, training=False)
        network_output = self.dense_output_layer(output_layer_input)

        return network_output

