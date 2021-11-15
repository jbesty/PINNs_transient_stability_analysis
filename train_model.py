import numpy as np
import pathlib
import pickle
import tensorflow as tf

from dataset_handling import divide_dataset, prepare_data
from power_system_functions import create_power_system
from PINN import PinnModel


def train_model(simulation_id, NN_type, n_power_steps_training, n_time_steps_training, seed_tensorflow):
    """
    The core training routine.

    :param simulation_id: a unique idenfier, useful when running multiple training setups etc
    :param NN_type: One of the three types NN, dtNN, or PINN - loss function is adjusted accordingly
    :param n_power_steps_training: parameter to control the number of data points
    :param n_time_steps_training: parameter to control the number of data points
    :param seed_tensorflow: seed for initialisation of the weights
    """

    # -----------------------------
    # Defining the relevant directories
    # -----------------------------

    # TODO: Define the path to store all relevant data
    #
    # directory_data: pathlib.Path = pathlib.Path('Here_goes_your_path')

    raise Exception('Please specify directory_data, then delete this Exception.')

    directory_logging = directory_data / 'logs' / simulation_id

    directory_results = directory_data / 'result_datasets'

    directory_model_weights = directory_data / 'model_weights'

    directory_quantile = directory_data / 'quantiles'

    # -----------------------------
    # Simple type check
    # -----------------------------

    if type(simulation_id) is not str:
        raise Exception('Provide simulation_id as string.')

    if type(NN_type) is not str:
        raise Exception('Provide NN_type as string.')

    if type(n_power_steps_training) is not int:
        raise Exception('Provide n_power_steps_training as integer.')

    if type(n_time_steps_training) is not int:
        raise Exception('Provide n_time_steps_training as integer.')

    if type(seed_tensorflow) is not int:
        raise Exception('Provide seed_tensorflow as integer.')

    # -----------------------------
    # Define the power system, here, by default the investigated Kundur two area system
    # -----------------------------

    power_system = create_power_system()
    n_states = power_system['n_states']

    # -----------------------------
    # Basic NN and training parameters and logging setup
    # -----------------------------

    neurons_in_hidden_layer = [200, 200]
    epochs_total = 10000
    learning_rate_initial = 0.01
    learning_rate_decay = 0.99

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=directory_logging,
                                                          histogram_freq=1,
                                                          profile_batch=0,
                                                          write_graph=False,
                                                          )

    # -----------------------------
    # instantiate model
    # -----------------------------

    model = PinnModel(neurons_in_hidden_layer=neurons_in_hidden_layer,
                      power_system=power_system,
                      case='line_tripped',
                      seed=seed_tensorflow)

    # -----------------------------
    # load data and select data and collocation points
    # -----------------------------
    with open(directory_data / 'datasets' / 'complete_dataset.pickle', "rb") as f:
        complete_data = pickle.load(f)

    training_data, training_data_pure, validation_data, testing_data = divide_dataset(dataset=complete_data,
                                                                                      n_power_steps_training=n_power_steps_training,
                                                                                      n_time_steps_training=n_time_steps_training)

    # -----------------------------
    # setting of the loss weights, heuristic
    # -----------------------------

    loss_weights_initial = np.array([1.0, 1.0, 1.0, 1.0,
                                     2.0, 2.0, 2.0, 2.0,
                                     1.0, 1.0,
                                     0.5, 0.5, 0.5, 0.5,
                                     0.04, 0.04, 0.04, 0.04,
                                     0.12, 0.12,
                                     1000.0, 1000.0, 1000.0, 1000.0,
                                     5.0, 5.0, 5.0, 5.0,
                                     3.0, 3.0])

    factor_data_points = (5 * 9) / n_time_steps_training / n_power_steps_training * 5.0e+05

    loss_weights_combined = loss_weights_initial.copy()
    loss_weights_combined[20:30] = loss_weights_initial[20:30] / factor_data_points

    # -----------------------------
    # NN type specific settings, primarily regarding the loss calculation
    # -----------------------------
    if NN_type == 'NN':
        X_training, y_training = prepare_data(training_data_pure, n_states=n_states)
        sample_weights_static = np.split(training_data_pure['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(training_data_pure['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(np.ones(training_data_pure['data_type'].shape), indices_or_sections=n_states,
                                         axis=1)
        loss_weights_NN_type = np.hstack([np.ones(n_states), np.zeros(n_states), np.zeros(n_states)])
        loss_weights_np = loss_weights_combined * loss_weights_NN_type
        patience = 1000
    elif NN_type == 'dtNN':
        X_training, y_training = prepare_data(training_data_pure, n_states=n_states)
        sample_weights_static = np.split(training_data_pure['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(training_data_pure['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(np.ones(training_data_pure['data_type'].shape), indices_or_sections=n_states,
                                         axis=1)
        loss_weights_NN_type = np.hstack([np.ones(n_states), np.ones(n_states), np.zeros(n_states)])
        loss_weights_np = loss_weights_combined * loss_weights_NN_type
        patience = 1000
    elif NN_type == 'PINN':
        X_training, y_training = prepare_data(training_data, n_states=n_states)
        sample_weights_static = np.split(training_data['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(training_data['data_type'], indices_or_sections=n_states, axis=1) + \
                                np.split(np.ones(training_data['data_type'].shape), indices_or_sections=n_states,
                                         axis=1)
        loss_weights_NN_type = np.hstack([np.ones(n_states), np.ones(n_states), np.ones(n_states)])
        loss_weights_np = loss_weights_combined * loss_weights_NN_type
        patience = 2500

    else:
        raise Exception('Invalid NN_type.')

    # -----------------------------
    # validation set preparation
    # -----------------------------
    X_validation, y_validation = prepare_data(validation_data, n_states=n_states)
    validation_sample_weights = np.split(np.concatenate([np.ones((X_validation[0].shape[0], n_states)),
                                                         np.zeros((X_validation[0].shape[0], n_states)),
                                                         np.zeros((X_validation[0].shape[0], n_states))], axis=1),
                                         indices_or_sections=3*n_states, axis=1)

    # -----------------------------
    # final training preparation
    # -----------------------------
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate_initial,
        decay_rate=learning_rate_decay,
        decay_steps=100)

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler),
                  loss=[mse] * 3*n_states,
                  loss_weights=loss_weights_np.tolist(),
                  )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=patience, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )

    # -----------------------------
    # training without intermediate plotting
    # -----------------------------
    history_epochs = model.fit(X_training,
                               y_training,
                               initial_epoch=model.epoch_count,
                               epochs=model.epoch_count + epochs_total,
                               batch_size=int(X_training[0].shape[0]),
                               sample_weight=sample_weights_static,
                               validation_data=(X_validation, y_validation, validation_sample_weights),
                               validation_freq=1,
                               verbose=0,
                               shuffle=True,
                               callbacks=[early_stopping_callback, tensorboard_callback])
    model.epoch_count = model.epoch_count + epochs_total

    # -----------------------------
    # save model weights
    # -----------------------------
    model.set_weights(early_stopping_callback.best_weights)
    model.save_weights(filepath=directory_model_weights / f'weights_{simulation_id}.h5')

    # -----------------------------
    # store test data for detailed analyses, comment if summary statistics are sufficient
    # -----------------------------
    complete_data = model.update_test_data_with_prediction(test_data=complete_data)

    with open(directory_results / f'dataset_{simulation_id}.pickle', 'wb') as file_opener:
        pickle.dump(complete_data, file_opener)

    # -----------------------------
    # error analysis
    # -----------------------------
    error = np.square(complete_data['states_prediction'] - complete_data['states_results'])

    quantile_values = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                                0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995,
                                0.998, 0.999, 1.000])

    quantile_results = np.quantile(error, quantile_values, axis=0)
    mean_results = np.mean(error, axis=0)

    with open(directory_quantile / f'quantiles_{simulation_id}.pickle', 'wb') as file_opener:
        pickle.dump(quantile_results, file_opener)

    with open(directory_quantile / f'mean_{simulation_id}.pickle', 'wb') as file_opener:
        pickle.dump(mean_results, file_opener)

    # -----------------------------
    # console output after training
    # -----------------------------
    np.set_printoptions(precision=4)
    print(f'Simulation ID {simulation_id}:')
    print(f'MSE states    : {mean_results}')
    print(f'max SE states : {np.max(error, axis=0)}')
    pass


if __name__ == '__main__':
    train_model('testID', 'NN', 5, 9, 3215)
