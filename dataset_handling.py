import numpy as np

# -----------------------------
# Functions for handling the dataset, namely filtering, dividing, and preparing it.
# -----------------------------


def filter_dataset(dataset, filter_indices):
    dataset_copy = dataset.copy()
    for key in dataset_copy.keys():
        dataset_copy[key] = dataset_copy[key][filter_indices, :]

    return dataset_copy


def divide_dataset(dataset,
                   n_power_steps_training,
                   n_time_steps_training):
    power_min = 0.0
    power_max = 6.0
    time_min = 0.0
    time_max = 2.0
    bus_disturbance = 4

    n_power_steps_collocation = 25
    n_time_steps_collocation = 41

    # ------ data points including initial conditions on collocation trajectories  -----------------
    linspace_power_steps = np.around(np.linspace(power_min, power_max, n_power_steps_training), decimals=5)
    linspace_time_steps = np.around(np.linspace(time_min, time_max, n_time_steps_training), decimals=5)

    indices_power_steps = np.isin(np.around(dataset['power'][:, bus_disturbance], decimals=5), linspace_power_steps)
    indices_time_steps = np.isin(np.around(dataset['time'][:, 0], decimals=5), linspace_time_steps)

    linspace_power_steps_initial = np.around(np.linspace(power_min, power_max, n_power_steps_collocation), decimals=5)

    indices_power_steps_initial = np.isin(np.around(dataset['power'][:, bus_disturbance], decimals=5),
                                          linspace_power_steps_initial)
    indices_time_steps_initial = np.isin(np.around(dataset['time'][:, 0], decimals=5), 0)
    filter_indices_training_initial = np.logical_and(indices_power_steps_initial, indices_time_steps_initial)

    filter_indices_training = np.logical_or(np.logical_and(indices_power_steps, indices_time_steps),
                                            filter_indices_training_initial)

    # ------ collocation points -----------------
    linspace_power_steps = np.around(np.linspace(power_min, power_max, n_power_steps_collocation), decimals=5)
    linspace_time_steps = np.around(np.linspace(time_min, time_max, n_time_steps_collocation), decimals=5)

    indices_power_steps = np.isin(np.around(dataset['power'][:, bus_disturbance], decimals=5), linspace_power_steps)
    indices_time_steps = np.isin(np.around(dataset['time'][:, 0], decimals=5), linspace_time_steps)
    filter_indices_collocation = np.logical_and(np.logical_and(indices_power_steps, indices_time_steps),
                                                np.logical_not(filter_indices_training))

    # ------ validation data -----------------
    linspace_power_steps = np.around(np.linspace(power_min, power_max, n_power_steps_collocation) + 0.10000, decimals=5)
    linspace_time_steps = np.around(np.linspace(time_min, time_max, n_time_steps_collocation) + 0.02400, decimals=5)

    indices_power_steps = np.isin(np.around(dataset['power'][:, bus_disturbance], decimals=5), linspace_power_steps)
    indices_time_steps = np.isin(np.around(dataset['time'][:, 0], decimals=5), linspace_time_steps)
    filter_indices_validation = np.logical_and(indices_power_steps, indices_time_steps)

    # ------ testing data points -----------------
    linspace_power_steps = np.around(np.linspace(power_min, power_max, 61), decimals=5)
    linspace_time_steps = np.around(np.linspace(time_min, time_max, 201), decimals=5)

    indices_power_steps = np.isin(np.around(dataset['power'][:, bus_disturbance], decimals=5), linspace_power_steps)
    indices_time_steps = np.isin(np.around(dataset['time'][:, 0], decimals=5), linspace_time_steps)
    filter_indices_testing = np.logical_and(indices_power_steps, indices_time_steps)

    if sum(filter_indices_training) != n_power_steps_training * (n_time_steps_training - 1) + n_power_steps_collocation:
        raise Exception('Error in training data filtering')
    else:
        print(f'Filtered {sum(filter_indices_training)} training data points.')

    if sum(filter_indices_collocation) != n_power_steps_collocation * n_time_steps_collocation - sum(
            filter_indices_training):
        raise Exception('Error in collocation data filtering')
    else:
        print(f'Filtered {sum(filter_indices_collocation)} collocation data points.')

    if sum(filter_indices_validation) != (n_power_steps_collocation - 1) * (n_time_steps_collocation - 1):
        raise Exception('Error in validation data filtering')
    else:
        print(f'Filtered {sum(filter_indices_validation)} validation data points.')

    if sum(filter_indices_testing) != 61 * 201:
        raise Exception('Error in test data filtering')
    else:
        print(f'Filtered {sum(filter_indices_testing)} test data points.')

    training_data_pure = filter_dataset(dataset=dataset, filter_indices=filter_indices_training)
    collocation_data = filter_dataset(dataset=dataset, filter_indices=filter_indices_collocation)
    validation_data = filter_dataset(dataset=dataset, filter_indices=filter_indices_validation)
    testing_data = filter_dataset(dataset=dataset, filter_indices=filter_indices_testing)

    collocation_data['data_type'] = collocation_data['data_type'] * 0

    training_data = training_data_pure.copy()
    for key in training_data.keys():
        training_data[key] = np.concatenate([training_data_pure[key],
                                             collocation_data[key]],
                                            axis=0)

    return training_data, training_data_pure, validation_data, testing_data


def prepare_data(dataset, n_states):
    X_dataset = [dataset['time'],
                 dataset['power']]

    y_dataset = np.split(dataset['states_results'], indices_or_sections=n_states, axis=1) + \
                np.split(dataset['states_t_results'], indices_or_sections=n_states, axis=1) + \
                [np.zeros((dataset['states_initial'].shape[0], 1))] * n_states

    return X_dataset, y_dataset
