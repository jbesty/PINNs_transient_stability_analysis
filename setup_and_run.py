# !/usr/bin/env python3
import time
import hashlib
import pandas as pd
import numpy as np
import itertools
import pickle
import pathlib
import tensorflow as tf
import multiprocessing as mp
from train_model import train_model

tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)
tf.config.threading.set_intra_op_parallelism_threads(num_threads=1)

# -----------------------------
# Wrapper script for running multiple variations in parallel. The definition for each training is stored in a
# 'setup_table' and contains the unique simulation IDs to identify each training run.
# -----------------------------


def setup_and_run():
    tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads=1)

    # TODO: Define the path to store all relevant data
    # directory_data: pathlib.Path = pathlib.Path('Here_goes_your_path')
    raise Exception('Please specify directory_data, then delete this Exception.')

    current_time = int(time.time() * 1000)
    setup_id = hashlib.md5(str(current_time).encode())

    setup_id_path = directory_data / 'setup_tables' / f'setupID_{setup_id.hexdigest()}.pickle'

    setup_table_names = ['NN_type',
                         'data_points',
                         'seed_tensorflow']

    # to be usable in "itertools.product(*parameters)"
    NN_type = ['NN', 'dtNN', 'PINN']
    data_points = [[5, 5], [5, 9], [5, 21], [9, 9], [13, 9], [25, 41]]
    np.random.seed(94589)
    seed_tensorflow = np.random.randint(0, 1000000, 20).tolist()

    parameters = [NN_type,
                  data_points,
                  seed_tensorflow]

    setup_table = pd.DataFrame(itertools.product(*parameters), columns=setup_table_names)

    setup_table.insert(0, "setupID", setup_id.hexdigest())
    n_power_steps_training = [pair[0] for pair in setup_table['data_points']]
    n_time_steps_training = [pair[1] for pair in setup_table['data_points']]
    setup_table.insert(loc=3, column="n_power_steps_training", value=n_power_steps_training)
    setup_table.insert(loc=4, column="n_time_steps_training", value=n_time_steps_training)

    simulation_ids_unhashed = current_time + 1 + setup_table.index.values
    simulation_ids = []
    for simulation_id in simulation_ids_unhashed:
        simulation_ids_hashed = hashlib.md5(str(simulation_id).encode())
        simulation_ids.append(simulation_ids_hashed.hexdigest())

    setup_table.insert(1, "simulation_id", simulation_ids)

    with open(setup_id_path, "wb") as f:
        pickle.dump(setup_table, f)

    print('Created setup table with %i entries' % setup_table.shape[0])

    starmap_variables = [(simulation_id, NN_type, n_power_steps_training, n_time_steps_training, seed_tensorflow) for
                         (simulation_id, NN_type, n_power_steps_training, n_time_steps_training, seed_tensorflow) in
                         zip(setup_table['simulation_id'], setup_table['NN_type'],
                             setup_table['n_power_steps_training'],
                             setup_table['n_time_steps_training'], setup_table['seed_tensorflow'])]

    with mp.Pool(20) as pool:
        pool.starmap(train_model, starmap_variables)

    pass


if __name__ == '__main__':
    setup_and_run()
