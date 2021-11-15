import functools
import numpy as np
import pathlib
import pickle
from scipy import integrate
import time

from power_system_functions import create_system_matrices, compute_equilibrium_state, ode_right_hand_side_solve, create_power_system


# -----------------------------
# functions for simulating the specified trajectory and creating a dataset by creating data points on a pre-defined
# grid structure.
# Needs be run only once if the dataset is kept constant.
# -----------------------------

def input_data_initialised(n_ops, power_system):
    """
    Standard initialisation
    """
    time_zeros = np.zeros((n_ops, 1))
    power_zeros = np.zeros((n_ops, power_system['n_buses']))
    states_initial = np.zeros((n_ops, power_system['n_states']))

    states_results_zeros = np.zeros((n_ops, power_system['n_states']))
    states_t_results_zeros = np.zeros((n_ops, power_system['n_states']))
    data_type_zeros = np.zeros((n_ops, power_system['n_states']))

    data_initialised = {'time': time_zeros,
                        'power': power_zeros,
                        'states_initial': states_initial,
                        'states_results': states_results_zeros,
                        'states_t_results': states_t_results_zeros,
                        'data_type': data_type_zeros}

    return data_initialised


def create_training_data(n_time_steps, n_power_steps, power_min, power_max, t_settle_disturbance, t_short_circuit,
                         bus_disturbance, power_system):
    data_ops = input_data_initialised(n_ops=n_power_steps,
                                      power_system=power_system)
    power_ops = np.zeros((n_power_steps, power_system['n_buses']))
    power_ops[:, bus_disturbance] = np.linspace(power_min, power_max, n_power_steps)
    data_ops.update(time=np.ones((n_power_steps, 1)) * t_settle_disturbance,
                    power=power_ops)

    x_equilibrium_undisturbed = compute_equilibrium_state(power_system,
                                                          u_disturbance=None,
                                                          slack_bus=power_system['slack_bus_idx'],
                                                          system_case='normal')

    data_ops.update(states_initial=np.repeat(x_equilibrium_undisturbed.T, repeats=n_power_steps, axis=0),
                    data_type=np.ones((n_power_steps, power_system['n_states'])))

    data_ops = evaluate_ops(data_ops, 'normal', power_system)

    data_ops.update(states_initial=data_ops['states_results'],
                    time=np.ones((n_power_steps, 1)) * t_short_circuit)

    data_ops = evaluate_ops(data_ops, 'short_circuit', power_system)

    shorted_bus_angles = data_ops['states_results'][:, 9:10]
    shorted_bus_angle_offset = np.floor((shorted_bus_angles + np.pi) / (2 * np.pi)) * 2 * np.pi
    states_results = data_ops['states_results']
    states_results[:, 9:10] = states_results[:, 9:10] - shorted_bus_angle_offset

    t_max = power_system['t_max']
    data_ops.update(states_initial=states_results,
                    time=np.ones((n_power_steps, 1)) * t_max)

    start_time = time.time()
    data_ops = evaluate_op_trajectory(data_ops,
                                      n_time_steps=n_time_steps,
                                      system_case='line_tripped',
                                      power_system=power_system)
    print(time.time() - start_time)

    data_ops = calculate_data_ode_right_hand_side(data_ops, 'line_tripped', power_system)

    return data_ops


def calculate_data_ode_right_hand_side(data_ops, system_case, power_system):
    states_results = data_ops['states_results']
    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system, case=system_case)

    u_disturbance = data_ops['power'] @ G.T
    u = u_0.T + u_disturbance

    solver_func = functools.partial(ode_right_hand_side_solve, A=A, B=B, C=C, D=D, F=F)

    solver_results = map(solver_func,
                         data_ops['time'],
                         states_results,
                         u)

    list_solver_results = list(solver_results)

    states_t_results = np.concatenate([single_solver_result.reshape((1, -1)) for single_solver_result in
                                       list_solver_results],
                                      axis=0)

    data_ops.update(states_t_results=states_t_results)

    return data_ops


def evaluate_ops(data_ops, system_case, power_system):
    states_initial = data_ops['states_initial']
    t_span = np.concatenate([data_ops['time'] * 0,
                             data_ops['time']], axis=1)

    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system, case=system_case)

    u_disturbance = data_ops['power'] @ G.T
    u = u_0.T + u_disturbance
    solver_func = functools.partial(solve_ode, A=A, B=B, C=C, D=D, F=F)

    solver_results = map(solver_func,
                         t_span,
                         data_ops['time'],
                         states_initial,
                         u)

    list_solver_results = list(solver_results)

    states_results = np.concatenate([single_solver_result.T for single_solver_result in list_solver_results], axis=0)

    data_ops.update(states_results=states_results)

    return data_ops


def solve_ode(t_span,
              t_eval,
              states_initial,
              u, A, B, C, D, F):
    ode_solution = integrate.solve_ivp(ode_right_hand_side_solve,
                                       t_span=t_span,
                                       y0=states_initial.flatten(),
                                       args=[u, A, B, C, D, F],
                                       t_eval=t_eval,
                                       rtol=1e-5)

    return ode_solution.y


def evaluate_op_trajectory(data_ops, n_time_steps, system_case, power_system):
    n_ops = data_ops['time'].shape[0]
    t_max = power_system['t_max']

    t_span = np.concatenate([np.zeros(data_ops['time'].shape),
                             np.ones(data_ops['time'].shape) * t_max], axis=1)
    t_eval_vector = np.linspace(start=0, stop=t_max, num=n_time_steps).reshape((1, -1))
    t_eval = np.repeat(t_eval_vector, repeats=n_ops, axis=0)

    states_initial = data_ops['states_initial']
    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system, case=system_case)

    u_disturbance = data_ops['power'] @ G.T
    u = u_0.T + u_disturbance

    solver_func = functools.partial(solve_ode, A=A, B=B, C=C, D=D, F=F)

    solver_results = map(solver_func,
                         t_span,
                         t_eval,
                         states_initial,
                         u)

    list_solver_results = list(solver_results)

    states_results = np.concatenate([single_solver_result.T for single_solver_result in list_solver_results], axis=0)

    data_ops.update(time=t_eval.flatten().reshape((-1, 1)),
                    power=np.repeat(data_ops['power'], repeats=n_time_steps, axis=0),
                    states_initial=np.repeat(data_ops['states_initial'], repeats=n_time_steps, axis=0),
                    states_results=states_results,
                    data_type=np.repeat(data_ops['data_type'], repeats=n_time_steps, axis=0))

    return data_ops


if __name__ == '__main__':
    power_system = create_power_system()

    t_max = 2.0
    power_system['t_max'] = t_max
    t_settle_disturbance = 5.0
    n_time_steps = 1001
    n_power_steps = 121
    power_min = 0.0
    power_max = 6.0
    bus_disturbance = 4
    t_short_circuit = 0.05

    training_data = create_training_data(n_time_steps=n_time_steps,
                                         n_power_steps=n_power_steps,
                                         power_min=power_min,
                                         power_max=power_max,
                                         t_settle_disturbance=t_settle_disturbance,
                                         t_short_circuit=t_short_circuit,
                                         bus_disturbance=bus_disturbance,
                                         power_system=power_system)

    # TODO: Define the path to store all relevant data
    #
    # directory_data: pathlib.Path = pathlib.Path('Here_goes_your_path')
    raise Exception('Please specify directory_data, then delete this Exception.')

    with open(directory_data / 'datasets' / 'complete_dataset.pickle', 'wb') as f:
        pickle.dump(training_data, f)
