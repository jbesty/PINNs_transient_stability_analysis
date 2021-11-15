import numpy as np
from scipy.optimize import fsolve

# -----------------------------
# General functions that define the power system model and the state update equations, as well as functions used in
# the simulation of the trajectories.
# -----------------------------

def create_power_system():
    n_buses = 6
    n_generators = 4
    n_non_generators = n_buses - n_generators
    n_states = 2 * n_generators + 1 * n_non_generators

    omega_0 = 2 * np.pi * 60

    output_scaling = np.ones((n_states, 1))
    output_scaling[n_generators:2 * n_generators] = omega_0

    output_offset = np.zeros((n_states, 1))
    output_offset[n_generators:2 * n_generators] = -omega_0

    H_generators = np.array([58.5, 58.5, 55.575, 55.575])
    D_generators = 0.0 * np.ones(n_generators)

    D_non_generators = np.array([0.1, 0.2]) * 2

    P_load_set_point = np.array([-9.67, -17.67])
    P_generator_set_point = np.array([7, 7, 6.34, 7])
    P_set_point = np.hstack([P_generator_set_point, P_load_set_point])

    P_disturbance = np.zeros(n_buses)
    slack_bus_idx = 2

    V_magnitude = np.array([1.0300,
                            1.0100,
                            1.0300,
                            1.0100,
                            0.9610,
                            0.9714])

    # short circuit at bus 9
    V_magnitude_short_circuit = np.array([1.0300,
                                          1.0100,
                                          1.0300,
                                          1.0100,
                                          0.9610,
                                          0.000])

    B_susceptance = np.array([7.8461,
                              7.8461,
                              12.9499,
                              32.5581,
                              12.9499,
                              32.5581,
                              9.0982])

    # trip one line between bus 10 and 11 (line index 10), susceptance halfed
    B_susceptance_line_tripped = np.array([7.8461,
                                           7.8461,
                                           12.9499,
                                           32.5581,
                                           12.9499,
                                           32.5581,
                                           6.0655])

    b_from = np.array([0,
                       2,
                       0,
                       1,
                       2,
                       3,
                       4], dtype=int)

    b_to = np.array([1,
                     3,
                     4,
                     4,
                     5,
                     5,
                     5], dtype=int)

    V_i_V_j_B_full = V_magnitude[b_from] * V_magnitude[b_to] * B_susceptance
    V_i_V_j_B_short_circuit = V_magnitude_short_circuit[b_from] * V_magnitude_short_circuit[b_to] * B_susceptance
    V_i_V_j_B_line_tripped = V_magnitude[b_from] * V_magnitude[b_to] * B_susceptance_line_tripped

    incidence_matrix = np.array([[1, -1, 0, 0, 0, 0],
                                 [0, 0, 1, -1, 0, 0],
                                 [1, 0, 0, 0, -1, 0],
                                 [0, 1, 0, 0, -1, 0],
                                 [0, 0, 1, 0, 0, -1],
                                 [0, 0, 0, 1, 0, -1],
                                 [0, 0, 0, 0, 1, -1]])

    t_max = 2.0

    system_parameters = {'n_buses': n_buses,
                         'n_generators': n_generators,
                         'n_non_generators': n_non_generators,
                         'n_states': n_states,
                         'slack_bus_idx': slack_bus_idx,
                         'H_generators': H_generators,
                         'D_generators': D_generators,
                         'omega_0': omega_0,
                         'output_scaling': output_scaling,
                         'D_non_generators': D_non_generators,
                         'P_disturbance': P_disturbance,
                         'P_set_point': P_set_point,
                         'V_i_V_j_B_full': V_i_V_j_B_full,
                         'V_i_V_j_B_short_circuit': V_i_V_j_B_short_circuit,
                         'V_i_V_j_B_line_tripped': V_i_V_j_B_line_tripped,
                         'incidence_matrix': incidence_matrix,
                         't_max': t_max,
                         'output_offset': output_offset}

    print('Successfully created the reduced Kundur 2 area system (6 buses, 4 generators)!')

    return system_parameters


def create_system_matrices(power_system, case='normal'):

    n_g = power_system['n_generators']
    n_b = power_system['n_buses']
    n_d = n_b - n_g
    H_total = sum(power_system['H_generators'])

    # --------------------------------
    # A-matrix
    A_11 = np.zeros((n_g, n_g))
    A_12 = (np.eye(n_g) * H_total - np.repeat(power_system['H_generators'].reshape((1, n_g)), repeats=n_g,
                                              axis=0)) / H_total
    A_21 = np.zeros((n_g, n_g))
    A_22 = np.diag(-power_system['omega_0'] / (2 * power_system['H_generators']) * (
            power_system['D_generators'] + power_system['K_g_generators']))

    A_13 = np.zeros((n_g, n_d))
    A_23 = np.zeros((n_g, n_d))
    A_31 = np.zeros((n_d, n_g))
    A_32 = np.zeros((n_d, n_g))
    A_33 = np.zeros((n_d, n_d))

    A = np.block([
        [A_11, A_12, A_13],
        [A_21, A_22, A_23],
        [A_31, A_32, A_33]
    ])

    # --------------------------------
    # F-matrix
    F_11 = np.zeros((n_g, n_g))
    F_21 = np.diag(-power_system['omega_0'] / (2 * power_system['H_generators']))

    F_12 = np.zeros((n_g, n_d))
    F_22 = np.zeros((n_g, n_d))
    F_31 = np.zeros((n_d, n_g))
    F_32 = np.diag(-1 / power_system['D_non_generators'])

    F = np.block([
        [F_11, F_12],
        [F_21, F_22],
        [F_31, F_32]
    ])

    # --------------------------------
    # B-matrix
    # B_11 = -np.ones((n_g, 1))
    B_11 = np.zeros((n_g, 1))
    B_12 = np.zeros((n_g, n_g))
    B_21 = np.reshape(power_system['omega_0'] / (2 * power_system['H_generators']) * power_system[
        'K_g_generators'], (n_g, 1))
    B_22 = np.diag(power_system['omega_0'] / (2 * power_system['H_generators']))

    B_13 = np.zeros((n_g, n_d))
    B_23 = np.zeros((n_g, n_d))
    B_31 = np.zeros((n_d, 1))
    B_32 = np.zeros((n_d, n_g))
    B_33 = np.diag(1 / power_system['D_non_generators'])

    B = np.block([
        [B_11, B_12, B_13],
        [B_21, B_22, B_23],
        [B_31, B_32, B_33]
    ])

    # --------------------------------
    # U-matrix
    U_11 = np.eye(n_g)
    U_12 = np.zeros((n_g, n_g))
    U_13 = np.zeros((n_g, n_d))

    U_21 = np.zeros((n_d, n_g))
    U_22 = np.zeros((n_d, n_g))
    U_23 = np.eye(n_d)

    U = np.block([
        [U_11, U_12, U_13],
        [U_21, U_22, U_23]
    ])

    C = power_system['incidence_matrix'] @ U

    if case == 'normal':
        D = power_system['incidence_matrix'].T @ np.diag(power_system['V_i_V_j_B_full'])
    elif case == 'short_circuit':
        D = power_system['incidence_matrix'].T @ np.diag(power_system['V_i_V_j_B_short_circuit'])
    elif case == 'line_tripped':
        D = power_system['incidence_matrix'].T @ np.diag(power_system['V_i_V_j_B_line_tripped'])
    else:
        raise Exception('Specify a valid case')

    # adjustment of u to accommodate power disturbance input
    G = np.block([
        [np.zeros((1, n_b))],
        [np.eye(n_b)]
    ])

    # set point of the power before any disturbance
    u_0 = np.hstack([power_system['omega_0'],
                     power_system['P_set_point'][:n_g] + power_system['D_generators'] * power_system['omega_0'],
                     power_system['P_set_point'][n_g:]]).reshape((-1, 1))

    # initial value for equilibrium computation
    x_0 = np.hstack([np.zeros(n_g),
                     np.ones(n_g) * power_system['omega_0'],
                     np.zeros(n_d)]).reshape((-1, 1))

    return A, B, C, D, F, G, u_0, x_0


def compute_equilibrium_state(power_system, u_disturbance=None, slack_bus=None, system_case='normal'):
    A, B, C, D, F, G, u_0, x_0 = create_system_matrices(power_system=power_system, case=system_case)

    if u_disturbance is not None:
        u = u_0 + u_disturbance
    else:
        u = u_0

    if system_case == 'short_circuit':
        raise Exception('No equilibrium will be found for short circuit configurations.')

    x_equilibrium, info_dict, ier, mesg = fsolve(ode_right_hand_side,
                                                 x0=x_0,
                                                 args=(u, A, B, C, D, F, slack_bus),
                                                 xtol=1.49012e-08,
                                                 full_output=True)

    if not np.allclose(info_dict['fvec'],
                       np.zeros(info_dict['fvec'].shape),
                       atol=1e-08):
        raise Exception(f'No equilibrium found. Error message {mesg}')
    else:
        return x_equilibrium.reshape((-1, 1))


def ode_right_hand_side(x, u, A, B, C, D, F, slack=None):
    x_vector = np.reshape(x, (-1, 1))
    if slack is not None:
        x_vector[slack] = 0

    FCX = D @ np.sin(C @ x_vector)

    dx = A @ x_vector + F @ FCX + B @ u
    return dx[:, 0]


def ode_right_hand_side_solve(t, x, u, A, B, C, D, F):
    x_vector = np.reshape(x, (-1, 1))
    u_vector = np.reshape(u, (-1, 1))

    FCX = D @ np.sin(C @ x_vector)

    dx = A @ x_vector + F @ FCX + B @ u_vector
    return dx[:, 0]