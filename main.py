import os
import numpy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from Analysis_Module import *


def initial_conditions(file: str):
    distances = np.loadtxt(file, unpack=True,
                           usecols=1, skiprows=2)
    y_0 = list(distances)[0]
    v_0 = 0
    return [y_0, v_0]


def initial_plot(file: str):
    time = np.loadtxt(file, unpack=True,
                      usecols=0, skiprows=2)
    distances = np.loadtxt(file, unpack=True,
                           usecols=1, skiprows=2)
    plt.plot(time, distances, marker="o", ls='',
             label="Raw Data")
    plt.show()


def spring_const_calc(period: float, mass_grams: float,
                      mass_error: float, peri_error: float):
    m = mass_grams / 1000
    m_err = mass_error / 1000
    twopi = 2 * np.pi
    denom = (period / twopi) ** 2
    k = m / denom

    period_square_error = error_prop_multiplication(period**2, [[period, peri_error], [period, peri_error]])
    denom_error = (1/twopi**2)*period_square_error
    k_err = error_prop_multiplication(k, [[m, m_err], [denom, denom_error]])

    return k, k_err


def time_array(disp_t: float, t_not: float, t_total: float):
    time_values = np.arange(t_not, t_total, disp_t)

    return time_values


def time_step_posn(y_init: float, v_init: float, disp_t: float):
    y_plus_1 = y_init + disp_t * v_init

    return y_plus_1


def fe_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                          disp_t: float):
    v_plus_1 = v_init - disp_t * (ang_freq_osc ** 2) * y_init
    return v_plus_1


def symplectic_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                                  disp_t: float):
    v_plus_1 = v_init - disp_t * (ang_freq_osc ** 2) * y_init
    return v_plus_1


def forward_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                        disp_t: float, t_not: float,
                        t_total: float, mass: float, spring_const: float):
    # TODO: add uncertainties
    mass = mass/1000
    y_init = y_init/100
    t_pos_vel_energ_array = []
    times = list(time_array(disp_t, t_not, t_total))
    print(times)

    E_init = (1 / 2) * spring_const * y_init ** 2

    init_parameters = [times[0], y_init, v_init, E_init]

    t_pos_vel_energ_array.append(init_parameters)

    y_plus_1 = y_init
    v_plus_1 = v_init

    for i in np.arange(len(times)):
        y_i = y_plus_1
        y_plus_1 = time_step_posn(y_i, v_plus_1,
                                  disp_t)
        v_plus_1 = fe_time_step_velocity(y_i,
                                         v_plus_1, ang_freq_osc,
                                         disp_t)
        print(v_plus_1)
        E = (1 / 2) * mass * v_plus_1 ** 2 + (1 / 2) * spring_const * y_plus_1 ** 2
        t_pos_vel_energ_array.append([times[i] + disp_t, y_plus_1, v_plus_1, E])

    return t_pos_vel_energ_array


def symplectic_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                           disp_t: float, t_not: float,
                           t_total: float, mass: float, spring_const: float, y_err, mass_err, k_err):
    t_pos_vel_energ_array = []
    t_pos_vel_energ_uncertainty_array = []
    times = list(time_array(disp_t, t_not, t_total))
    mass = mass/1000
    y_init = y_init/100
    E_init = (1 / 2) * spring_const * y_init ** 2

    v_err = 0
    init_parameters = [times[0], y_init, v_init, E_init]

    t_pos_vel_energ_array.append(init_parameters)
    t_pos_vel_energ_uncertainty_array.append([0, y_err, v_err, 0])
    y_plus_1 = y_init
    v_plus_1 = v_init

    for i in np.arange(len(times)):
        y_err = error_prop_addition([y_err, disp_t * v_err])
        y_plus_1 = time_step_posn(y_plus_1, v_plus_1,
                                  disp_t)
        v_err = error_prop_addition([v_err, error_prop_multiplication(disp_t * (ang_freq_osc ** 2) * y_plus_1,
                                                                      [[mass, mass_err], [spring_const, k_err],
                                                                       [y_plus_1, y_err]])])
        v_plus_1 = symplectic_time_step_velocity(y_plus_1,
                                                 v_plus_1, ang_freq_osc,
                                                 disp_t)
        E = (1 / 2) * mass * v_plus_1 ** 2 + (1 / 2) * spring_const * y_plus_1 ** 2
        e_err = error_prop_addition([error_prop_multiplication((1 / 2) * mass * v_plus_1 ** 2,
                                                               [[mass, mass_err], [v_plus_1, v_err],
                                                                [v_plus_1, v_err]]),
                                     error_prop_multiplication((1 / 2) * spring_const * y_plus_1 ** 2,
                                                               [[spring_const, k_err], [y_plus_1, y_err],
                                                                [y_plus_1, y_err]])])
        t_pos_vel_energ_array.append([times[i] + disp_t, y_plus_1, v_plus_1, E])
        t_pos_vel_energ_uncertainty_array.append([0, y_err, v_err, e_err])

    return t_pos_vel_energ_array, t_pos_vel_energ_uncertainty_array


def quick_plot(x: list, y: list):
    plt.plot(x, y, marker="o", ls='')
    plt.show()


def column_extractor(columnnum: int, original_array, time_list):
    column = []
    for i in np.arange(len(time_list)):
        column.append(original_array[i][columnnum])
    return column


if __name__ == '__main__':
    initial_plot("SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")

    # 2. Define Constants

    # Just a note here - when you see a variable with "nd" at the end it means
    # "no dampener"

    disp_t = 0.01
    t_not = 0
    t_total = 10
    period_nd = 0.68
    period_error = 0.0005
    mass_nd = 200.2
    # TODO: check the errors and calculate k_err
    y_error = 0.000001
    m_error = 0.05

    k_nd, k_error = spring_const_calc(period_nd, mass_nd, m_error, period_error)
    print(k_nd, k_error)
    angular_freq_osc_nd = np.sqrt(k_nd / (mass_nd / 1000))

    t_i = time_array(disp_t, t_not, t_total)

    y_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[0]
    v_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[0]
    print(y_0)

    forward_euler_mass_nd_array = forward_euler_array(y_0 / 100, 0, angular_freq_osc_nd,
                                                      disp_t, t_not,
                                                      t_total, mass_nd, k_nd)

    symplectic_euler_mass_nd_array, symplectic_euler_uncertainty_array = \
        symplectic_euler_array(y_0 / 100, 0, angular_freq_osc_nd, disp_t, t_not, t_total, mass_nd, k_nd, y_error,
                               m_error, k_error)

    # print(generated_mass_nd_array)

    fe_ys = column_extractor(1, forward_euler_mass_nd_array, t_i)
    print(fe_ys)

    # for i in np.arange(len(t_i)):
    #    fe_ys.append(forward_euler_mass_nd_array[i][1])

    fe_vs = column_extractor(2, forward_euler_mass_nd_array, t_i)

    # for i in np.arange(len(t_i)):
    #    fe_vs.append(forward_euler_mass_nd_array[i][2])

    fe_Es = column_extractor(3, forward_euler_mass_nd_array, t_i)
    # for i in np.arange(len(t_i)):
    #    fe_Es.append(forward_euler_mass_nd_array[i][3])
    print(symplectic_euler_uncertainty_array[:100])
    sy_ys = column_extractor(1, symplectic_euler_mass_nd_array, t_i)
    sy_vs = column_extractor(2, symplectic_euler_mass_nd_array, t_i)
    sy_Es = column_extractor(3, symplectic_euler_mass_nd_array, t_i)
    quick_plot(t_i, fe_ys)
    quick_plot(t_i, fe_vs)
    quick_plot(fe_ys, fe_vs)
    quick_plot(t_i, fe_Es)

    quick_plot(t_i, sy_ys)
    quick_plot(t_i, sy_vs)
    quick_plot(sy_ys, sy_vs)
    quick_plot(t_i, sy_Es)
