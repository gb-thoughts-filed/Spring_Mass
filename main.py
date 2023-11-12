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

    # period_square_error = error_prop_multiplication(period**2, [[period, peri_error], [period, peri_error]])
    period_square_error = error_prop_exponent(period, peri_error, period**2, 2)
    denom_error = (1/twopi**2)*period_square_error
    k_err = error_prop_multiplication(k, [[m, m_err], [denom, denom_error]])

    return k, k_err


def time_array(disp_t: float, t_not: float, t_total: float):
    time_values = np.arange(t_not, t_total, disp_t)

    return time_values


def time_step_posn(y_init: float, v_init: float, disp_t: float, y_err_init: float, v_error_init):
    y_plus_1 = y_init + disp_t * v_init
    disp_t_velocity_err = disp_t*v_error_init
    y_err = error_prop_addition([y_err_init, disp_t_velocity_err])
    return y_plus_1, y_err


def fe_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                          disp_t: float, y_err_init: float, v_error_init, mass, spring_const, mass_err, k_err):

    # mass not adjusted here as it's already adjusted in forward euler array function

    # velocity step calculation
    v_plus_1 = v_init - disp_t * (ang_freq_osc ** 2) * y_init

    # velocity error calculations
    one_divide_mass_err = error_prop_exponent(mass, mass_err, 1/mass, -1)
    ang_freq_osc_square_err = error_prop_multiplication((ang_freq_osc ** 2), [[1/mass, one_divide_mass_err], [spring_const, k_err]])

    yi_ang_freq_osc_square_err = error_prop_multiplication(
        (ang_freq_osc ** 2)*y_init, [[(ang_freq_osc ** 2), ang_freq_osc_square_err],[y_init, y_err_init]])

    v_err = error_prop_addition([v_error_init, yi_ang_freq_osc_square_err])

    return v_plus_1, v_err


def symplectic_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                                  disp_t: float):
    v_plus_1 = v_init - disp_t * (ang_freq_osc ** 2) * y_init
    return v_plus_1


def forward_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                        disp_t: float, t_not: float,
                        t_total: float, mass: float, spring_const: float, y_err, mass_err, k_err):
    # TODO: add uncertainties
    mass = mass/1000
    mass_err = mass_err/1000
    y_init = y_init/100
    t_pos_vel_energ_array = []
    t_pos_vel_energ_uncertainty_array = []
    times = list(time_array(disp_t, t_not, t_total))
    # print(times)
    v_error_init = 0

    # Initial Energy Calculation
    E_init = (1 / 2) * spring_const * y_init ** 2

    #Initial Energy Error Calculation
    # y_init_square_err = error_prop_multiplication(y_init**2, [[y_init, y_err], [y_init, y_err]])
    y_init_square_err = error_prop_exponent(y_init, y_err, y_init**2, 2)
    E_error_init = (1/2)*error_prop_multiplication(spring_const * y_init ** 2, [[spring_const, k_err], [y_init ** 2, y_init_square_err]])

    init_parameters = [times[0], y_init, v_init, E_init]
    init_parameters_uncertainty = [0, y_err, v_error_init, E_error_init]

    t_pos_vel_energ_array.append(init_parameters)
    t_pos_vel_energ_uncertainty_array.append(init_parameters_uncertainty)

    for i in np.arange(len(times)):
        y_plus_1, y_plus_1_err = time_step_posn(t_pos_vel_energ_array[i][1], t_pos_vel_energ_array[i][2],
                                  disp_t, t_pos_vel_energ_uncertainty_array[i][1], t_pos_vel_energ_uncertainty_array[i][2])
        v_plus_1, v_plus_1_err = fe_time_step_velocity(t_pos_vel_energ_array[i][1],
                                         t_pos_vel_energ_array[i][2], ang_freq_osc,
                                         disp_t, t_pos_vel_energ_uncertainty_array[i][1], t_pos_vel_energ_uncertainty_array[i][2],
                                            mass, spring_const, mass_err, k_err)
        # print(v_plus_1)

        E = (1 / 2) * mass * v_plus_1 ** 2 + (1 / 2) * spring_const * y_plus_1 ** 2

        # Energy Error Calculation
        v_plus_1_square_err = error_prop_multiplication(v_plus_1 ** 2, [[v_plus_1, v_plus_1_err], [v_plus_1, v_plus_1_err]])
        y_plus_1_square_err = error_prop_multiplication(y_plus_1 ** 2, [[y_plus_1, y_plus_1_err], [y_plus_1, y_plus_1_err]])
        kin_eneg_err = (1/2)*error_prop_multiplication(mass * v_plus_1 ** 2, [[mass, mass_err],[v_plus_1 ** 2, v_plus_1_square_err]])
        pot_eneg_err = (1/2)*error_prop_multiplication(spring_const * y_plus_1 ** 2,
                                                       [[spring_const, k_err],[y_plus_1 ** 2, y_plus_1_square_err]])
        E_error = error_prop_addition([kin_eneg_err, pot_eneg_err])

        t_pos_vel_energ_array.append([times[i] + disp_t, y_plus_1, v_plus_1, E])
        t_pos_vel_energ_uncertainty_array.append([0, y_plus_1_err, v_plus_1_err, E_error])

    return t_pos_vel_energ_array, t_pos_vel_energ_uncertainty_array


def symplectic_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                           disp_t: float, t_not: float,
                           t_total: float, mass: float, spring_const: float, y_err, mass_err, k_err):
    t_pos_vel_energ_array = []
    t_pos_vel_energ_uncertainty_array = []
    times = list(time_array(disp_t, t_not, t_total))
    mass = mass/1000
    m_err = mass_err/1000
    y_init = y_init/100
    E_init = (1 / 2) * spring_const * y_init ** 2

    v_err = 0
    init_parameters = [times[0], y_init, v_init, E_init]

    t_pos_vel_energ_array.append(init_parameters)
    t_pos_vel_energ_uncertainty_array.append([0, y_err, v_err, 0])

    for i in np.arange(len(times)):
        y_err = error_prop_addition([y_err, disp_t * v_err])
        y_plus_1, y_err_extra = time_step_posn(t_pos_vel_energ_array[i][1], t_pos_vel_energ_array[i][2],
                                  disp_t, t_pos_vel_energ_uncertainty_array[i][1], t_pos_vel_energ_uncertainty_array[i][2])
        v_err = error_prop_addition([v_err, error_prop_multiplication(disp_t * (ang_freq_osc ** 2) * y_plus_1,
                                                                      [[mass, m_err], [spring_const, k_err],
                                                                       [y_plus_1, y_err]])])
        v_plus_1 = symplectic_time_step_velocity(y_plus_1,
                                                 t_pos_vel_energ_array[i][2], ang_freq_osc,
                                                 disp_t)
        E = (1 / 2) * mass * v_plus_1 ** 2 + (1 / 2) * spring_const * y_plus_1 ** 2
        e_err = error_prop_addition([error_prop_multiplication((1 / 2) * mass * v_plus_1 ** 2,
                                                               [[mass, m_err], [v_plus_1, v_err],
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


def full_plot(x: list, y:list , y_error: list, x_name: str, y_name: str, title: str):

    plt.errorbar(x, y, yerr=y_error, marker="o", ls="")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()
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
    # print(k_nd, k_error)

    angular_freq_osc_nd = np.sqrt(k_nd / (mass_nd / 1000))

    t_i = time_array(disp_t, t_not, t_total)

    y_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[0]
    v_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[1]
    # print(y_0)

    forward_euler_mass_nd_array, forward_euler_mass_nd_uncertainty_array = forward_euler_array(
        y_0, 0, angular_freq_osc_nd, disp_t, t_not, t_total, mass_nd, k_nd, y_error, m_error, k_error)

    symplectic_euler_mass_nd_array, symplectic_euler_uncertainty_array = \
        symplectic_euler_array(y_0, 0, angular_freq_osc_nd, disp_t, t_not, t_total, mass_nd, k_nd, y_error,
                               m_error, k_error)

    # print(forward_euler_mass_nd_uncertainty_array)

    fe_ys = column_extractor(1, forward_euler_mass_nd_array, t_i)
    fe_ys_uncertainty = column_extractor(1, forward_euler_mass_nd_uncertainty_array, t_i)
    print(fe_ys_uncertainty)
    print(fe_ys)


    fe_vs = column_extractor(2, forward_euler_mass_nd_array, t_i)
    fe_vs_uncertainty = column_extractor(2, forward_euler_mass_nd_uncertainty_array, t_i)

    fe_Es = column_extractor(3, forward_euler_mass_nd_array, t_i)
    fe_Es_uncertainty = column_extractor(3, forward_euler_mass_nd_uncertainty_array, t_i)

    # print(symplectic_euler_uncertainty_array[:100])
    sy_ys = column_extractor(1, symplectic_euler_mass_nd_array, t_i)
    sy_ys_uncertainty = column_extractor(1, symplectic_euler_uncertainty_array, t_i)

    sy_vs = column_extractor(2, symplectic_euler_mass_nd_array, t_i)
    sy_Es = column_extractor(3, symplectic_euler_mass_nd_array, t_i)

    full_plot(t_i, fe_ys, fe_ys_uncertainty, "Time (s)", "Position of Spring (m)",
              "Time(s) v.s. Position of Spring (m) Approximated Using Forward Euler Method")
    full_plot(t_i, fe_vs, fe_vs_uncertainty, "Time (s)", "Velocity of Spring (m)",
              "Time(s) v.s. Velocity of Spring (m) Approximated Using Forward Euler Method")
    quick_plot(fe_ys, fe_vs)
    full_plot(t_i, fe_Es, fe_Es_uncertainty, "Time (s)", "Energy of Spring (J)",
              "Time(s) v.s. Energy of Spring (J) Approximated Using Forward Euler Method")
    quick_plot(t_i, fe_Es)

    full_plot(t_i, sy_ys, sy_ys_uncertainty, "Time (s)", "Position of Spring (m)",
              "Time(s) v.s. Position of Spring (m) Approximated Using Symplectic Euler Method")
    quick_plot(t_i, sy_ys)
    quick_plot(t_i, sy_vs)
    quick_plot(sy_ys, sy_vs)
    quick_plot(t_i, sy_Es)

# full_plot(x: list, y:list , y_error: list, x_name: str, y_name: str, title: str)
