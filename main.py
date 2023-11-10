import os
import numpy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


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


def spring_const_calc(period: float, mass_grams: float):

    m = mass_grams/1000
    twopi = 2*np.pi
    denom = (period/twopi)**2
    k = m/denom

    return k


def time_array(disp_t: float, t_not: float, t_total: float):

    time_values = np.arange(t_not, t_total, disp_t)

    return time_values


def time_step_posn(y_init: float, v_init: float, disp_t: float):
    y_plus_1 = y_init + disp_t*v_init

    return y_plus_1


def fe_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                          disp_t: float):

    v_plus_1 = v_init - disp_t*(ang_freq_osc**2)*y_init
    return v_plus_1


def symplectic_time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                          disp_t: float):

    v_plus_1 = v_init - disp_t*(ang_freq_osc**2)*y_init
    return v_plus_1


def forward_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                        disp_t: float, t_not: float,
                        t_total: float, mass: float, spring_const: float):
    t_pos_vel_energ_array = []
    times = list(time_array(disp_t, t_not, t_total))

    E_init = (1/2)*spring_const*y_init**2

    init_parameters = [times[0], y_init, v_init, E_init]

    t_pos_vel_energ_array.append(init_parameters)

    for i in np.arange(len(times)):
        y_plus_1 = time_step_posn(t_pos_vel_energ_array[i][1], t_pos_vel_energ_array[i][2],
                                  disp_t)
        v_plus_1 = fe_time_step_velocity(t_pos_vel_energ_array[i][1],
                                         t_pos_vel_energ_array[i][2], ang_freq_osc,
                                         disp_t)
        E = (1/2)*mass*v_plus_1**2 + (1/2)*spring_const*y_plus_1**2
        t_pos_vel_energ_array.append([times[i] + disp_t, y_plus_1, v_plus_1, E])

    return t_pos_vel_energ_array


def symplectic_euler_array(y_init: float, v_init: float, ang_freq_osc: float,
                        disp_t: float, t_not: float,
                        t_total: float, mass: float, spring_const: float):
    t_pos_vel_energ_array = []
    times = list(time_array(disp_t, t_not, t_total))

    E_init = (1/2)*spring_const*y_init**2

    init_parameters = [times[0], y_init, v_init, E_init]

    t_pos_vel_energ_array.append(init_parameters)

    for i in np.arange(len(times)):
        y_plus_1 = time_step_posn(t_pos_vel_energ_array[i][1], t_pos_vel_energ_array[i][2],
                                  disp_t)
        v_plus_1 = symplectic_time_step_velocity(y_plus_1,
                                         t_pos_vel_energ_array[i][2], ang_freq_osc,
                                         disp_t)
        E = (1/2)*mass*v_plus_1**2 + (1/2)*spring_const*y_plus_1**2
        t_pos_vel_energ_array.append([times[i] + disp_t, y_plus_1, v_plus_1, E])

    return t_pos_vel_energ_array


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

    #Just a note here - when you see a variable with "nd" at the end it means
    # "no dampener"

    disp_t = 0.01
    t_not = 0
    t_total = 10
    period_nd = 0.68
    mass_nd = 200.2
    k_nd = spring_const_calc(period_nd, mass_nd)
    print(k_nd)
    angular_freq_osc_nd = np.sqrt(k_nd/(mass_nd/1000))

    t_i = time_array(disp_t, t_not, t_total)

    y_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[0]
    print(y_0)

    forward_euler_mass_nd_array = forward_euler_array(y_0, 0, angular_freq_osc_nd,
                                                      disp_t, t_not,
                                                      t_total, k_nd, mass_nd)

    symplectic_euler_mass_nd_array = symplectic_euler_array(y_0, 0, angular_freq_osc_nd,
                                                      disp_t, t_not,
                                                      t_total, k_nd, mass_nd)

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


