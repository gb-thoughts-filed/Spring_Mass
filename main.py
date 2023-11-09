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
    k = m/(period/2*np.pi)**2

    return k


def time_array(disp_t: float, t_not: float, t_total: float):

    time_values = np.arange(t_not, t_total, disp_t)

    return time_values


def time_step_posn(y_init: float, v_init: float, disp_t: float):
    y_plus_1 = y_init + disp_t*v_init
    return y_plus_1


def time_step_velocity(y_init: float, v_init: float, ang_freq_osc: float,
                       disp_t: float):

    v_plus_1 = v_init - disp_t*(ang_freq_osc**2)*y_init
    return v_plus_1


def generated_loop_array(y_init: float, v_init: float, ang_freq_osc: float,
                         disp_t: float, t_not: float, t_total: float):
    t_pos_vel_array = []
    times = list(time_array(disp_t, t_not, t_total))

    init_parameters = [times[0], y_init, v_init]

    t_pos_vel_array.append(init_parameters)

    for i in np.arange(len(times)):
        y_plus_1 = time_step_posn(t_pos_vel_array[i][1], t_pos_vel_array[i][2],
                                  disp_t)
        v_plus_1 = time_step_velocity(t_pos_vel_array[i][1],
                                      t_pos_vel_array[i][2], ang_freq_osc,
                                      disp_t)
        t_pos_vel_array.append([times[i]+disp_t, y_plus_1, v_plus_1])

    return t_pos_vel_array


def quick_plot(x: list, y: list):

    plt.plot(x, y, marker="o", ls='')
    plt.show()






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
    angular_freq_osc_nd = np.sqrt(k_nd/mass_nd)

    t_i = time_array(disp_t, t_not, t_total)

    y_0 = initial_conditions(
        "SpringMass_posn_onlymass_Oct31_Junyu_Gabrielle.txt")[0]

    generated_mass_nd_array = generated_loop_array(y_0, 0, angular_freq_osc_nd,
                                                   disp_t, t_not, t_total)

    # print(generated_mass_nd_array)

    ys = []

    for i in np.arange(len(t_i)):
        ys.append(generated_mass_nd_array[i][1])

    vs = []

    for i in np.arange(len(t_i)):
        vs.append(generated_mass_nd_array[i][2])


    quick_plot(t_i, ys)
    quick_plot(t_i, vs)
    quick_plot(vs, ys)


