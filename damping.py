import matplotlib.pyplot as plt
import numpy as np
from main import *


def damped_velocity(position, velocity, angular_frequency, damping_coefficient, delta_t):
    return velocity - (position * (angular_frequency ** 2) + damping_coefficient * velocity) * delta_t


def energy_calculation(y, v, k, m):
    return (1 / 2) * m * (v ** 2) + (1 / 2) * k * (y ** 2)


def predict_damping_motion_forward_euler(y_init, v_init, k, m, damping_coefficient, time_lst):
    time_delta = time_lst[1] - time_lst[0]
    t_y_v_e_prediction = [[time_lst[0], y_init, v_init, energy_calculation(y_init, v_init, k, m)]]
    for i in range(len(time_lst) - 1):
        v = damped_velocity(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], np.sqrt(k / m), damping_coefficient,
                            time_delta)
        y = time_step_posn(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], time_delta)
        e = energy_calculation(y, v, k, m)
        t_y_v_e_prediction.append([time_lst[i + 1], y, v, e])
    return t_y_v_e_prediction


def predict_damping_motion_symplectic_euler(y_init, v_init, k, m, damping_coefficient, time_lst):
    time_delta = time_lst[1] - time_lst[0]
    print(time_delta)
    t_y_v_e_prediction = [[time_lst[0], y_init, v_init, energy_calculation(y_init, v_init, k, m)]]
    for i in range(len(time_lst) - 1):
        y = time_step_posn(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], time_delta)
        v = damped_velocity(y, t_y_v_e_prediction[i][2], np.sqrt(k / m), damping_coefficient,
                            time_delta)
        e = energy_calculation(y, v, k, m)
        t_y_v_e_prediction.append([time_lst[i + 1], y, v, e])
    return t_y_v_e_prediction


def damped_plots(t_list, y_list, v_list, e_list, y_data, y_err, set_name):
    plt.figure("position, {}".format(set_name))
    plt.title(
        "Harmonic Oscillation with Dampener Time(s) vs. Position of Mass (m) Prediction of {} Method and Raw Data "
        "Recorded".format(
            set_name), wrap=True)
    plt.xlabel("time(s)")
    plt.ylabel("position(m)")
    plot_x_vs_y(t_list, 0, y_data, y_err, "raw data", None)
    plot_directional_graph(t_list, y_list, set_name, 0)
    plt.legend()
    plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2)
    plt.savefig("Figs/position, {}".format(set_name), dpi=600)

    plt.figure("velocity, {}".format(set_name))
    plt.title(
        "Harmonic Oscillation with Dampener Time(s) vs. velocity (m/s) Prediction of {} Method".format(
            set_name), wrap=True)
    plt.xlabel("time(s)")
    plt.ylabel("velocity(m/s)")
    plot_directional_graph(t_list, v_list, set_name, 0)
    plt.legend()
    plt.savefig("Figs/velocity, {}".format(set_name), dpi=600)

    plt.figure("position vs velocity, {}".format(set_name))
    plt.gca().set_box_aspect(aspect=1)
    plt.title(
        "Harmonic Oscillation with Dampener position(m) vs. velocity (m/s) Prediction of {} Method".format(
            set_name), wrap=True)
    plt.xlabel("velocity(m/s)")
    plt.ylabel("position(m)")
    plot_directional_graph(y_list, v_list, "position vs velocity {}".format(set_name), 6)
    plt.savefig("Figs/position vs velocity, {}".format(set_name), dpi=600)

    plt.figure("energy, {}".format(set_name))
    plt.title(
        "Harmonic Oscillation with Dampener Energy(J) vs. velocity (m/s) Prediction of {} Method".format(
            set_name), wrap=True)
    plt.xlabel("time(s)")
    plt.ylabel("energy(J)")
    plot_directional_graph(t_list, e_list, set_name, 0)
    plt.savefig("Figs/energy, {}".format(set_name), dpi=600)

    plt.figure("residual, {}".format(set_name))
    plt.title(
        "Harmonic Oscillation with Dampener, Prediction of {} Method Residual".format(
            set_name), wrap=True)
    plt.xlabel("time(s)")
    plt.ylabel("position(m)")
    chi_sq = plot_residual(t_list, y_data, y_err,
                           y_list,
                           set_name, predict_damping_motion_symplectic_euler)
    plt.savefig("Figs/residual, {}".format(set_name), dpi=600)
    print(chi_sq)


def wave_y(t, zeta, ang_freq, phase, y0, a):
    return a * np.exp(-zeta * t) * np.cos(ang_freq * t + phase) + y0


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 13})
    # TODO: check params
    period = 0.7205  # second
    mass = 217.4  # grams
    damping_factor = 0.034
    balanced_position = 0.233
    y_error = 0.293 / 100
    v_0 = 0
    spring_constant = spring_const_calc(period, mass)
    print(spring_constant)

    time_list, y_data_list = np.loadtxt("SpringMass_posn_dampener_Oct31_Junyu_Gabrielle.txt",
                                        unpack=True, skiprows=2, usecols=(0, 1))
    v_data_list = np.loadtxt("SpringMass_velocity_dampener_Oct31_Junyu_Gabrielle.txt",
                             unpack=True, skiprows=2, usecols=1)
    e_data_list = energy_calculation(y_data_list, v_data_list, spring_constant, mass / 1000)
    y_data_list = y_data_list / 100
    t_y_v_e_list = predict_damping_motion_symplectic_euler(y_data_list[0] - balanced_position, v_0, spring_constant,
                                                           mass / 1000,
                                                           damping_factor,
                                                           time_list)
    damped_plots(time_list, np.array(column_extractor(1, t_y_v_e_list, time_list)) + balanced_position,
                 column_extractor(2, t_y_v_e_list, time_list),
                 column_extractor(3, t_y_v_e_list, time_list), y_data_list, y_error, "Symplectic Euler")

    t_y_v_e_list = predict_damping_motion_forward_euler(y_data_list[0], 0, spring_constant, mass / 1000,
                                                        damping_factor, time_list)

    damped_plots(time_list, np.array(column_extractor(1, t_y_v_e_list, time_list)) + balanced_position,
                 column_extractor(2, t_y_v_e_list, time_list),
                 column_extractor(3, t_y_v_e_list, time_list), y_data_list, y_error, "Forward Euler")

    plt.figure("best fit")
    popt, pcov, prediction = plot_x_vs_y(time_list - 0.530, 0, y_data_list, np.full_like(y_data_list, y_error),
                                         "best fit",
                                         wave_y)
    plt.legend()
    plt.figure("data_energy")
    plot_directional_graph(time_list, energy_calculation(y_data_list, v_data_list, spring_constant, mass),"energy data, 0")

    print(popt)
    plt.show()
