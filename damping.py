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


def damped_plots(t_list, y_list, v_list, e_list, set_name):
    plt.figure("position")
    plt.xlabel("time")
    plt.ylabel("position")
    plot_x_vs_y(t_list, 0, y_list, 0, set_name, None)
    plt.figure("velocity, {}".format(set_name))
    plt.xlabel("time")
    plt.ylabel("velocity")
    plot_x_vs_y(t_list, 0, v_list, 0, set_name, None)
    plt.figure("position vs velocity, {}".format(set_name))
    plt.xlabel("velocity")
    plt.ylabel("position")
    plot_x_vs_y(y_list, 0, v_list, 0, set_name, None)
    plt.figure("energy, {}".format(set_name))
    plt.xlabel("time")
    plt.ylabel("energy")
    plot_x_vs_y(t_list, 0, e_list, 0, set_name, None)


if __name__ == "__main__":
    # TODO: check params
    period = 0.720  # second
    mass = 210  # grams
    damping_factor = 0.03
    balanced_position = 0.233
    spring_constant = spring_const_calc(period, mass)
    print(spring_constant)

    time_list, y_data_list = np.loadtxt("SpringMass_posn_dampener_Oct31_Junyu_Gabrielle.txt",
                                        unpack=True, skiprows=2, usecols=(0, 1))
    v_data_list = np.loadtxt("SpringMass_velocity_dampener_Oct31_Junyu_Gabrielle.txt",
                             unpack=True, skiprows=2, usecols=1)
    e_data_list = energy_calculation(y_data_list, v_data_list, spring_constant, mass/1000)
    y_data_list = y_data_list / 100
    plt.figure("position")
    plot_x_vs_y(time_list, 0, y_data_list, 0, "data", None)
    # t_y_v_e_list = predict_damping_motion_forward_euler(y_data_list[0], 0, spring_constant, mass / 1000,
    # damping_factor, time_list) print(t_y_v_e_list) damped_plots(time_list, column_extractor(1, t_y_v_e_list,
    # time_list), column_extractor(2, t_y_v_e_list, time_list), column_extractor(3, t_y_v_e_list, time_list),
    # "forward euler")
    t_y_v_e_list = predict_damping_motion_symplectic_euler(y_data_list[0] - balanced_position, 0, spring_constant,
                                                           mass / 1000,
                                                           damping_factor,
                                                           time_list)
    damped_plots(time_list, np.array(column_extractor(1, t_y_v_e_list, time_list)) + balanced_position,
                 column_extractor(2, t_y_v_e_list, time_list),
                 column_extractor(3, t_y_v_e_list, time_list), "symplectic euler")
    plt.show()
