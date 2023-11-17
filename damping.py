import matplotlib.pyplot as plt
import numpy as np
from main import *


def damped_velocity(position, velocity, angular_frequency, damping_coefficient, delta_t):
    return velocity - (position * (angular_frequency ** 2) + damping_coefficient * velocity) * delta_t


def energy_calculation(y, v, k, m):
    return (1 / 2) * m * (v ** 2) + (1 / 2) * k * (y ** 2)


def position_predict(y, v, delta_t):
    return y + delta_t * v


def predict_damping_motion_forward_euler(y_init, v_init, k, m, damping_coefficient, time_lst):
    time_delta = time_lst[1] - time_lst[0]
    t_y_v_e_prediction = [[time_lst[0], y_init, v_init, energy_calculation(y_init, v_init, k, m)]]
    for i in range(len(time_lst) - 1):
        v = damped_velocity(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], np.sqrt(k / m), damping_coefficient,
                            time_delta)
        y = position_predict(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], time_delta)
        e = energy_calculation(y, v, k, m)
        t_y_v_e_prediction.append([time_lst[i + 1], y, v, e])
    return t_y_v_e_prediction


def predict_damping_motion_symplectic_euler(y_init, v_init, k, m, damping_coefficient, time_lst):
    time_delta = time_lst[1] - time_lst[0]
    print(time_delta)
    t_y_v_e_prediction = [[time_lst[0], y_init, v_init, energy_calculation(y_init, v_init, k, m)]]
    for i in range(len(time_lst) - 1):
        y = position_predict(t_y_v_e_prediction[i][1], t_y_v_e_prediction[i][2], time_delta)
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
    print("chisq {}: {}".format(set_name, chi_sq))


def wave_y(t, zeta, ang_freq, phase, y0, a):
    return a * np.exp(-zeta * t) * np.cos(ang_freq * t + phase) + y0


def reynolds_number(density, velocity, characteristic_length, dynamic_viscosity):
    return density * velocity * characteristic_length / dynamic_viscosity


if __name__ == "__main__":
    time_list, y_data_list = np.loadtxt("SpringMass_posn_dampener_Oct31_Junyu_Gabrielle.txt",
                                        unpack=True, skiprows=2, usecols=(0, 1))
    t_delta = time_list[1] - time_list[0]
    time_list = time_list - time_list[0]
    y_data_list = y_data_list / 100
    v_data_list = np.loadtxt("SpringMass_velocity_dampener_Oct31_Junyu_Gabrielle.txt",
                             unpack=True, skiprows=2, usecols=1)
    plt.rcParams.update({'font.size': 13})
    v_data_list = v_data_list / 100

    period = 0.721  # second
    mass = 217.4  # grams
    damping_factor = 0.0372
    balanced_position = 0.233
    y_error = 0.293 / 100
    v_error = (y_error * np.sqrt(2)) / t_delta
    m_error = 0.05 / 1000
    v_0 = 0
    spring_constant, k11 = spring_const_calc(period, mass, 0, 0)
    mass = mass / 1000
    k_error = spring_constant * m_error / mass
    air_density = 1.293
    air_viscosity = 1.48 * (10 ** -5)
    print(spring_constant)

    e_data_list = energy_calculation(y_data_list, v_data_list, spring_constant, mass)
    e_data_error = error_prop_addition(
        [0.5 * error_prop_multiplication(mass * v_data_list ** 2, [[mass, m_error], [v_data_list ** 2, v_error]]),
         0.5 * error_prop_multiplication(spring_constant * y_data_list ** 2, [[spring_constant, k_error],
                                                                              [y_data_list ** 2, y_error]])]
    )
    print(e_data_error)
    t_y_v_e_list = predict_damping_motion_symplectic_euler(y_data_list[0] - balanced_position, v_0, spring_constant,
                                                           mass,
                                                           damping_factor,
                                                           time_list)
    plt.figure("position, Symplectic Euler")
    y_decay_list = (y_data_list[0] - balanced_position) * np.exp(-(damping_factor / 2) * time_list) + balanced_position
    plot_directional_graph(time_list, (y_data_list[0] - balanced_position) * np.exp(-(damping_factor / 2) * time_list) +
                           balanced_position, r"Amplitude Decay Model $A(t)=A_0e^{-\frac{\gamma}{2}t}$", 0)
    damped_plots(time_list, np.array(column_extractor(1, t_y_v_e_list, time_list)) + balanced_position,
                 column_extractor(2, t_y_v_e_list, time_list),
                 column_extractor(3, t_y_v_e_list, time_list), y_data_list, y_error, "Symplectic Euler")

    t_y_v_e_list = predict_damping_motion_forward_euler(y_data_list[0], 0, spring_constant, mass,
                                                        damping_factor, time_list)

    damped_plots(time_list, np.array(column_extractor(1, t_y_v_e_list, time_list)) + balanced_position,
                 column_extractor(2, t_y_v_e_list, time_list),
                 column_extractor(3, t_y_v_e_list, time_list), y_data_list, y_error, "Forward Euler")

    plt.figure("best fit")
    popt, pcov, prediction = plot_x_vs_y(time_list, 0, y_data_list, np.full_like(y_data_list, y_error),
                                         "best fit",
                                         wave_y)
    plt.legend()

    plt.figure("energy, Symplectic Euler")
    energy_data_list = energy_calculation(y_data_list - balanced_position, v_data_list, spring_constant,
                                          mass)
    plot_x_vs_y(time_list, 0, energy_data_list, 0, "energy from data", None)
    energy_decay_list = energy_data_list[0] * np.exp(-damping_factor * time_list)
    plot_directional_graph(time_list, energy_decay_list, r"energy decay calculated with $E(t)=E_0e^{-\gamma t}$", 0)
    plt.legend()
    plt.savefig("Figs/energy, Symplectic Euler", dpi=600)
    print(popt)
    print(np.mean(y_data_list))

    max_disp = []
    max_disp_t = []
    period_int = int(np.floor(period / (time_list[1] - time_list[0])))
    for i in range(12):
        max_disp_y = max(y_data_list[i * period_int: (i + 1) * period_int])
        max_disp.append(max_disp_y)
        max_disp_t.append(
            (i * period_int + np.where(y_data_list[i * period_int: (i + 1) * period_int] == max_disp_y)[0][0]))
    if y_data_list[-1] not in max_disp:
        max_disp.append(y_data_list[-1])
        max_disp_t.append(len(time_list) - 1)
    print(max_disp)
    print(max_disp_t)
    max_disp_decay = [y_decay_list[i] for i in max_disp_t]
    plt.figure("max displacement residual")
    plt.title(
        r"Difference Between Maximum Displacement and Predicted Amplitude of the Decay Model $A(t)=A_0e^{-\frac{\gamma}{2}t}$",
        wrap=True)
    plt.xlabel("Time(s)")
    plt.ylabel("Difference(m)")
    print(plot_residual(np.array(max_disp_t) / 100, np.array(max_disp), y_error, np.array(max_disp_decay),
                        "", 2))
    plt.tight_layout(pad=1.4)
    plt.savefig("Figs\max displacement residual")

    plt.figure("Reynolds number")
    plt.title("Reynolds number of the oscillation over time")
    plt.ylabel("Reynolds number")
    plt.xlabel("Time(s)")
    print(v_data_list)
    print(y_decay_list)
    plt.plot(time_list,
             reynolds_number(air_density, np.abs(v_data_list), 2 * (y_decay_list - balanced_position), air_viscosity),
             label="Reynolds number of the oscillation")
    plt.plot(time_list, np.full_like(time_list, 2300), label="Re = 2300")
    plt.legend()
    plt.savefig("Figs/Reynolds number", dpi=600)
    plt.show()
