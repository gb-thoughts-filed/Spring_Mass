import numpy as np
from matplotlib import pyplot as plt


def histogram_plot(data, rounded_range, interval):
    plt.xlabel("Number of Counts")
    plt.ylabel("Probability Density")
    # round the range to nearest 10 that in include the all the num in data.
    print(rounded_range, (rounded_range[1] - rounded_range[0]) / interval)
    plt.hist(data, bins=int((rounded_range[1] - rounded_range[0]) / interval),
             density=True,
             range=rounded_range
             , edgecolor='black', label="Data Recorded")


if __name__ == "__main__":
    sample_num_plate, position = np.loadtxt \
        ("SpringMass_posn_uncertainty_against_powersupply_Oct31_Junyu_Gabrielle.txt", skiprows=2,
         unpack=True, usecols=(0, 1))
    histogram_plot(position, (16.872, 16.910), 0.002)
    std_err = np.std(position) / np.sqrt(len(position))
    print(std_err)
    plt.show()
