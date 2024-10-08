#!/usr/bin/python
from scipy.optimize import curve_fit, brentq, fmin
from matplotlib import pyplot
import numpy as np


def get_na_triple_gaussian_fit():
    """Calculate numerical aperture based on measured power upon angle."""
    angle, amplitude = read_data('pomiar.txt')

    angle_min, angle_max, angle_center, center_width, min_ampl, max_ampl = get_data_parameters(
        angle, amplitude)

    initial_params = [max_ampl / 3, angle_center, center_width] * 3  # triple times repeated
    initial_params.append(min_ampl)
    fit_params, _ = curve_fit(gauss_triple_amp, angle, amplitude, initial_params)
    amplitude_fit = gauss_triple_amp(angle, *fit_params)
    amplitude_fit_gauss1, amplitude_fit_gauss2, amplitude_fit_gauss3, amplitude_fit_gauss4 = get_component_gauss(
        angle, fit_params)

    # before calculating aperture, subtract the baseline
    # if amplitude values at edges differs, make average
    fit_params_corrected_baseline, baseline = correct_baseline(fit_params, amplitude_fit)
    max_ampl_fit = gauss_triple_amp(
        fmin(lambda x: -1 * gauss_triple_amp(x, *fit_params_corrected_baseline), 90),
        *fit_params_corrected_baseline)
    percentage_criteria = [1, 5, 13.5]
    na = []  # numerical aperture

    # find given power drop and calculate aperture:
    for value in percentage_criteria:
        angle_left = brentq(get_amplitude_difference, angle_min, angle_center, args=(
            gauss_triple_amp, fit_params_corrected_baseline, max_ampl_fit, value), xtol=1e-06)
        angle_right = brentq(get_amplitude_difference, angle_center, angle_max, args=(
            gauss_triple_amp, fit_params_corrected_baseline, max_ampl_fit, value), xtol=1e-06)
        na.append(np.sin(np.pi / 180 * (angle_right - angle_left) / 2))

    print(na)
    plot_data(angle, amplitude, amplitude_fit, amplitude_fit_gauss1, amplitude_fit_gauss2,
              amplitude_fit_gauss3, amplitude_fit_gauss4)
    save_data_gauss(na, angle, amplitude, amplitude_fit, fit_params, 'data_triple_gauss.txt',
                    'NA_triple_gauss.txt', 'fit_params.txt')


def save_data_gauss(na, angle, amplitude, amplitude_fit, fit_params, filename_data, filename_na,
                    filename_fit_params):
    """Save measurement and fit data and calculated numerical aperture to files."""
    with open(filename_na, 'w', encoding="utf-8") as data_file:
        for line in na:
            data_file.writelines(f"{line} \n")

    with open(filename_data, 'w', encoding="utf-8") as data_file:
        for an, amp, amp_fit in zip(angle, amplitude, amplitude_fit):
            data_file.writelines(f"{an} \t  {amp} \t {amp_fit} \n")

    with open(filename_fit_params, 'w', encoding="utf-8") as data_file:
        data_file.writelines(
            "a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2) + a3*exp(-((x-b3)/c3)^2) + y0 \n \n")
        for line in fit_params:
            data_file.writelines(f"{line} \n")


def gauss_triple_amp(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, y0):
    """Calculate gaussian-type function value at given x for given parameters."""
    return a1 * np.exp(- ((x - b1) / c1) ** 2) + a2 * np.exp(- ((x - b2) / c2) ** 2) + a3 * np.exp(
        - ((x - b3) / c3) ** 2) + y0


def get_component_gauss(angle, fit_params):
    """Get compositional gauss functions."""
    amplitude_fit_gauss1 = gauss_triple_amp(angle, *fit_params[0:3], *np.array([0, 0, 1]),
                                            *np.array([0, 0, 1]), 0)
    amplitude_fit_gauss2 = gauss_triple_amp(angle, *np.array([0, 0, 1]), *fit_params[3:6],
                                            *np.array([0, 0, 1]), 0)
    amplitude_fit_gauss3 = gauss_triple_amp(angle, *np.array([0, 0, 1]), *np.array([0, 0, 1]),
                                            *fit_params[-4:-1], 0)
    amplitude_fit_gauss4 = gauss_triple_amp(angle, *np.array([0, 0, 1]), *np.array([0, 0, 1]),
                                            *np.array([0, 0, 1]), fit_params[-1])

    return amplitude_fit_gauss1, amplitude_fit_gauss2, amplitude_fit_gauss3, amplitude_fit_gauss4


def get_amplitude_difference(x, gauss_function, fit_params, max_ampl, percent):
    """Get difference between function value and given amplitude factor. """
    return gauss_function(x, *fit_params) - percent / 100 * max_ampl


def read_data(filename):
    """Read measurement data from file."""
    with open(filename, 'r', encoding="utf-8") as f:
        angle, amplitude = [], []

        for line in f.readlines():
            measurement = line.split()
            if len(measurement) < 2:
                continue
            angle.append(float(measurement[0]))
            amplitude.append(float(measurement[1]))

    return angle, amplitude


def get_data_parameters(angle, amplitude):
    """Get parameters of the gaussian-type data."""
    angle_min = min(angle)
    angle_max = max(angle)
    min_ampl = min(amplitude)
    angle_center = angle[amplitude.index(max_ampl := max(amplitude))]
    center_width = np.abs(
        angle[np.abs([ampl - 0.5 * max_ampl for ampl in amplitude]).argmin()] - angle_center) * 2

    return angle_min, angle_max, angle_center, center_width, min_ampl, max_ampl


def plot_data(angle, amplitude, amplitude_fit, amplitude_fit_gauss1, amplitude_fit_gauss2,
              amplitude_fit_gauss3, amplitude_fit_gauss4):
    """Plot measurement and fit data."""
    pyplot.scatter(angle, amplitude)
    pyplot.plot(angle, amplitude_fit, 'r-')
    pyplot.plot(angle, amplitude_fit_gauss1, 'b--')
    pyplot.plot(angle, amplitude_fit_gauss2, 'g--')
    pyplot.plot(angle, amplitude_fit_gauss3, 'm--')
    pyplot.plot(angle, amplitude_fit_gauss4, 'k--')
    pyplot.show()


def correct_baseline(fit_params, amplitude_fit):
    """Correct baseline in fit parameters."""
    middle_index = len(amplitude_fit) // 2
    min_ampl_fit_left = min(amplitude_fit[0:middle_index])
    min_ampl_fit_right = min(amplitude_fit[middle_index:-1])
    baseline = (min_ampl_fit_left + min_ampl_fit_right) / 2
    fit_params_corrected_baseline = np.array([*fit_params[0:-1], fit_params[-1] - baseline])
    return fit_params_corrected_baseline, baseline


if __name__ == "__main__":
    get_na_triple_gaussian_fit()
