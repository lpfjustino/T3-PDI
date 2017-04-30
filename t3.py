from scipy import signal
from scipy.fftpack import fft, ifft, fftshift
import matplotlib.pyplot as plt
import math

import numpy as np

# Performs the Fourier Analysis of the data in a given binary file
def fourier_analysis(filename, s, c, show=0):
    f = np.fromfile(filename, dtype=np.int32)
    g = apply_gaussian_windowing(f, s)

    # Number of observations
    n = len(f)

    # Normalized Fourier Spectrum for f and g
    half_obs = math.floor(n/2)
    F = fft(f)[:half_obs]
    G = fft(g)[:half_obs]
    nfs_f = (abs(F) / (2*n))[:half_obs]
    nfs_g = (abs(G) / (2*n))[:half_obs]

    # Showing comparison of the functions and their respective spectrum
    if show == 1:
     plot_comparison(f, g, nfs_f, nfs_g)

    # Filtering the spectrum
    F_hat = low_pass(F, G, c)

    # Computing the inverse FFT for the filtered spectrum
    f_hat = ifft(F_hat)

    # Showing comparison of the function f and its filtered version
    if show == 1:
        plot_comparison(f, f_hat)

    print(np.argmax(F))
    print(np.argmax(nfs_g))
    print(np.max(f))
    print(np.max(np.real(f_hat)))


# Applies Gaussian Windowing of length s on f
def apply_gaussian_windowing(f, s):
    n = len(f)
    window = signal.gaussian(n, std=n/s)
    return f*window


# Shows a m by n subplot (where m = cols) of given arguments
def plot_comparison(*plots, cols=2):
    n = len(plots)
    rows = math.ceil(n/cols)

    _, ax_arr = plt.subplots(rows, cols)

    for (i, plot) in enumerate(plots):
        row = math.floor(i/cols)
        col = i - (row * cols)

        if n > cols: # More than 1 line, ax_arr is a matrix
            ax_arr[row, col].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax_arr[row, col].plot(np.real(plot))
        else: # ax_arr is an array
            ax_arr[col].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax_arr[col].plot(np.real(plot))

    plt.show()


# Applies a low pass filter on f using g (which is f after applying gaussian
# windowing) and constant c to calculate the threshold
def low_pass(F, G, c):
    # Computing the threshold
    T = c * np.argmax(abs(G))

    F_hat = []

    for (i,v) in enumerate(F):
        if v >= T:
            F_hat.append(0)
        else:
            F_hat.append(v)

    return F_hat


fourier_analysis(filename="sound1.bin", s=5, c=2, show=1)