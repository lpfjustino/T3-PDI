from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import math

import numpy as np

def fourier_analysis(filename, s, c, show=0):
    f = np.fromfile(filename, dtype=np.int32)
    g = apply_gaussian_windowing(f, s)

    n = len(f)

    nfs_f = abs(fft(f)) / (2*n)
    nfs_g = abs(fft(g)) / (2*n)

    plot_comparison(f, g, nfs_f, nfs_g)

def apply_gaussian_windowing(f, s):
    n = len(f)
    window = signal.gaussian(n, std=n/s)
    return f*window


def plot_comparison(*plots, cols=2):
    n = len(plots)
    rows = math.ceil(n/cols)

    _, ax_arr = plt.subplots(rows, cols)

    for (i, plot) in enumerate(plots):
        row = math.floor(i/cols)
        col = i - (row * cols)

        if n > cols: # More than 1 line, ax_arr is a matrix
            ax_arr[row, col].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax_arr[row, col].plot(plot)
        else: # ax_arr is an array
            ax_arr[col].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax_arr[col].plot(plot)

    plt.show()

fourier_analysis(filename="sound1.bin", s=5, c=2, show=1)