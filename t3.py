from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import math

import numpy as np

def fourier_analysis(filename, s, c, show=0):
    f = np.fromfile(filename, dtype=np.int32)
    g = apply_gaussian_windowing(f, s)

    plot_signal(f)
    plot_signal(g)


def apply_gaussian_windowing(f, s):
    window = signal.gaussian(len(f), std=s)
    return f*window


def plot_signal(f):
    plt.plot(f)
    plt.show()

fourier_analysis(filename="file", s=100, c=2)