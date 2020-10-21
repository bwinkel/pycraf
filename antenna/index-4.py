import numpy as np
import matplotlib.pyplot as plt
from pycraf.antenna import *
from astropy import units as u

phi = np.linspace(0, 21, 1000) * u.deg
diam = np.array([1, 2, 10])[:, np.newaxis] * u.m
wavlen = 0.03 * u.m  # about 10 GHz
G_max = fl_G_max_from_size(diam, wavlen)
gain = fl_pattern(phi, diam, wavlen, G_max)

plt.plot(phi, gain.T, '-')
plt.legend(['d=1 m', 'd=2 m', 'd=10 m'])
plt.xlim((0, 21))
plt.xlabel('Phi [deg]')
plt.ylabel('Gain [dBi]')
plt.show()