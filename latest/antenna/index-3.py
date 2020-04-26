import numpy as np
import matplotlib.pyplot as plt
from pycraf.antenna import ras_pattern
import astropy.units as u


phi = np.linspace(0, 20, 1000) * u.deg
diam = np.array([10, 50, 100]) * u.m
gain = ras_pattern(phi, diam[:, np.newaxis], 0.21 * u.m)
plt.plot(phi, gain.T, '-')
plt.legend(['d=10 m', 'd=50 m', 'd=100 m'])
plt.xlim((0, 2.8))
plt.xlabel('Phi [deg]')
plt.ylabel('Gain [dBi]')
plt.show()

# zoom-in with Bessel correction
phi = np.linspace(0, 2.8, 10000) * u.deg
gain = ras_pattern(phi, 100 * u.m, 0.21 * u.m, do_bessel=True)
plt.plot(phi, gain, 'k-')
plt.xlim((0, 2.8))
plt.xlabel('Phi [deg]')
plt.ylabel('Gain [dBi]')
plt.show()