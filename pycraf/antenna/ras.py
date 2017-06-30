#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'ras_pattern',
    ]


@helpers.ranged_quantity_input(
    phi=(-180, 180, apu.deg),
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    eta_a=(0, None, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dBi
    )
def ras_pattern(
        phi, diameter, wavelength, eta_a=100. * apu.percent, do_bessel=False
        ):
    '''
    Antenna gain as a function of angular distance after ITU-R RA.1631.

    Parameters
    ----------
    phi - angular distance in degrees
    diameter - antenna diameter
    wavelength - observing wavelength
    do_bessel - if set to True, use Bessel function approx. for inner 1 deg
    eta_a - antenna efficiency (default: 100%)

    Returns
    -------
    antenna gain in dBi

    Notes
    -----
    See ITU-R RA.1631 for explanation and applicability of this model.


    Example
    -------

    from pycraf.antenna import *
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as apu

    phi = np.linspace(0, 20, 1000) * apu.deg
    diam = np.array([10, 50, 100])[:, np.newaxis] * apu.m
    gain = ras_pattern(phi, diam, 0.21 * apu.m)
    plt.plot(phi, gain.T, '-')
    plt.show()

    phi = np.linspace(0, 2, 10000) * apu.deg
    gain = ras_pattern(phi, 100 * apu.m, 0.21 * apu.m, do_bessel=True)
    plt.plot(phi, gain, '-')
    plt.show()
    '''

    phi = np.abs(phi)
    # eta_a = eta_a / 100.

    # the following are independent on phi, no need to compute after broadcast
    d_wlen = diameter / wavelength

    # Note, we use the version that accounts for antenna efficiency
    # see ITU radio astronomy handbook, page 50
    # gmax = 20 * np.log10(np.pi * d_wlen)
    gmax = 10 * np.log10(eta_a * (np.pi * d_wlen) ** 2)

    g1 = -1. + 15. * np.log10(d_wlen)
    phi_m = 20. / d_wlen * np.sqrt(gmax - g1)
    phi_r = 15.85 * d_wlen ** -0.6

    # note automatic broadcasting should be possible
    # _tmp = np.broadcast(phi, _diam, _wlen)
    (
        phi, d_wlen, gmax, g1, phi_m, phi_r,
        ) = np.broadcast_arrays(
        phi, d_wlen, gmax, g1, phi_m, phi_r,
        )
    gain = np.zeros(phi.shape, np.float64)

    # case 1:
    mask = (0 <= phi) & (phi < phi_m)
    gain[mask] = gmax[mask] - 2.5e-3 * (d_wlen[mask] * phi[mask]) ** 2

    # case 2:
    mask = (phi_m <= phi) & (phi < phi_r)
    gain[mask] = g1[mask]

    # case 3:
    mask = (phi_r <= phi) & (phi < 10.)
    gain[mask] = 29. - 25. * np.log10(phi[mask])

    # case 4:
    mask = (10. <= phi) & (phi < 34.1)
    gain[mask] = 34. - 30. * np.log10(phi[mask])

    # case 5:
    mask = (34.1 <= phi) & (phi < 80.)
    gain[mask] = -12.

    # case 6:
    mask = (80. <= phi) & (phi < 120.)
    gain[mask] = -7.

    # case 7:
    mask = (120. <= phi) & (phi <= 180.)
    gain[mask] = -12.

    if do_bessel:

        from scipy.special import j1

        phi_0 = 69.88 / d_wlen
        x_pi = np.radians(np.pi / 2. * d_wlen * phi)

        # case 1:
        mask = (0 <= phi) & (phi < phi_0)
        tmp_x = x_pi[mask]
        gain[mask] = gmax[mask] + 20 * np.log10(j1(2 * tmp_x) / tmp_x)

        # case 2:
        mask = (phi_0 <= phi) & (phi < 1.)
        B_sqrt = 10 ** 1.6 * np.radians(np.pi * d_wlen[mask] / 2.)
        tmp_x = x_pi[mask]
        gain[mask] = 20 * np.log10(
            B_sqrt * np.cos(2 * tmp_x - 0.75 * np.pi + 0.0953) / tmp_x
            )

    return gain


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
